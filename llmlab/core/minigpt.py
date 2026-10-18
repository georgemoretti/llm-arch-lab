"""Мини-GPT — эталоны задач A15–A17 (блок A, модуль 4 «Свой GPT»).

Задачи A16–A18 импортируют готовые части отсюда, а не из кода пользователя: ошибка в A15 не должна ломать
следующие задачи. Совпадение функций и методов с ``solution.py`` задач проверяет ``tests/test_core_minigpt.py``.

Имена слоёв — как в ``llmlab/core/gpt2.py`` (wte, wpe, ln_1, attn, ln_2, mlp.c_fc, mlp.c_proj, ln_f, lm_head):
в C00 мини-GPT превращается в GPT-2 заменой одноголового attention на многоголовое.
Обозначения: B — batch, T — длина, C — n_embd, V — словарь, D — размер головы.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- A15: эмбеддинги


class TokenPositionEmbedding(nn.Module):
    """Эмбеддинг токена + обучаемый эмбеддинг позиции: [B, T] → [B, T, C]."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)   # токены
        self.wpe = nn.Embedding(block_size, n_embd)   # позиции 0 … block_size − 1
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [B, T] (long) → [B, T, C]; T > block_size → ValueError."""
        T = idx.shape[1]
        if T > self.block_size:
            raise ValueError(f"длина {T} больше block_size = {self.block_size}: для позиции {T - 1} нет эмбеддинга")
        pos = torch.arange(T, device=idx.device)
        return self.wte(idx) + self.wpe(pos)        # [B, T, C] + [T, C] → broadcast по батчу


def tie_weights(lm_head: nn.Linear, wte: nn.Embedding) -> None:
    """Связать выходной слой с эмбеддингами: один и тот же тензор весов [V, C] в обоих модулях."""
    lm_head.weight = wte.weight


def count_params(model: nn.Module) -> int:
    """Число обучаемых чисел модели; связанные веса считаются один раз."""
    return sum(p.numel() for p in model.parameters())


class EmbeddingLM(nn.Module):
    """Языковая модель без attention: (токен + позиция) → выходной слой, связанный с эмбеддингами."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.emb = TokenPositionEmbedding(vocab_size, block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        tie_weights(self.lm_head, self.emb.wte)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx [B, T] → (logits [B, T, V], loss — средняя cross-entropy по всем B·T позициям или None)."""
        logits = self.lm_head(self.emb(idx))        # [B, T, V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss


# ---------------------------------------------------------------- A16: одноголовое attention


def causal_mask(T: int) -> torch.Tensor:
    """Маска [T, T] типа bool: True там, куда позиции разрешено смотреть (сама и прошлые)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """Scaled dot-product attention: q, k [B, T, D], v [B, T, Dv] → (out [B, T, Dv], weights [B, T, T])."""
    D = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)          # [B, T, T]: строка — запрос, столбец — ключ
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))      # до softmax: запрещённые ключи получат вес 0
    weights = F.softmax(scores, dim=-1)                       # каждая строка суммируется в 1
    return weights @ v, weights


class AttentionHead(nn.Module):
    """Одна голова causal self-attention: [B, T, C] → [B, T, head_size]."""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", causal_mask(block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        out, _ = attention(self.query(x), self.key(x), self.value(x), self.mask[:T, :T])
        return out


# ---------------------------------------------------------------- A17: блок трансформера


class MLP(nn.Module):
    """FFN как в GPT-2: C → 4C, GELU (tanh-аппроксимация), 4C → C."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """Блок трансформера: attention и MLP с остаточными связями; LayerNorm до подслоя (pre-LN) или после (post-LN)."""

    def __init__(self, n_embd: int, block_size: int, pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = AttentionHead(n_embd, n_embd, block_size)   # одна голова на весь n_embd (много голов — C00)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:                                # GPT-2 и почти все современные LLM
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:                                            # исходный Transformer (2017)
            x = self.ln_1(x + self.attn(x))
            x = self.ln_2(x + self.mlp(x))
        return x
