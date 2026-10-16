"""Базовая модель GPT-2 — отправная точка блока C.

Все задачи блока C описывают, «что поменялось относительно GPT-2», поэтому этот файл
специально простой и читаемый: никаких оптимизаций, только архитектура.

Ключевые свойства GPT-2 (Radford et al., 2019), которые позже заменят:

- позиции: обучаемые абсолютные эмбеддинги ``wpe``            → RoPE (Llama), NoPE;
- нормализация: ``LayerNorm`` с bias, Pre-Norm                  → RMSNorm, QK-Norm, sandwich;
- attention: Multi-Head, у каждой головы свои K и V            → GQA/MQA, MLA, SWA, linear;
- FFN: Linear → GELU → Linear, расширение ×4                     → SwiGLU, MoE;
- bias во всех Linear                                            → без bias (Llama и др.);
- выходной слой связан с эмбеддингами (weight tying).

Размеры реального GPT-2 small (config.json на HF, ``openai-community/gpt2``):
vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12.
В тестах курса используются игрушечные размеры.

Обозначения форм тензоров во всём курсе:
B — batch, T — длина последовательности, C — d_model, H — число голов, D — C // H.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024  # максимальная длина контекста (n_positions)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    layer_norm_eps: float = 1e-5

    @classmethod
    def tiny(cls, **overrides) -> "GPT2Config":
        """Игрушечная конфигурация для тестов: обучается и прогоняется за миллисекунды."""
        base = dict(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_embd=32)
        return cls(**{**base, **overrides})


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention (реализуется пользователем в задаче C00)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd должно делиться на n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # Одна матрица сразу для Q, K и V: [C] → [3C]. Так было в оригинальном GPT-2.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Выходная проекция: склеенные головы [C] → [C].
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Нижнетреугольная маска: позиция t видит только позиции ≤ t.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        q, k, v = self.c_attn(x).split(C, dim=-1)              # 3 × [B, T, C]
        # [B, T, C] → [B, T, H, D] → [B, H, T, D]: каждая голова — отдельный «канал» внимания.
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Оценки сходства: [B, H, T, D] @ [B, H, D, T] → [B, H, T, T].
        # Делим на √D, чтобы дисперсия логитов не росла с размером головы.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)                      # по ключам: строка суммируется в 1
        weights = self.attn_dropout(weights)

        out = weights @ v                                        # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)     # склеиваем головы обратно
        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    """FFN GPT-2: расширение ×4, GELU (tanh-аппроксимация, как в оригинале), сжатие."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    """Pre-Norm блок: x + Attn(LN(x)), затем x + MLP(LN(x))."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # residual-поток не нормализуется — только вход подслоя
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # токены
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # абсолютные позиции
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying: вход и выход делят матрицу
        self.apply(self._init_weights)
        # Масштабируем выходные проекции residual-веток на 1/√(2·n_layer),
        # чтобы дисперсия residual-потока не росла с глубиной (рецепт GPT-2).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied-веса считаются один раз

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: [B, T] (int64) → logits: [B, T, vocab]; loss — если переданы targets [B, T]."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Длина {T} больше block_size={self.config.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))   # [B, T, C] + [T, C] (broadcast по батчу)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))            # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Авторегрессивная генерация БЕЗ KV-cache: на каждом шаге прогоняем весь префикс.

        Квадратичная стоимость по длине — именно её убирает KV-cache (задача C05).
        temperature=0 → жадный выбор.
        """
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
