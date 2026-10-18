"""Эмбеддинги токенов и обучаемых позиций, weight tying, языковая модель без attention.

Эталон. Имена слоёв — как в llmlab/core/gpt2.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



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
