"""Одноголовое scaled dot-product attention с causal-маской.

Эталон. Имена слоёв — как в llmlab/core/gpt2.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



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
