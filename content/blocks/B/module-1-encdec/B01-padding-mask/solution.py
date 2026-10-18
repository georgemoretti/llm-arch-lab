"""Padding-маска и двунаправленное multi-head attention.

Эталон. Маски — bool, True = «можно смотреть». Имена проекций — q_proj / k_proj / v_proj / out_proj (как у
энкодеров в Hugging Face: BART, Marian; в nn.MultiheadAttention три первые склеены в in_proj).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B, S] → bool [B, 1, 1, S]: True у настоящих токенов. Оси 1 и 2 — головы и запросы (растянутся)."""
    return (ids != pad_id)[:, None, None, :]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """q [B, H, T, D], k [B, H, S, D], v [B, H, S, Dv] → (out [B, H, T, Dv], weights [B, H, T, S]).

    mask — bool, растягивается до [B, H, T, S]; True — ключ разрешён.
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])      # [B, H, T, S]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))          # запрещённые ключи — до softmax
    weights = F.softmax(scores, dim=-1)                            # распределение по ключам
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention: запросы из x [B, T, C], ключи и значения из context [B, S, C] (по умолчанию x)."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, C] → [B, H, L, C / H]."""
        B, L, C = x.shape
        return x.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                need_weights: bool = False):
        context = x if context is None else context
        q = self.split_heads(self.q_proj(x))                           # [B, H, T, D]
        k = self.split_heads(self.k_proj(context))                     # [B, H, S, D]
        v = self.split_heads(self.v_proj(context))                     # [B, H, S, D]
        out, weights = attention(q, k, v, mask)                        # [B, H, T, D]
        B, H, T, D = out.shape
        out = self.out_proj(out.transpose(1, 2).reshape(B, T, H * D))  # склеить головы → [B, T, C]
        return (out, weights) if need_weights else out
