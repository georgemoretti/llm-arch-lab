"""Padding-маска и двунаправленное multi-head attention.

Реализуй padding_mask, attention, MultiHeadAttention.split_heads и MultiHeadAttention.forward.
Конструктор готов. Маски — bool, True = «можно смотреть».
Готовые F.scaled_dot_product_attention и nn.MultiheadAttention в этой задаче запрещены.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B, S] → bool [B, 1, 1, S]: True у настоящих токенов. Оси 1 и 2 — головы и запросы (растянутся)."""
    # TODO: True там, где токен не PAD. Затем добавь две оси так, чтобы маска растягивалась
    #       по головам и по запросам, а различалась только по ключам (последняя ось).
    raise NotImplementedError("padding_mask")


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """q [B, H, T, D], k [B, H, S, D], v [B, H, S, Dv] → (out [B, H, T, Dv], weights [B, H, T, S]).

    mask — bool, растягивается до [B, H, T, S]; True — ключ разрешён.
    """
    # TODO 1: оценки q @ kᵀ по двум последним осям, деление на √D → [B, H, T, S]. T и S могут различаться.
    # TODO 2: если mask передана — запрещённым ключам −inf (до softmax).
    # TODO 3: softmax по оси ключей, затем weights @ v.
    raise NotImplementedError("attention")


class MultiHeadAttention(nn.Module):
    """Multi-head attention: запросы из x [B, T, C], ключи и значения из context [B, S, C] (по умолчанию x)."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        # Готово, не меняй: имена проекций проверяют тесты.
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, C] → [B, H, L, C / H]."""
        # TODO: последнюю ось C разрежь на H кусков по C / H, затем переставь ось голов перед осью позиций.
        #       Порядок важен: какие числа окажутся в одной голове?
        raise NotImplementedError("split_heads")

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                need_weights: bool = False):
        # TODO 1: context по умолчанию — x (self-attention). Иначе это cross-attention (пригодится в B04).
        # TODO 2: q — из x, k и v — из context; каждую проекцию разбей на головы.
        # TODO 3: attention с маской; головы обратно в [B, T, C] (transpose + reshape), затем out_proj.
        # TODO 4: вернуть out или (out, weights), если need_weights.
        raise NotImplementedError("MultiHeadAttention.forward")
