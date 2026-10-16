"""C18: упрощённое sparse attention в стиле DSA (DeepSeek-V3.2) — индексатор + top-k ключей.

Реализуй:
1. LightningIndexer.forward — дешёвые оценки «насколько ключ s нужен запросу t» для всех пар;
2. topk_mask — по оценкам выбрать в каждой строке top_k допустимых ключей (булева маска);
3. sparse_attention — обычный softmax-attention, но только по выбранным ключам;
4. DSAttention.forward — собрать всё вместе.

Позиционного кодирования здесь нет (у DeepSeek RoPE есть и в индексаторе, и в MLA) — сосредоточься на выборе ключей.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):
    """Лёгкий «оценщик» пар (запрос t, ключ s): несколько маленьких голов, ReLU, взвешенная сумма.

    I(t, s) = Σ_h w_{t,h} · ReLU(q_{t,h} · k_s) / √(H_I · d_I); ключ индексатора один на токен (как MQA).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()                                  # конструктор не меняй
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.weights_proj = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → оценки [B, T, T]; будущие ключи (s > t) = −inf."""
        # TODO 1: q [B, T, H_I, d_I], k [B, T, d_I] (одна «голова» ключей на всех), w [B, T, H_I] / √H_I.
        # TODO 2: q·k по каждой голове индексатора → [B, T, H_I, S], делить на √d_I (einsum "bthd,bsd->bths").
        # TODO 3: ReLU, затем взвешенная сумма по головам с весами w → [B, T, S].
        # TODO 4: будущим ключам — −inf. Почему маскировать надо здесь, а не только в основном attention?
        raise NotImplementedError("LightningIndexer.forward")


def topk_mask(index_scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """index_scores [B, T, S] → bool [B, T, S]: True у top_k лучших ДОПУСТИМЫХ (не −inf) ключей строки."""
    # TODO 1: k = min(top_k, S); индексы лучших k по последней оси (по ключам).
    # TODO 2: булева маска из индексов (scatter_ по последней оси).
    # TODO 3: в строке t допустимых ключей t + 1. Что вернёт topk, если k > t + 1? Убери лишнее.
    raise NotImplementedError("topk_mask")


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """q [B, H, T, d], k/v [B, H, S, d], keep [B, T, S] (общая для всех голов) → [B, H, T, d]."""
    # TODO: scores = q·kᵀ/√d; где keep = False — −inf (маске нужна ось голов); softmax по ключам; @ v.
    raise NotImplementedError("sparse_attention")


class DSAttention(nn.Module):
    """MHA, в которой каждый запрос смотрит только на top_k ключей, выбранных индексатором."""

    def __init__(self, d_model: int, n_heads: int, index_n_heads: int, index_head_dim: int, top_k: int):
        super().__init__()                                  # конструктор не меняй
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.indexer = LightningIndexer(d_model, index_n_heads, index_head_dim)

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        """x [B, T, C] → [B, T, C] (и маска выбранных ключей [B, T, T], если return_mask)."""
        # TODO 1: q, k, v → [B, H, T, d] (как в C00).
        # TODO 2: оценки индексатора по x → topk_mask(…, self.top_k).
        # TODO 3: sparse_attention → [B, T, C] → o_proj. Вернуть (out, keep), если return_mask.
        raise NotImplementedError("DSAttention.forward")
