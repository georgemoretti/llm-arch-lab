"""Одноголовое scaled dot-product attention с causal-маской.

Реализуй causal_mask, attention и AttentionHead.forward. Конструктор AttentionHead готов.
Готовые F.scaled_dot_product_attention и nn.MultiheadAttention в этой задаче запрещены.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int) -> torch.Tensor:
    """Маска [T, T] типа bool: True там, куда позиции разрешено смотреть (сама и прошлые)."""
    # TODO: строка i — запрос позиции i, столбец j — ключ позиции j. Разрешено j ≤ i.
    #       Какая функция torch оставляет нижний треугольник матрицы?
    raise NotImplementedError("causal_mask")


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """Scaled dot-product attention: q, k [B, T, D], v [B, T, Dv] → (out [B, T, Dv], weights [B, T, T])."""
    # TODO 1: оценки сходства запросов с ключами: q @ kᵀ по двум последним осям → [B, T, T].
    # TODO 2: раздели на √D (D — размер запроса). Зачем? Посмотри на дисперсию q·k при росте D.
    # TODO 3: если mask передана — запрещённым ключам поставь −inf. До softmax или после — и почему?
    # TODO 4: softmax по оси ключей (каждая строка — распределение по ключам), затем weights @ v.
    raise NotImplementedError("attention")


class AttentionHead(nn.Module):
    """Одна голова causal self-attention: [B, T, C] → [B, T, head_size]."""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        # Готово, не меняй.
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", causal_mask(block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: q, k, v из x тремя проекциями; маска для текущей длины T — левый верхний угол self.mask;
        #       вернуть только выход attention (веса не нужны).
        raise NotImplementedError("AttentionHead.forward")
