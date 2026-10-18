"""Синусоидальные позиционные кодировки (Vaswani et al., 2017, §3.5) и эмбеддинг входа Transformer.

Реализуй sinusoidal_positions, shift_matrix и TransformerEmbedding.forward. Конструктор готов.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_positions(n_pos: int, d_model: int) -> torch.Tensor:
    """Таблица [n_pos, d_model]: PE[p, 2i] = sin(p·ω_i), PE[p, 2i+1] = cos(p·ω_i), ω_i = 10000^(−2i/d_model)."""
    # TODO 1: позиции p = 0 … n_pos − 1 столбцом [n_pos, 1] и частоты ω_i для i = 0 … d_model/2 − 1.
    #         Подсказка к показателю: 2i пробегает 0, 2, 4, … — это arange(0, d_model, 2).
    # TODO 2: углы p·ω_i → [n_pos, d_model / 2] (broadcasting).
    # TODO 3: sin — в чётные столбцы, cos — в нечётные (чередование, а не две половины).
    raise NotImplementedError("sinusoidal_positions")


def shift_matrix(k: int, d_model: int) -> torch.Tensor:
    """Матрица M [d_model, d_model], для которой PE[p + k] = M @ PE[p] при любом p: поворот каждой пары (sin, cos)."""
    # TODO: для каждой пары столбцов (2i, 2i+1) выпиши sin(a + b) и cos(a + b) через sin a, cos a и b = k·ω_i.
    #       Получится блок 2×2 на диагонали M. Остальные элементы — нули.
    raise NotImplementedError("shift_matrix")


class TransformerEmbedding(nn.Module):
    """Эмбеддинг токена × √d_model + синусоидальная позиция: [B, T] → [B, T, C]."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_id: int):
        super().__init__()
        # Готово, не меняй.
        self.d_model, self.max_len = d_model, max_len
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # Матрица общая с выходным слоем (B05), поэтому маленькая: std = 1/√C. Умножение на √C в forward
        # возвращает эмбеддингам масштаб ~1 — тот же, что у синусоид.
        nn.init.normal_(self.tok.weight, std=d_model ** -0.5)
        with torch.no_grad():
            self.tok.weight[pad_id].zero_()
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # TODO 1: если длина T больше max_len — ValueError (таблица позиций кончилась).
        # TODO 2: эмбеддинг токенов × √d_model плюс позиции 0 … T − 1 из self.pe (растянутся по батчу).
        raise NotImplementedError("TransformerEmbedding.forward")
