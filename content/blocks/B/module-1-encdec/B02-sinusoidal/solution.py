"""Синусоидальные позиционные кодировки (Vaswani et al., 2017, §3.5) и эмбеддинг входа Transformer.

Эталон.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_positions(n_pos: int, d_model: int) -> torch.Tensor:
    """Таблица [n_pos, d_model]: PE[p, 2i] = sin(p·ω_i), PE[p, 2i+1] = cos(p·ω_i), ω_i = 10000^(−2i/d_model)."""
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]                        # [n_pos, 1]
    omega = 10000.0 ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)  # [d_model / 2]
    angle = pos * omega                                                            # [n_pos, d_model / 2]
    pe = torch.zeros(n_pos, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


def shift_matrix(k: int, d_model: int) -> torch.Tensor:
    """Матрица M [d_model, d_model], для которой PE[p + k] = M @ PE[p] при любом p: поворот каждой пары (sin, cos)."""
    omega = 10000.0 ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    c, s = torch.cos(k * omega), torch.sin(k * omega)
    m = torch.zeros(d_model, d_model)
    m[0::2, 0::2] = torch.diag(c)      # sin(a + b) = sin a · cos b + cos a · sin b
    m[0::2, 1::2] = torch.diag(s)
    m[1::2, 0::2] = torch.diag(-s)     # cos(a + b) = cos a · cos b − sin a · sin b
    m[1::2, 1::2] = torch.diag(c)
    return m


class TransformerEmbedding(nn.Module):
    """Эмбеддинг токена × √d_model + синусоидальная позиция: [B, T] → [B, T, C]."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_id: int):
        super().__init__()
        self.d_model, self.max_len = d_model, max_len
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # Матрица общая с выходным слоем (B05), поэтому маленькая: std = 1/√C. Умножение на √C в forward
        # возвращает эмбеддингам масштаб ~1 — тот же, что у синусоид.
        nn.init.normal_(self.tok.weight, std=d_model ** -0.5)
        with torch.no_grad():
            self.tok.weight[pad_id].zero_()
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        if T > self.max_len:
            raise ValueError(f"длина {T} больше max_len = {self.max_len}")
        return self.tok(ids) * math.sqrt(self.d_model) + self.pe[:T]
