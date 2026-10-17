"""Эталон «Найди баг: SwiGLU FFN» (исправленная версия): SwiGLU FFN, как в Llama."""

import torch
import torch.nn as nn


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (она же Swish с β = 1): x · σ(x). Форма не меняется."""
    return x * torch.sigmoid(x)


def llama_ffn_hidden_dim(d_model: int, multiple_of: int = 256, ffn_dim_multiplier: float | None = None) -> int:
    """Скрытый размер FFN по правилу Llama: 2/3 от 4·d, множитель, округление вверх до multiple_of."""
    hidden = int(2 * (4 * d_model) / 3)                  # 8d/3: столько же параметров, сколько у MLP 4d
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)   # округление ВВЕРХ


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)   # ветка, через которую идёт SiLU
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)     # линейная ветка
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)   # обратно в d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, T, C] → gate, up [B, T, F] → поэлементно → [B, T, F] → down → [B, T, C]
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
