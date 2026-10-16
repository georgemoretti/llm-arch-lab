"""C01d: RMSNorm из нашего порта Llama."""

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm по последней оси.

    x: [..., C], weight: [C] → выход [..., C] того же dtype, что и x.
    """
    input_dtype = x.dtype
    h = x.float()                                        # считаем в fp32: x² в fp16 легко переполняется
    inv_rms = 1.0 / (h.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)  # [..., 1]
    return (h * inv_rms).to(input_dtype) * weight        # масштаб по каналам, без сдвига (bias нет)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))      # γ, инициализация единицами

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)
