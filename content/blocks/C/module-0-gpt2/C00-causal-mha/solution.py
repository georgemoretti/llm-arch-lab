"""Эталон: causal multi-head self-attention, как в GPT-2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention с causal-маской.

    q, k, v: [B, H, T, D] → выход [B, H, T, D].
    """
    T, D = q.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)                       # [B, H, T, T]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))  # True = можно смотреть
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)                                    # нормируем по ключам
    return weights @ v                                                     # [B, H, T, D]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)  # Q, K, V одной матрицей
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)      # выходная проекция

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        q, k, v = self.c_attn(x).split(C, dim=-1)                  # 3 × [B, T, C]
        q = q.view(B, T, H, D).transpose(1, 2)                     # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        out = causal_attention(q, k, v)                            # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)       # [B, T, C]
        return self.c_proj(out)
