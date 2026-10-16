"""Готовые компоненты Llama — эталоны задач модуля 1 блока C.

Задачи модуля 1 строят Llama по кусочкам: C01 RMSNorm, C02 SwiGLU, C03 RoPE, C04 GQA,
C05 KV-cache, C06 блок и модель. Чтобы поздние задачи не зависели от ошибок в ранних (и чтобы
финальная модель Фазы 8 собиралась из одного места), эталонные версии лежат здесь, а задачи
импортируют их: ``from llmlab.core.llama import RMSNorm, apply_rope, ...``.

⚠️ Это готовые ответы на задачи C01–C05. Если ещё не решил их — не подглядывай.

Код намеренно совпадает с ``content/blocks/C/module-1-llama/C0*/solution.py`` (имена и формулы).
Расхождение ловит ``tests/test_core_llama.py``.

Имена параметров — как в Hugging Face ``transformers`` (``modeling_llama.py``), чтобы в Фазе 8
загружать реальные веса без переименований.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------- C01 RMSNorm


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm по последней оси: x [..., C], weight [C] → [..., C] того же dtype."""
    input_dtype = x.dtype
    h = x.float()
    inv_rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (h * inv_rms).to(input_dtype) * weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------- C02 SwiGLU


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def llama_ffn_hidden_dim(d_model: int, multiple_of: int = 256, ffn_dim_multiplier: float | None = None) -> int:
    hidden = int(2 * (4 * d_model) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------- C03 RoPE


def rope_inv_freq(head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    return theta ** (-exponents)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float = 10000.0):
    """cos, sin углов для позиций ``positions`` [T] → два тензора [T, D] (раскладка «половинами»)."""
    angles = positions.float()[:, None] * rope_inv_freq(head_dim, theta)[None, :]
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D], cos/sin [T, D] → [B, H, T, D]."""
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------- C04 GQA


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] → [B, H_kv·n_rep, T, D]; голова h берёт KV-голову h // n_rep."""
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    return x[:, :, None].expand(B, H_kv, n_rep, T, D).reshape(B, H_kv * n_rep, T, D)


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int,
                   batch: int = 1, bytes_per_elem: int = 2) -> int:
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem
