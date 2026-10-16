"""C10d: NoPE по расписанию SmolLM3 — код из C10 с одной ошибкой. Найди и исправь её."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны C03 и C04


def rope_layers(n_layers: int, no_rope_layer_interval: int) -> list[bool]:
    """SmolLM3: каждый interval-й слой (считая с 1) — без RoPE. True = в слое есть RoPE."""
    return [i % no_rope_layer_interval != 0 for i in range(n_layers)]


class NoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, use_rope: bool = True,
                 rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]. GQA + causal-маска; RoPE — только если self.use_rope."""
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        if self.use_rope:                                   # NoPE-слой: позиций нет вообще
            cos, sin = rope_cos_sin(torch.arange(T, device=x.device), D, self.rope_theta)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()    # маска остаётся всегда
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


def build_attention_layers(n_layers: int, d_model: int, n_heads: int, n_kv_heads: int,
                           no_rope_layer_interval: int, rope_theta: float = 10000.0) -> nn.ModuleList:
    """Стек attention-слоёв SmolLM3: в каждом слое RoPE или NoPE по расписанию rope_layers."""
    return nn.ModuleList(NoPEAttention(d_model, n_heads, n_kv_heads, use_rope, rope_theta)
                         for use_rope in rope_layers(n_layers, no_rope_layer_interval))
