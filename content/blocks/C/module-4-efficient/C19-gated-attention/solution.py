"""Эталон C19: gated attention (Qwen3-Next) — сигмоидный гейт на выходе attention и partial RoPE."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm, apply_rope, repeat_kv, rope_cos_sin   # эталоны C01, C03, C04


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D], cos/sin [T, R] (R ≤ D, чётное) → RoPE на первых R каналах, остальные D − R без изменений."""
    R = cos.shape[-1]
    x_rot, x_pass = x[..., :R], x[..., R:]
    return torch.cat((apply_rope(x_rot, cos, sin), x_pass), dim=-1)


class GatedAttention(nn.Module):
    """GQA + QK-Norm + partial RoPE + выходной гейт: out = o_proj(attn(x) ⊙ σ(gate(x)))."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 partial_rotary_factor: float = 0.25, rope_theta: float = 10000.0, eps: float = 1e-6):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.rope_theta = rope_theta
        # q_proj выдаёт на каждую голову [запрос (head_dim) | гейт (head_dim)] — как в HF Qwen3-Next
        self.q_proj = nn.Linear(d_model, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q, gate = self.q_proj(x).view(B, T, H, 2 * D).chunk(2, dim=-1)              # [B, T, H, D] × 2
        gate = gate.reshape(B, T, H * D)
        q = self.q_norm(q).transpose(1, 2)                                            # [B, H, T, D]
        k = self.k_norm(self.k_proj(x).view(B, T, H_kv, D)).transpose(1, 2)           # [B, H_kv, T, D]
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), self.rotary_dim, self.rope_theta)  # [T, R]
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        k, v = repeat_kv(k, H // H_kv), repeat_kv(v, H // H_kv)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v         # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o_proj(out * torch.sigmoid(gate))
