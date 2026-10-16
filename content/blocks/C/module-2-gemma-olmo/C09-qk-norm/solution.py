"""Эталон C09: attention с QK-Norm (до RoPE) — по голове (Gemma 3) или по всей проекции (OLMo 2)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm, apply_rope, repeat_kv, rope_cos_sin   # эталоны C01, C03, C04



class QKNormAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, qk_norm: str = "head",
                 rope_theta: float = 10000.0, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0 and qk_norm in ("head", "full")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        # "head": одна норма размера D на все головы (Gemma 3, Qwen3);
        # "full": норма по всей проекции H·D и H_kv·D до разбиения на головы (OLMo 2).
        q_dim, k_dim = (self.head_dim, self.head_dim) if qk_norm == "head" else \
            (n_heads * self.head_dim, n_kv_heads * self.head_dim)
        self.q_norm = RMSNorm(q_dim, eps)
        self.k_norm = RMSNorm(k_dim, eps)

    def scores(self, x: torch.Tensor):
        """x [B, T, C] → (scores [B, H, T, T] до маски и softmax, v [B, H, T, D])."""
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q, k = self.q_proj(x), self.k_proj(x)                      # [B, T, H·D], [B, T, H_kv·D]
        if self.qk_norm == "full":
            q, k = self.q_norm(q), self.k_norm(k)                  # OLMo 2: до разбиения на головы
        q = q.view(B, T, H, D).transpose(1, 2)                     # [B, H, T, D]
        k = k.view(B, T, H_kv, D).transpose(1, 2)                  # [B, H_kv, T, D]
        if self.qk_norm == "head":
            q, k = self.q_norm(q), self.k_norm(k)                  # Gemma 3: по последней оси D
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), D, self.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)    # RoPE — ПОСЛЕ нормы
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        return q @ k.transpose(-2, -1) / math.sqrt(D), v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        B, T, _ = x.shape
        scores, v = self.scores(x)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
