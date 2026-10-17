"""QK-Norm — RMSNorm на запросах и ключах перед скалярным произведением.

Реализуй:
1. QKNormAttention.scores — q, k, v, QK-Norm, RoPE, repeat_kv и scores = q·kᵀ/√D (до маски);
2. QKNormAttention.forward — causal-маска, softmax, @ v, o_proj.

Режимы: qk_norm="head" (Gemma 3, Qwen3) — норма по каждой голове отдельно;
        qk_norm="full" (OLMo 2) — норма по всей проекции до разбиения на головы.
Нормы q_norm и k_norm уже созданы в конструкторе нужного размера.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm, apply_rope, repeat_kv, rope_cos_sin   # эталоны «RMSNorm», «RoPE», «GQA и MQA»



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
        # TODO 1: проекции q, k. В режиме "full" нормы применяются ЗДЕСЬ — к [B, T, H·D] целиком.
        # TODO 2: разбить на головы: q [B, H, T, D], k [B, H_kv, T, D]. В режиме "head" нормы — здесь.
        # TODO 3: RoPE на q и k. До или после нормы? (README: что сломается в обратном порядке.)
        # TODO 4: v, repeat_kv для k и v, scores = q·kᵀ/√D. Вернуть (scores, v).
        raise NotImplementedError("QKNormAttention.scores")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        # TODO: scores, v = self.scores(x) → causal-маска → softmax → @ v → [B, T, H·D] → o_proj.
        raise NotImplementedError("QKNormAttention.forward")
