"""C19: gated attention (Qwen3-Next) — сигмоидный гейт на выходе attention и partial RoPE.

Реализуй:
1. apply_partial_rope — RoPE только на первых R каналах головы, остальные пропустить как есть;
2. GatedAttention.forward — attention как в C09 (GQA + QK-Norm), но с partial RoPE и выходным гейтом
   out = o_proj(attn ⊙ σ(gate)), где гейт — вторая половина q_proj каждой головы.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm, apply_rope, repeat_kv, rope_cos_sin   # эталоны C01, C03, C04


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D], cos/sin [T, R] (R ≤ D, чётное) → RoPE на первых R каналах, остальные D − R без изменений."""
    # TODO 1: R берётся из cos.shape[-1]; раздели x на вращаемую часть [..., :R] и остальное.
    # TODO 2: к вращаемой части — apply_rope из C03 (готов в llmlab.core.llama), остальное не трогать.
    # TODO 3: склеить обратно по последней оси.
    raise NotImplementedError("apply_partial_rope")


class GatedAttention(nn.Module):
    """GQA + QK-Norm + partial RoPE + выходной гейт: out = o_proj(attn(x) ⊙ σ(gate(x)))."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 partial_rotary_factor: float = 0.25, rope_theta: float = 10000.0, eps: float = 1e-6):
        super().__init__()                                  # конструктор не меняй
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
        # TODO 1: q_proj(x) → [B, T, H, 2·D] → по последней оси пополам: запрос [B, T, H, D] и гейт.
        #         Почему не q_proj(x).chunk(2) по всей проекции? Посмотри, как устроены веса в HF Qwen3-Next.
        # TODO 2: QK-Norm по голове (q_norm, k_norm — как в C09), затем головы на ось 1: [B, H, T, D].
        # TODO 3: cos, sin = rope_cos_sin(arange(T), self.rotary_dim, θ) → apply_partial_rope к q и k.
        # TODO 4: GQA (repeat_kv), scores/√D, causal-маска, softmax, @ v → [B, T, H·D].
        # TODO 5: умножить на σ(гейт) поэлементно и только потом o_proj.
        raise NotImplementedError("GatedAttention.forward")
