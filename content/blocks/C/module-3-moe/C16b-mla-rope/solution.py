"""Эталон C16b: MLA с decoupled RoPE (DeepSeek-V2/V3) и память его KV-cache."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm, apply_rope, rope_cos_sin   # эталоны C01 и C03
from llmlab.core.mla import LatentCache, causal_mask             # эталон C16


class MLARopeAttention(nn.Module):
    """MLA DeepSeek-V2/V3 (без сжатия запроса): часть каждой головы — без позиций (из латента),
    часть — RoPE с ОДНИМ общим на все головы ключом k_rope."""

    def __init__(self, d_model: int, n_heads: int, kv_lora_rank: int, qk_nope_head_dim: int,
                 qk_rope_head_dim: int, v_head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * (qk_nope_head_dim + qk_rope_head_dim), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: LatentCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, dn, dr, dv, r = (self.n_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim,
                            self.kv_lora_rank)
        start = cache.seq_len if cache is not None else 0
        cos, sin = rope_cos_sin(torch.arange(start, start + T, device=x.device), dr, self.rope_theta)
        q = self.q_proj(x).view(B, T, H, dn + dr).transpose(1, 2)
        q_nope, q_rope = q.split([dn, dr], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)                                         # [B, H, T, dr]
        c, k_rope = self.kv_a_proj_with_mqa(x).split([r, dr], dim=-1)                 # [B, T, r], [B, T, dr]
        c = self.kv_a_layernorm(c)
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin).squeeze(1)                 # одна «голова» на всех
        if cache is not None:
            c, k_rope = cache.update(c, k_rope)                                       # [B, S, r], [B, S, dr]
        S = c.shape[1]
        kv = self.kv_b_proj(c).view(B, S, H, dn + dv).transpose(1, 2)
        k_nope, v = kv.split([dn, dv], dim=-1)                                        # [B, H, S, dn], [B, H, S, dv]
        k = torch.cat((k_nope, k_rope.unsqueeze(1).expand(B, H, S, dr)), dim=-1)      # [B, H, S, dn + dr]
        q = torch.cat((q_nope, q_rope), dim=-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dn + dr)
        scores = scores.masked_fill(~causal_mask(T, S, x.device), float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


def mla_kv_cache_bytes(n_layers: int, kv_lora_rank: int, qk_rope_head_dim: int, seq_len: int,
                       batch: int = 1, bytes_per_elem: int = 2) -> int:
    """KV-cache MLA: на токен и слой — латент r и общий RoPE-ключ d_r (без множителя 2 за K и V)."""
    return n_layers * (kv_lora_rank + qk_rope_head_dim) * seq_len * batch * bytes_per_elem
