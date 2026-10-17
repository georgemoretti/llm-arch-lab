"""MLA с decoupled RoPE (DeepSeek-V2/V3) и память его KV-cache.

Реализуй:
1. MLARopeAttention.forward — у каждой головы две части: «без позиций» (из латента, как в «MLA») и
   «с RoPE» (ключ один на все головы, приходит из kv_a_proj_with_mqa вместе с латентом);
2. mla_kv_cache_bytes — сколько памяти занимает такой кеш.

LatentCache («MLA») хранит латенты c и повёрнутые RoPE-ключи; causal_mask(T, S) — маска со сдвигом.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm, apply_rope, rope_cos_sin   # эталоны «RMSNorm» и «RoPE»
from llmlab.core.mla import LatentCache, causal_mask             # эталон «MLA»


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
        # TODO 1: позиции новых токенов start..start+T−1 (start = cache.seq_len ДО update), cos, sin для d_r.
        # TODO 2: q [B, H, T, dn + dr] → split на q_nope и q_rope; RoPE только на q_rope.
        # TODO 3: kv_a_proj_with_mqa(x) → split на латент c [B, T, r] и k_rope [B, T, dr]; c нормировать;
        #         RoPE на k_rope (у него нет оси голов: apply_rope ждёт [.., T, dr] — добавь ось и убери).
        # TODO 4: кеш: c, k_rope = cache.update(c, k_rope) — хранятся повёрнутые ключи.
        # TODO 5: kv_b_proj(c) → k_nope [B, H, S, dn], v [B, H, S, dv] (как в «MLA»).
        # TODO 6: k = [k_nope ; k_rope, размноженный на все H голов], q = [q_nope ; q_rope];
        #         scores = q·kᵀ / √(dn + dr), маска causal_mask(T, S), softmax, @ v, o_proj.
        raise NotImplementedError("MLARopeAttention.forward")


def mla_kv_cache_bytes(n_layers: int, kv_lora_rank: int, qk_rope_head_dim: int, seq_len: int,
                       batch: int = 1, bytes_per_elem: int = 2) -> int:
    """KV-cache MLA: на токен и слой — латент r и общий RoPE-ключ d_r (без множителя 2 за K и V)."""
    # TODO: сколько чисел на токен и слой лежит в кеше MLA? Нужен ли множитель 2 «за K и V»?
    raise NotImplementedError("mla_kv_cache_bytes")
