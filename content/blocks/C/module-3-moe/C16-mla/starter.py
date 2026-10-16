"""C16: Multi-head Latent Attention (MLA) — K и V из общего латента, в кеше только латент.

Реализуй:
1. LatentCache.update — дописать латенты новых токенов и вернуть все;
2. MLAttention.forward — сжать x в латент c, (кеш), восстановить K и V всех голов из c, causal attention;
3. MLAttention.forward_absorbed — тот же результат БЕЗ восстановления K и V: W_UK «поглощается» в
   запрос, W_UV — в выход, attention считается прямо по латентам.

Позиционного кодирования в этой задаче нет (как NoPE-слой из C10) — RoPE добавим в C16b.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm   # эталон C01


class LatentCache:
    """KV-cache MLA: латенты c [B, S, r] (и, в C16b, общие RoPE-ключи [B, S, d_r]) вместо K и V."""

    def __init__(self):
        self.c: torch.Tensor | None = None
        self.k_rope: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return 0 if self.c is None else self.c.shape[1]

    def update(self, c: torch.Tensor, k_rope: torch.Tensor | None = None):
        """Дописывает новые токены по оси времени (dim=1) → (весь c, весь k_rope или None)."""
        # TODO: как KVCache в C05, но латенты лежат как [B, S, r] — склеивать по оси времени dim=1.
        #       k_rope понадобится только в C16b: если он не None — хранить так же.
        raise NotImplementedError("LatentCache.update")


def causal_mask(T: int, S: int, device) -> torch.Tensor:
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


class MLAttention(nn.Module):
    """MLA без позиционного кодирования: K и V восстанавливаются из общего латента c = norm(W_dkv x)."""

    def __init__(self, d_model: int, n_heads: int, kv_lora_rank: int, head_dim: int, v_head_dim: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.kv_a_proj = nn.Linear(d_model, kv_lora_rank, bias=False)          # сжатие: W_DKV
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, n_heads * (head_dim + self.v_head_dim), bias=False)  # W_UK, W_UV
        self.o_proj = nn.Linear(n_heads * self.v_head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: LatentCache | None = None) -> torch.Tensor:
        """x [B, T, C] (новые токены) → [B, T, C]."""
        # TODO 1: q [B, H, T, d]; латент c = kv_a_layernorm(kv_a_proj(x))  [B, T, r].
        # TODO 2: если есть кеш — дописать в него c и дальше работать со всеми латентами [B, S, r].
        #         Что дешевле хранить: c или уже восстановленные K, V?
        # TODO 3: kv_b_proj(c) → [B, S, H, d + dv] → по головам → split на k [B, H, S, d] и v [B, H, S, dv].
        # TODO 4: scores = q·kᵀ/√d, маска [T, S] со сдвигом (как в C05), softmax, @ v, o_proj.
        raise NotImplementedError("MLAttention.forward")

    def forward_absorbed(self, x: torch.Tensor, cache: LatentCache | None = None) -> torch.Tensor:
        """То же самое, но без вызова kv_b_proj: attention прямо по латентам."""
        # TODO 1: q и c — как в forward (с кешем).
        # TODO 2: веса kv_b_proj.weight [H·(d + dv), r] → view(H, d + dv, r) → W_uk [H, d, r], W_uv [H, dv, r].
        # TODO 3: q_lat = q·W_uk  [B, H, T, r]  (einsum "bhtd,hdr->bhtr"); scores = q_lat·cᵀ/√d  [B, H, T, S].
        # TODO 4: маска, softmax, o_lat = weights·c  [B, H, T, r]; out = o_lat·W_uvᵀ  [B, H, T, dv]; o_proj.
        raise NotImplementedError("MLAttention.forward_absorbed")
