"""NoPE — слои без позиционного кодирования через каждый N-й слой (как в SmolLM3).

Реализуй:
1. rope_layers — расписание: в каких слоях RoPE есть, в каких нет;
2. NoPEAttention.forward — GQA + causal-маска, RoPE только при self.use_rope;
3. build_attention_layers — стек слоёв по расписанию.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны «RoPE» и «GQA и MQA»


def rope_layers(n_layers: int, no_rope_layer_interval: int) -> list[bool]:
    """SmolLM3: каждый interval-й слой (считая с 1) — без RoPE. True = в слое есть RoPE."""
    # TODO: список длины n_layers. Как в Gemma («Sliding window attention»): особый слой — последний в каждой группе
    #       из interval слоёв, индексы с 0. Здесь «особый» = без RoPE, а True = RoPE есть.
    raise NotImplementedError("rope_layers")


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
        # TODO 1: q [B, H, T, D], k и v [B, H_kv, T, D].
        # TODO 2: RoPE на q и k — только если self.use_rope.
        # TODO 3: repeat_kv, scores, causal-маска, softmax, @ v, o_proj.
        #         Нужна ли маска NoPE-слою? (README: откуда NoPE вообще знает о позициях.)
        raise NotImplementedError("NoPEAttention.forward")


def build_attention_layers(n_layers: int, d_model: int, n_heads: int, n_kv_heads: int,
                           no_rope_layer_interval: int, rope_theta: float = 10000.0) -> nn.ModuleList:
    """Стек attention-слоёв SmolLM3: в каждом слое RoPE или NoPE по расписанию rope_layers."""
    # TODO: nn.ModuleList из NoPEAttention, use_rope каждого слоя — из rope_layers.
    raise NotImplementedError("build_attention_layers")
