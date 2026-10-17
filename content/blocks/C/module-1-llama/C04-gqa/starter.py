"""Grouped-Query Attention (GQA).

Реализуй:
1. repeat_kv — размножить KV-головы, чтобы каждой голове запроса досталась своя (общая для группы);
2. kv_cache_bytes — сколько памяти занимает KV-cache (ради этого GQA и придумали);
3. GroupedQueryAttention.forward — causal attention, где n_heads голов запросов делят
   n_kv_heads голов ключей/значений.

Готовый F.scaled_dot_product_attention в этой задаче запрещён — attention пишется вручную, как в «Causal Multi-Head Attention».
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] → [B, H_kv · n_rep, T, D]; голова запроса h использует KV-голову h // n_rep."""
    # TODO: копии одной KV-головы должны идти ПОДРЯД: [k0, k0, k1, k1] при n_rep = 2, а не [k0, k1, k0, k1].
    #       Какая операция PyTorch повторяет элементы «подряд», а какая — «по кругу»?
    raise NotImplementedError("repeat_kv")


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int,
                   batch: int = 1, bytes_per_elem: int = 2) -> int:
    """Размер KV-cache в байтах для всей модели."""
    # TODO: что хранится для КАЖДОГО токена на КАЖДОМ слое? Сколько там тензоров и какой они формы?
    raise NotImplementedError("kv_cache_bytes")


class GroupedQueryAttention(nn.Module):
    # Конструктор дан готовым — не меняй имена и формы: тесты сравнивают с эталоном по весам.
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads должно делиться на n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads          # сколько голов запросов делят одну KV-голову
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        # TODO 1: q → [B, H, T, D]; k и v → [B, H_kv, T, D] (view + transpose, как в «Causal Multi-Head Attention»).
        # TODO 2: размножь k и v до H голов через repeat_kv.
        # TODO 3: scores = q·kᵀ / √D, causal-маска, softmax по ключам, @ v.
        # TODO 4: склей головы обратно в [B, T, H·D] и примени o_proj.
        raise NotImplementedError("GroupedQueryAttention.forward")
