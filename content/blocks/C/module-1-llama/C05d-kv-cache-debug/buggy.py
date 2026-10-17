"""KV-cache для attention Llama (GQA + RoPE)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны «RoPE» и «GQA и MQA»


class KVCache:
    """KV-cache одного слоя: ключи (уже повёрнутые RoPE) и значения всех прошлых токенов."""

    def __init__(self):
        self.k: torch.Tensor | None = None      # [B, H_kv, S, D]
        self.v: torch.Tensor | None = None      # [B, H_kv, S, D]

    @property
    def seq_len(self) -> int:
        """Сколько токенов уже в кеше (S)."""
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Дописывает k, v новых токенов [B, H_kv, T, D] и возвращает весь кеш [B, H_kv, S + T, D]."""
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)     # ось времени
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v


class LlamaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, rope_theta: float = 500000.0):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """x [B, T, C] — T новых токенов; cache — что было до них (или None) → [B, T, C]."""
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim

        # 1. Позиции новых токенов для RoPE.
        positions = torch.arange(T, device=x.device)
        cos, sin = rope_cos_sin(positions, D, self.rope_theta)              # [T, D]

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)                 # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2)              # [B, H_kv, T, D]
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # 2. В кеш кладём повёрнутые k и неразмноженные KV-головы.
        if cache is not None:
            k, v = cache.update(k, v)                                       # [B, H_kv, S, D]
        S = k.shape[2]                                                      # S = токены в кеше
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)           # [B, H, S, D]

        # 3. Маска [T, S]: запрос i стоит на позиции start + i и видит ключи 0..start + i.
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)                     # [B, H, T, S]
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v                                 # [B, H, T, D]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))
