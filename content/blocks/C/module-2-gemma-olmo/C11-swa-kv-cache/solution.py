"""Эталон: KV-cache для sliding window attention — кольцевой буфер на W токенов."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны «RoPE» и «GQA и MQA»


class RingKVCache:
    """KV-cache sliding-слоя: хранит не больше ``window`` последних токенов в кольцевом буфере.

    Токен на абсолютной позиции p лежит в ячейке p % window. ``pos`` — абсолютные позиции в
    ячейках (−1 — ячейка пуста), по ним attention строит маску.
    """

    def __init__(self, window: int):
        self.window = window
        self.k: torch.Tensor | None = None      # [B, H_kv, W, D]
        self.v: torch.Tensor | None = None      # [B, H_kv, W, D]
        self.pos: torch.Tensor | None = None    # [W], long
        self.seq_len = 0                        # сколько токенов прошло через кеш всего

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """k, v новых токенов [B, H_kv, T, D] на позициях seq_len … seq_len+T−1.

        → (K, V, k_pos): ключи, которые могут понадобиться новым запросам, — содержимое буфера ДО
        записи плюс все новые, [B, H_kv, W + T, D], и их позиции [W + T]. Затем новые токены
        записываются в буфер (последние W из них).
        """
        B, H_kv, T, D = k.shape
        W = self.window
        if self.k is None:
            self.k = k.new_zeros(B, H_kv, W, D)
            self.v = v.new_zeros(B, H_kv, W, D)
            self.pos = torch.full((W,), -1, dtype=torch.long, device=k.device)
        new_pos = torch.arange(self.seq_len, self.seq_len + T, device=k.device)
        K = torch.cat((self.k, k), dim=2)
        V = torch.cat((self.v, v), dim=2)
        k_pos = torch.cat((self.pos, new_pos))
        keep = min(T, W)
        slots = new_pos[-keep:] % W
        self.k[:, :, slots] = k[:, :, -keep:]
        self.v[:, :, slots] = v[:, :, -keep:]
        self.pos[slots] = new_pos[-keep:]
        self.seq_len += T
        return K, V, k_pos


class SlidingWindowCachedAttention(nn.Module):
    """SWA из «Sliding window attention» + RingKVCache. x [B, T, C] — новые токены, cache — прошлые (или None)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, window: int,
                 rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: RingKVCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        start = cache.seq_len if cache is not None else 0
        q_pos = torch.arange(start, start + T, device=x.device)
        cos, sin = rope_cos_sin(q_pos, D, self.rope_theta)
        q = apply_rope(self.q_proj(x).view(B, T, H, D).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2), cos, sin)
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        if cache is not None:
            k, v, k_pos = cache.update(k, v)
        else:
            k_pos = q_pos
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        dist = q_pos[:, None] - k_pos[None, :]
        mask = (k_pos[None, :] >= 0) & (dist >= 0) & (dist < self.window)
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


def swa_kv_cache_bytes(layer_types: list[str], window: int, n_kv_heads: int, head_dim: int,
                       seq_len: int, batch: int = 1, bytes_per_elem: int = 2) -> int:
    """KV-cache модели со смесью sliding/full слоёв: sliding хранит min(seq_len, window) токенов."""
    tokens = sum(min(seq_len, window) if t == "sliding_attention" else seq_len for t in layer_types)
    return 2 * tokens * n_kv_heads * head_dim * batch * bytes_per_elem
