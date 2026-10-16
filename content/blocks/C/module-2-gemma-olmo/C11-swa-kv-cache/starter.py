"""C11: KV-cache для sliding window attention — кольцевой буфер на W токенов.

Реализуй:
1. RingKVCache.update — вернуть ключи/значения (и их позиции), нужные новым запросам, и записать
   новые токены в кольцевой буфер: токен на позиции p — в ячейку p % W;
2. SlidingWindowCachedAttention.forward — SWA из C07, работающий с этим кешем: маска строится по
   АБСОЛЮТНЫМ ПОЗИЦИЯМ ключей, а не по их индексам в тензоре;
3. swa_kv_cache_bytes — память KV-cache модели со смесью sliding/full слоёв.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны C03 и C04


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
        # TODO 1: при первом вызове завести буферы: self.k, self.v — нули [B, H_kv, W, D],
        #         self.pos — torch.full((W,), -1) (ячейка пуста).
        # TODO 2: позиции новых токенов: seq_len … seq_len + T − 1.
        # TODO 3: что вернуть: всё, что может понадобиться новым запросам, — содержимое буфера ДО записи
        #         и все T новых токенов, вместе с их позициями (пустые ячейки — позиция −1).
        #         Порядок ключей не важен: маска будет строиться по позициям.
        # TODO 4: записать новые токены в ячейки p % W. Если T > W, в буфер попадут только последние W.
        #         Не забудь обновить self.pos и self.seq_len.
        raise NotImplementedError("RingKVCache.update")


class SlidingWindowCachedAttention(nn.Module):
    """SWA из C07 + RingKVCache. x [B, T, C] — новые токены, cache — прошлые (или None)."""

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
        # TODO 1: позиции запросов q_pos = start … start + T − 1, где start = cache.seq_len ДО update.
        # TODO 2: q, k, v; RoPE на q и k с позициями q_pos (как в C05).
        # TODO 3: с кешем — k, v, k_pos = cache.update(k, v); без кеша — k_pos = q_pos.
        # TODO 4: repeat_kv, scores [B, H, T, S].
        # TODO 5: маска [T, S] по позициям: ключ существует (k_pos ≥ 0), не из будущего и в окне.
        #         Почему здесь не подойдёт sliding_window_mask(T, S, W) из C07?
        # TODO 6: softmax → @ v → o_proj.
        raise NotImplementedError("SlidingWindowCachedAttention.forward")


def swa_kv_cache_bytes(layer_types: list[str], window: int, n_kv_heads: int, head_dim: int,
                       seq_len: int, batch: int = 1, bytes_per_elem: int = 2) -> int:
    """KV-cache модели со смесью sliding/full слоёв: sliding хранит min(seq_len, window) токенов."""
    # TODO: sliding-слой хранит min(seq_len, window) токенов, full — seq_len. Дальше как kv_cache_bytes
    #       из C04: 2 (K и V) × токены × n_kv_heads × head_dim × batch × байт на число.
    raise NotImplementedError("swa_kv_cache_bytes")
