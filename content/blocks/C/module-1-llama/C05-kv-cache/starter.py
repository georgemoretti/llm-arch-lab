"""C05: KV-cache для attention Llama (GQA + RoPE).

Реализуй:
1. KVCache.update — дописать k, v новых токенов к кешу и вернуть весь кеш;
2. LlamaAttention.forward — attention из C04 + RoPE из C03, который умеет работать с кешем:
   на вход приходят только НОВЫЕ токены, а смотреть они должны на все прошлые.

RoPE и repeat_kv даны готовыми (эталоны C03 и C04).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны C03 и C04


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
        # TODO: первый вызов — просто запомнить; дальше — склеить с тем, что есть. По какой оси?
        raise NotImplementedError("KVCache.update")


class LlamaAttention(nn.Module):
    # Конструктор дан готовым — не меняй имена и формы: тесты сравнивают с эталоном по весам.
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
        # TODO 1: на каких абсолютных позициях стоят эти T токенов, если в кеше уже S штук?
        #         Посчитай cos, sin через rope_cos_sin(positions, head_dim, self.rope_theta).
        # TODO 2: q [B, H, T, D], k и v [B, H_kv, T, D]; поверни q и k (apply_rope).
        # TODO 3: если кеш есть — допиши в него k, v и дальше работай со всем кешем.
        #         Что выгоднее хранить: k до или после RoPE? до или после repeat_kv?
        # TODO 4: repeat_kv, scores [B, H, T, S], маска [T, S], softmax, @ v, o_proj.
        #         Запрос i стоит на позиции S − T + i. Какие ключи ему можно видеть?
        raise NotImplementedError("LlamaAttention.forward")
