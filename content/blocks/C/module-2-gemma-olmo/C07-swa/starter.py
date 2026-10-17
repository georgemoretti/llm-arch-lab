"""sliding window attention (SWA) и паттерн local:global по слоям.

Реализуй:
1. sliding_window_mask — маску, где запрос видит только последние `window` позиций (включая свою);
2. layer_types — расписание слоёв Gemma: какие слои sliding, какие full;
3. SlidingWindowAttention.forward — attention из модуля 1 (GQA + RoPE) с этой маской.

RoPE и repeat_kv даны готовыми (эталоны «RoPE» и «GQA и MQA»).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны «RoPE» и «GQA и MQA»


def sliding_window_mask(T: int, S: int | None = None, window: int | None = None,
                        device: torch.device | None = None) -> torch.Tensor:
    """Разрешённые пары (запрос, ключ) → bool [T, S] (True = можно смотреть).

    S — сколько всего ключей (по умолчанию S = T); запросы — последние T из S позиций, то есть
    запрос i стоит на позиции S − T + i (как в «KV-cache» с кешем). window=None — обычная causal-маска.
    """
    # TODO 1: абсолютные позиции запросов [T, 1] и ключей [1, S] — через torch.arange и broadcasting.
    # TODO 2: causal-часть: ключ не позже запроса.
    # TODO 3: окно: сколько позиций назад можно смотреть, если окно из `window` позиций
    #         включает сам токен? Проверь себя на window=1 — должна остаться только диагональ.
    raise NotImplementedError("sliding_window_mask")


def layer_types(n_layers: int, sliding_window_pattern: int) -> list[str]:
    """Список длины n_layers из 'sliding_attention' / 'full_attention'.

    Gemma 3: pattern = 6 → в каждой группе из 6 слоёв 5 sliding и 1 full. Какой слой в группе
    full — первый или последний? (См. README.)
    """
    # TODO: одна строка со списковым включением; индексы слоёв начинаются с 0.
    raise NotImplementedError("layer_types")


class SlidingWindowAttention(nn.Module):
    # Конструктор дан готовым — не меняй имена и формы: тесты сравнивают с эталоном по весам.
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, window: int | None,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        # TODO 1: q [B, H, T, D], k и v [B, H_kv, T, D]; RoPE на q и k (позиции 0..T−1).
        # TODO 2: repeat_kv, scores = q·kᵀ/√D  [B, H, T, T].
        # TODO 3: единственное отличие от LlamaAttention — маска: sliding_window_mask(T, window=…).
        # TODO 4: softmax → @ v → [B, T, H·D] → o_proj.
        raise NotImplementedError("SlidingWindowAttention.forward")
