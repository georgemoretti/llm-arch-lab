"""Эталон: sliding window attention и паттерн local:global по слоям."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, repeat_kv, rope_cos_sin   # эталоны «RoPE» и «GQA и MQA»


def sliding_window_mask(T: int, S: int | None = None, window: int | None = None,
                        device: torch.device | None = None) -> torch.Tensor:
    """Разрешённые пары (запрос, ключ) → bool [T, S]. Запросы — последние T из S позиций."""
    S = T if S is None else S
    q_pos = torch.arange(S - T, S, device=device)[:, None]    # [T, 1] абсолютные позиции запросов
    k_pos = torch.arange(S, device=device)[None, :]           # [1, S] позиции ключей
    allowed = k_pos <= q_pos                                  # causal: не смотрим в будущее
    if window is not None:
        allowed = allowed & (q_pos - k_pos < window)          # окно: window позиций, включая себя
    return allowed


def layer_types(n_layers: int, sliding_window_pattern: int) -> list[str]:
    """Gemma 2/3: каждый pattern-й слой (считая с 1) — full, остальные — sliding."""
    return ["sliding_attention" if (i + 1) % sliding_window_pattern else "full_attention"
            for i in range(n_layers)]


class SlidingWindowAttention(nn.Module):
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
        """x [B, T, C] → [B, T, C]. Как LlamaAttention без кеша, но с маской-окном."""
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), D, self.rope_theta)
        q = apply_rope(self.q_proj(x).view(B, T, H, D).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2), cos, sin)
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)                       # [B, H, T, T]
        mask = sliding_window_mask(T, window=self.window, device=x.device)   # [T, T]
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))
