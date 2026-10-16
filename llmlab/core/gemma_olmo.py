"""Готовые компоненты модуля 2 блока C (Gemma 3 / OLMo 2 / SmolLM3) — эталоны задач C07–C11.

C07 sliding window attention и паттерн local:global, C08 размещение нормализаций, C09 QK-Norm,
C10 NoPE, C11 KV-cache с кольцевым буфером для SWA. Поздние задачи и финальная модель (Фаза 8)
импортируют их отсюда: ``from llmlab.core.gemma_olmo import sliding_window_mask, ...``.

⚠️ Это готовые ответы на задачи C07–C11. Если ещё не решил их — не подглядывай.

Код намеренно совпадает с ``content/blocks/C/module-2-gemma-olmo/C*/solution.py``; расхождение
ловит ``tests/test_core_gemma_olmo.py``. Имена параметров — как в Hugging Face ``transformers``
(``modeling_gemma3.py``, ``modeling_olmo2.py``, ``modeling_smollm3.py``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import LlamaAttention, RMSNorm, SwiGLUFFN, apply_rope, repeat_kv, rope_cos_sin

# ---------------------------------------------------------------------------- C07 SWA


def sliding_window_mask(T: int, S: int | None = None, window: int | None = None,
                        device: torch.device | None = None) -> torch.Tensor:
    """Разрешённые пары (запрос, ключ) → bool [T, S]. Запросы — последние T из S позиций.

    Запрос на позиции p видит ключ j, если j ≤ p и p − j < window (окно из window позиций,
    включая себя — как в HF). window=None — обычная causal-маска.
    """
    S = T if S is None else S
    q_pos = torch.arange(S - T, S, device=device)[:, None]
    k_pos = torch.arange(S, device=device)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed = allowed & (q_pos - k_pos < window)
    return allowed


def layer_types(n_layers: int, sliding_window_pattern: int) -> list[str]:
    """Gemma 2/3: каждый ``pattern``-й слой (считая с 1) — full, остальные — sliding."""
    return ["sliding_attention" if (i + 1) % sliding_window_pattern else "full_attention"
            for i in range(n_layers)]


class SlidingWindowAttention(nn.Module):
    """GQA + RoPE + маска со скользящим окном (без кеша). window=None — полное causal attention."""

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
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), D, self.rope_theta)
        q = apply_rope(self.q_proj(x).view(B, T, H, D).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2), cos, sin)
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = sliding_window_mask(T, window=self.window, device=x.device)
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


# ---------------------------------------------------------------------------- C08 нормализации


class _Sublayers(nn.Module):
    """Общая часть блоков C08: attention и FFN из модуля 1 + нужные нормы."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_hidden: int, norms: list[str],
                 eps: float = 1e-6, rope_theta: float = 10000.0):
        super().__init__()
        self.self_attn = LlamaAttention(d_model, n_heads, n_kv_heads, rope_theta)
        self.mlp = SwiGLUFFN(d_model, d_hidden)
        for name in norms:
            setattr(self, name, RMSNorm(d_model, eps))


class PostNormBlock(_Sublayers):
    """Оригинальный Transformer (2017): норма ПОСЛЕ сложения с остатком."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["post_attention_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x):
        x = self.post_attention_layernorm(x + self.self_attn(x))
        return self.post_feedforward_layernorm(x + self.mlp(x))


class PreNormBlock(_Sublayers):
    """GPT-2 / Llama: норма на входе подслоя, остаток не нормируется."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["input_layernorm", "post_attention_layernorm"], **kw)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class OlmoPostNormBlock(_Sublayers):
    """OLMo 2: норма на ВЫХОДЕ подслоя, но внутри остаточной ветки."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["post_attention_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x):
        x = x + self.post_attention_layernorm(self.self_attn(x))
        return x + self.post_feedforward_layernorm(self.mlp(x))


class SandwichNormBlock(_Sublayers):
    """Gemma 2/3: норма и на входе, и на выходе подслоя (внутри остаточной ветки)."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["input_layernorm", "post_attention_layernorm",
                          "pre_feedforward_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


# ---------------------------------------------------------------------------- C09 QK-Norm


class QKNormAttention(nn.Module):
    """GQA + RoPE + QK-Norm. qk_norm="head" — по каждой голове (Gemma 3, Qwen3),
    "full" — по всей проекции до разбиения на головы (OLMo 2). Норма — ДО RoPE."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, qk_norm: str = "head",
                 rope_theta: float = 10000.0, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0 and qk_norm in ("head", "full")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        q_dim, k_dim = (self.head_dim, self.head_dim) if qk_norm == "head" else \
            (n_heads * self.head_dim, n_kv_heads * self.head_dim)
        self.q_norm = RMSNorm(q_dim, eps)
        self.k_norm = RMSNorm(k_dim, eps)

    def scores(self, x: torch.Tensor):
        """→ (scores [B, H, T, T] до маски и softmax, v [B, H, T, D])."""
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q, k = self.q_proj(x), self.k_proj(x)
        if self.qk_norm == "full":
            q, k = self.q_norm(q), self.k_norm(k)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H_kv, D).transpose(1, 2)
        if self.qk_norm == "head":
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), D, self.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        return q @ k.transpose(-2, -1) / math.sqrt(D), v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        scores, v = self.scores(x)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


# ---------------------------------------------------------------------------- C10 NoPE


def rope_layers(n_layers: int, no_rope_layer_interval: int) -> list[bool]:
    """SmolLM3: каждый ``interval``-й слой (считая с 1) — без RoPE. True = слой с RoPE."""
    return [(i + 1) % no_rope_layer_interval != 0 for i in range(n_layers)]


class NoPEAttention(nn.Module):
    """GQA + causal-маска; RoPE применяется, только если use_rope=True."""

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
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        if self.use_rope:
            cos, sin = rope_cos_sin(torch.arange(T, device=x.device), D, self.rope_theta)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


def build_attention_layers(n_layers: int, d_model: int, n_heads: int, n_kv_heads: int,
                           no_rope_layer_interval: int, rope_theta: float = 10000.0) -> nn.ModuleList:
    """Стек attention-слоёв SmolLM3: RoPE или NoPE по расписанию rope_layers."""
    return nn.ModuleList(NoPEAttention(d_model, n_heads, n_kv_heads, use_rope, rope_theta)
                         for use_rope in rope_layers(n_layers, no_rope_layer_interval))


# ---------------------------------------------------------------------------- C11 ring KV-cache


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
