"""Готовые компоненты эффективного внимания (модуль 4 блока C) — эталоны задач C18–C21.

C18 sparse attention в стиле DSA (DeepSeek-V3.2: lightning indexer + top-k ключей), C19 gated attention и
partial RoPE (Qwen3-Next), C20 линейное внимание (Katharopoulos et al., 2020), C21 Gated DeltaNet.
Поздние задачи (гибридный стек C22) и финальная модель (Фаза 8) импортируют их отсюда.

⚠️ Это готовые ответы на задачи C18–C21. Если ещё не решил их — не подглядывай.

Код намеренно совпадает с ``solution.py`` задач C18–C21 в ``content/blocks/C/module-4-efficient/``;
расхождение ловит ``tests/test_core_efficient.py``. Имена параметров C19 и C21 — как в Hugging Face
``transformers`` (``modeling_qwen3_next.py``: ``q_proj`` с гейтом, ``q_norm``, ``k_norm``, ``A_log``, ``dt_bias``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm, apply_rope, repeat_kv, rope_cos_sin

# ---------------------------------------------------------------------------- C18 DSA


class LightningIndexer(nn.Module):
    """Лёгкий «оценщик» пар (запрос t, ключ s): несколько маленьких голов, ReLU, взвешенная сумма.

    I(t, s) = Σ_h w_{t,h} · ReLU(q_{t,h} · k_s) / √(H_I · d_I); ключ индексатора один на токен (как MQA).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.weights_proj = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → оценки [B, T, T]; будущие ключи (s > t) = −inf."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)                 # [B, T, H_I, d_I]
        k = self.k_proj(x)                                                         # [B, T, d_I]
        w = self.weights_proj(x) / math.sqrt(self.n_heads)                         # [B, T, H_I]
        logits = torch.einsum("bthd,bsd->bths", q, k) / math.sqrt(self.head_dim)   # [B, T, H_I, S]
        scores = torch.einsum("bth,bths->bts", w, F.relu(logits))                  # [B, T, S]
        future = ~torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        return scores.masked_fill(future, float("-inf"))


def topk_mask(index_scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """index_scores [B, T, S] → bool [B, T, S]: True у top_k лучших ДОПУСТИМЫХ (не −inf) ключей строки."""
    k = min(top_k, index_scores.shape[-1])
    idx = index_scores.topk(k, dim=-1).indices                                     # [B, T, k]
    keep = torch.zeros_like(index_scores, dtype=torch.bool).scatter_(-1, idx, True)
    return keep & torch.isfinite(index_scores)          # в коротких строках topk захватывает и −inf


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """q [B, H, T, d], k/v [B, H, S, d], keep [B, T, S] (общая для всех голов) → [B, H, T, d]."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])                      # [B, H, T, S]
    scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))
    return F.softmax(scores, dim=-1) @ v


class DSAttention(nn.Module):
    """MHA, в которой каждый запрос смотрит только на top_k ключей, выбранных индексатором."""

    def __init__(self, d_model: int, n_heads: int, index_n_heads: int, index_head_dim: int, top_k: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.indexer = LightningIndexer(d_model, index_n_heads, index_head_dim)

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        """x [B, T, C] → [B, T, C] (и маска выбранных ключей [B, T, T], если return_mask)."""
        B, T, C = x.shape
        H, d = self.n_heads, self.head_dim
        q, k, v = (p(x).view(B, T, H, d).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        keep = topk_mask(self.indexer(x), self.top_k)                              # [B, T, T]
        out = sparse_attention(q, k, v, keep)                                      # [B, H, T, d]
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, C))
        return (out, keep) if return_mask else out


# ---------------------------------------------------------------------------- C19 gated attention


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D], cos/sin [T, R] (R ≤ D, чётное) → RoPE на первых R каналах, остальные D − R без изменений."""
    R = cos.shape[-1]
    x_rot, x_pass = x[..., :R], x[..., R:]
    return torch.cat((apply_rope(x_rot, cos, sin), x_pass), dim=-1)


class GatedAttention(nn.Module):
    """GQA + QK-Norm + partial RoPE + выходной гейт: out = o_proj(attn(x) ⊙ σ(gate(x)))."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 partial_rotary_factor: float = 0.25, rope_theta: float = 10000.0, eps: float = 1e-6):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.rope_theta = rope_theta
        # q_proj выдаёт на каждую голову [запрос (head_dim) | гейт (head_dim)] — как в HF Qwen3-Next
        self.q_proj = nn.Linear(d_model, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q, gate = self.q_proj(x).view(B, T, H, 2 * D).chunk(2, dim=-1)              # [B, T, H, D] × 2
        gate = gate.reshape(B, T, H * D)
        q = self.q_norm(q).transpose(1, 2)                                            # [B, H, T, D]
        k = self.k_norm(self.k_proj(x).view(B, T, H_kv, D)).transpose(1, 2)           # [B, H_kv, T, D]
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), self.rotary_dim, self.rope_theta)  # [T, R]
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        k, v = repeat_kv(k, H // H_kv), repeat_kv(v, H // H_kv)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v         # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o_proj(out * torch.sigmoid(gate))


# ---------------------------------------------------------------------------- C20 linear attention


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """φ(x) = elu(x) + 1 > 0 — заменяет exp(q·k) скалярным произведением φ(q)·φ(k)."""
    return F.elu(x) + 1


def linear_attention_noncausal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """q, k [B, H, T, d], v [B, H, T, dv] → [B, H, T, dv]; каждый запрос видит все ключи, без матрицы [T, T]."""
    q, k = feature_map(q), feature_map(k)
    kv = k.transpose(-2, -1) @ v                                    # [B, H, d, dv] — сначала φ(K)ᵀV
    z = k.sum(dim=-2)                                               # [B, H, d]
    return (q @ kv) / ((q @ z.unsqueeze(-1)) + eps)                 # [B, H, T, dv] / [B, H, T, 1]


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Causal, параллельная форма (для обучения): q, k [B, H, T, d], v [B, H, T, dv] → [B, H, T, dv]."""
    q, k = feature_map(q), feature_map(k)
    A = (q @ k.transpose(-2, -1)).tril()                            # [B, H, T, T], A_ts = φ(q_t)·φ(k_s), s ≤ t
    return (A @ v) / (A.sum(dim=-1, keepdim=True) + eps)            # нормировка — только по прошлым ключам


def linear_attention_recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state=None, eps: float = 1e-6):
    """Causal, рекуррентная форма (для генерации): токен за токеном, память O(d·dv) вместо O(T).

    state = (S [B, H, d, dv], z [B, H, d]) или None → (out [B, H, T, dv], новый state).
    """
    q, k = feature_map(q), feature_map(k)
    B, H, T, d = q.shape
    dv = v.shape[-1]
    if state is None:
        S, z = q.new_zeros(B, H, d, dv), q.new_zeros(B, H, d)
    else:
        S, z = state
    outs = []
    for t in range(T):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]           # S_t = S_{t−1} + φ(k_t) v_tᵀ
        z = z + k[:, :, t]                                          # z_t = z_{t−1} + φ(k_t)
        num = (q[:, :, t, :, None] * S).sum(dim=-2)                 # φ(q_t)ᵀ S_t   [B, H, dv]
        den = (q[:, :, t] * z).sum(dim=-1, keepdim=True)            # φ(q_t)ᵀ z_t   [B, H, 1]
        outs.append(num / (den + eps))
    return torch.stack(outs, dim=2), (S, z)


# ---------------------------------------------------------------------------- C21 Gated DeltaNet


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x / ‖x‖ по последней оси (как l2norm в HF / FLA)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                     g: torch.Tensor | None = None, state: torch.Tensor | None = None):
    """Рекуррентная форма (Gated) DeltaNet.

    q, k [B, H, T, dk] (уже L2-нормированы), v [B, H, T, dv], beta [B, H, T] ∈ (0, 1),
    g [B, H, T] ≤ 0 — лог-гейт забывания (None → DeltaNet без гейта), state [B, H, dk, dv] или None.
    → (out [B, H, T, dv], state [B, H, dk, dv]).
    """
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    S = k.new_zeros(B, H, dk, dv) if state is None else state
    q = q / dk ** 0.5
    outs = []
    for t in range(T):
        if g is not None:
            S = S * g[:, :, t].exp()[..., None, None]                 # S ← α_t S: забыть часть памяти
        k_t, v_t = k[:, :, t], v[:, :, t]                             # [B, H, dk], [B, H, dv]
        v_old = (S * k_t[..., None]).sum(dim=-2)                      # Sᵀk_t — что память сейчас отдаёт по k_t
        delta = beta[:, :, t, None] * (v_t - v_old)                   # [B, H, dv]
        S = S + k_t[..., None] * delta[..., None, :]                  # S ← S + k_t (β(v_t − Sᵀk_t))ᵀ
        outs.append((S * q[:, :, t, :, None]).sum(dim=-2))            # o_t = S_tᵀ q_t
    return torch.stack(outs, dim=2), S


class GatedDeltaNet(nn.Module):
    """Упрощённый слой Gated DeltaNet: проекции → L2-норма q, k → β, g → правило дельты → o_proj.

    Без короткой свёртки (conv1d), выходного гейта с RMSNorm и разного числа голов q/k и v — см. README.
    """

    def __init__(self, d_model: int, n_heads: int, head_k_dim: int, head_v_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_k_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_k_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_v_dim, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)        # β = σ(b)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)        # g = −exp(A_log)·softplus(a + dt_bias)
        self.A_log = nn.Parameter(torch.log(torch.empty(n_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.ones(n_heads))
        self.o_proj = nn.Linear(n_heads * head_v_dim, d_model, bias=False)

    def gates(self, x: torch.Tensor):
        """x [B, T, C] → (beta [B, H, T] ∈ (0, 1), g [B, H, T] ≤ 0)."""
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)
        g = -self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)
        return beta, g.transpose(1, 2)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """x [B, T, C] → (y [B, T, C], state [B, H, dk, dv])."""
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.head_k_dim, self.head_v_dim
        q = l2norm(self.q_proj(x).view(B, T, H, dk)).transpose(1, 2)               # [B, H, T, dk]
        k = l2norm(self.k_proj(x).view(B, T, H, dk)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dv).transpose(1, 2)                       # [B, H, T, dv]
        beta, g = self.gates(x)
        out, state = gated_delta_rule(q, k, v, beta, g, state)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv)), state
