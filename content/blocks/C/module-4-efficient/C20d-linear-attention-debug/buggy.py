"""C20d: линейное внимание (Katharopoulos et al., 2020) — kernel trick, параллельная и рекуррентная формы."""

import torch
import torch.nn.functional as F


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
    z = k.sum(dim=-2)                                               # [B, H, d]
    return (A @ v) / ((q @ z.unsqueeze(-1)) + eps)


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
