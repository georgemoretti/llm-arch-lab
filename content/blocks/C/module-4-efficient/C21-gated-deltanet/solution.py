"""Эталон: DeltaNet → Gated DeltaNet (Qwen3-Next) — рекуррентная форма правила дельты с гейтом забывания."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
