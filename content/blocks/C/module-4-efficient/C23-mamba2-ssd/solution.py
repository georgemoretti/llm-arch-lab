"""Эталон C23: упрощённый Mamba-2 (SSD) — рекуррентный скан и эквивалентная ему квадратичная «attention-форма»."""

import torch


def segsum(a: torch.Tensor) -> torch.Tensor:
    """a [..., T] → [..., T, T]: out[t, s] = a_{s+1} + … + a_t при s ≤ t (пустая сумма = 0), −inf при s > t."""
    T = a.shape[-1]
    cs = torch.cumsum(a, dim=-1)                                     # cs_t = a_0 + … + a_t
    out = cs[..., :, None] - cs[..., None, :]                        # cs_t − cs_s = a_{s+1} + … + a_t
    causal = torch.ones(T, T, dtype=torch.bool, device=a.device).tril()
    return out.masked_fill(~causal, float("-inf"))


def ssd_recurrent(x, dt, A, B, C, D, state=None):
    """Скан Mamba-2 по времени (как при генерации).

    x [b, T, H, P], dt [b, T, H] > 0, A [H] < 0, B и C [b, T, N] (общие для голов), D [H],
    state [b, H, N, P] или None → (y [b, T, H, P], state [b, H, N, P]).
    """
    b, T, H, P = x.shape
    N = B.shape[-1]
    h = x.new_zeros(b, H, N, P) if state is None else state
    ys = []
    for t in range(T):
        decay = torch.exp(dt[:, t] * A)                                          # [b, H]: e^{Δ_t A} ∈ (0, 1)
        write = (dt[:, t, :, None, None] * B[:, t, None, :, None]) * x[:, t, :, None, :]   # Δ_t · B_t ⊗ x_t
        h = decay[..., None, None] * h + write                                   # [b, H, N, P]
        y = (C[:, t, None, :, None] * h).sum(dim=-2)                            # C_tᵀ h_t  [b, H, P]
        ys.append(y + D[:, None] * x[:, t])
    return torch.stack(ys, dim=1), h


def ssd_quadratic(x, dt, A, B, C, D):
    """Та же модель целиком: y = (L ⊙ C Bᵀ)(Δ·x) + D·x, где L[t, s] = exp(Σ_{r=s+1..t} Δ_r A) — маска с затуханием."""
    L = torch.exp(segsum((dt * A).transpose(1, 2)))                          # [b, H, T, T], 0 выше диагонали
    scores = C @ B.transpose(1, 2)                                            # [b, T, T]: C_t · B_s
    M = L * scores[:, None]                                                   # [b, H, T, T]
    y = torch.einsum("bhts,bshp->bthp", M, dt[..., None] * x)               # [b, T, H, P]
    return y + D[:, None] * x
