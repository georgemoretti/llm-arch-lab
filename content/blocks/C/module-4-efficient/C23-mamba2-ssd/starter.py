"""C23 (hard, опционально): упрощённый Mamba-2 (SSD) — рекуррентный скан и эквивалентная ему «attention-форма».

Реализуй:
1. segsum — суммы a_{s+1} + … + a_t для всех пар s ≤ t (−inf над диагональю);
2. ssd_recurrent — скан по времени с состоянием h [b, H, N, P] (как при генерации);
3. ssd_quadratic — то же целиком через матрицу [T, T]: маска с затуханием ⊙ C·Bᵀ (как attention при обучении).
"""

import torch


def segsum(a: torch.Tensor) -> torch.Tensor:
    """a [..., T] → [..., T, T]: out[t, s] = a_{s+1} + … + a_t при s ≤ t (пустая сумма = 0), −inf при s > t."""
    # TODO 1: cumsum по последней оси; сумма a_{s+1..t} = cs_t − cs_s (броадкаст [..., T, 1] − [..., 1, T]).
    # TODO 2: над диагональю (s > t) — −inf. Что будет на диагонали?
    raise NotImplementedError("segsum")


def ssd_recurrent(x, dt, A, B, C, D, state=None):
    """Скан Mamba-2 по времени (как при генерации).

    x [b, T, H, P], dt [b, T, H] > 0, A [H] < 0, B и C [b, T, N] (общие для голов), D [H],
    state [b, H, N, P] или None → (y [b, T, H, P], state [b, H, N, P]).
    """
    # TODO 1: h — нули [b, H, N, P], если state нет.
    # TODO 2: цикл по t: затухание e^{Δ_t·A} [b, H]; запись Δ_t · (B_t ⊗ x_t) → [b, H, N, P] (B_t общий для голов);
    #         h = затухание · h + запись.
    # TODO 3: y_t = C_tᵀ h_t (сумма по N) + D · x_t  → [b, H, P]; собрать по оси 1.
    raise NotImplementedError("ssd_recurrent")


def ssd_quadratic(x, dt, A, B, C, D):
    """Та же модель целиком: y = (L ⊙ C Bᵀ)(Δ·x) + D·x, где L[t, s] = exp(Σ_{r=s+1..t} Δ_r A) — маска с затуханием."""
    # TODO 1: L = exp(segsum(Δ·A)) по каждой голове: (dt * A) [b, T, H] → [b, H, T] → [b, H, T, T].
    # TODO 2: scores = C·Bᵀ [b, T, T] (как q·kᵀ); M = L ⊙ scores (по головам).
    # TODO 3: y_t = Σ_s M[t, s] · Δ_s x_s  (einsum "bhts,bshp->bthp") + D · x.
    raise NotImplementedError("ssd_quadratic")
