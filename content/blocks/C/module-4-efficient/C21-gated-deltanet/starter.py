"""C21: DeltaNet → Gated DeltaNet (Qwen3-Next) — рекуррентная форма правила дельты с гейтом забывания.

Реализуй:
1. l2norm — нормировка векторов на единичную длину;
2. gated_delta_rule — шаг за шагом: (забыть) → прочитать старое значение по ключу → записать разность → прочитать выход;
3. GatedDeltaNet.gates и GatedDeltaNet.forward — слой вокруг этой рекуррентности.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x / ‖x‖ по последней оси (как l2norm в HF / FLA)."""
    # TODO: x · rsqrt(Σx² + eps) по последней оси (eps внутри корня, как в HF/FLA).
    raise NotImplementedError("l2norm")


def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                     g: torch.Tensor | None = None, state: torch.Tensor | None = None):
    """Рекуррентная форма (Gated) DeltaNet.

    q, k [B, H, T, dk] (уже L2-нормированы), v [B, H, T, dv], beta [B, H, T] ∈ (0, 1),
    g [B, H, T] ≤ 0 — лог-гейт забывания (None → DeltaNet без гейта), state [B, H, dk, dv] или None.
    → (out [B, H, T, dv], state [B, H, dk, dv]).
    """
    # TODO 1: S — нули [B, H, dk, dv], если state нет; запросы масштабировать на 1/√dk.
    # TODO 2: цикл по t:
    #   - если g есть: S ← exp(g_t)·S (до записи!);
    #   - v_old = Sᵀk_t  [B, H, dv] — что память сейчас «помнит» по ключу k_t;
    #   - S ← S + k_t ⊗ β_t(v_t − v_old)   (в C20 было бы S + k_t ⊗ v_t — чем это хуже?);
    #   - o_t = S_tᵀ q_t  [B, H, dv].
    # TODO 3: вернуть (выходы по оси 2, итоговое S).
    raise NotImplementedError("gated_delta_rule")


class GatedDeltaNet(nn.Module):
    """Упрощённый слой Gated DeltaNet: проекции → L2-норма q, k → β, g → правило дельты → o_proj.

    Без короткой свёртки (conv1d), выходного гейта с RMSNorm и разного числа голов q/k и v — см. README.
    """

    def __init__(self, d_model: int, n_heads: int, head_k_dim: int, head_v_dim: int):
        super().__init__()                                  # конструктор не меняй
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
        # TODO: β = σ(b_proj(x)); g = −exp(A_log)·softplus(a_proj(x) + dt_bias). Оба [B, T, H] → [B, H, T].
        raise NotImplementedError("GatedDeltaNet.gates")


    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """x [B, T, C] → (y [B, T, C], state [B, H, dk, dv])."""
        # TODO 1: q, k, v → головы [B, H, T, ·]; q и k — l2norm по последней оси (после разбиения на головы).
        # TODO 2: β, g = self.gates(x); gated_delta_rule(..., state).
        # TODO 3: [B, H, T, dv] → [B, T, H·dv] → o_proj; вернуть (y, state).
        raise NotImplementedError("GatedDeltaNet.forward")
