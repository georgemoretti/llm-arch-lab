"""LayerNorm: прямой проход и обратный, выведенный руками.

Эталон. Нормировка идёт по последней оси C, все ведущие оси (батч, позиции) — независимые строки.
"""

from __future__ import annotations

import torch


def layernorm_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """y = (x − μ) / sqrt(σ² + eps) · γ + β по последней оси.  x [..., C], gamma/beta [C] → y [..., C], cache."""
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)        # смещённая дисперсия (деление на C)
    rstd = (var + eps).rsqrt()                              # 1 / σ, [..., 1]
    xhat = (x - mu) * rstd
    return xhat * gamma + beta, (xhat, rstd, gamma)


def layernorm_backward(dy: torch.Tensor, cache):
    """dy [..., C] → dx [..., C], dgamma [C], dbeta [C]."""
    xhat, rstd, gamma = cache
    lead = tuple(range(dy.dim() - 1))                       # все оси, кроме C
    dgamma = (dy * xhat).sum(dim=lead)
    dbeta = dy.sum(dim=lead)
    g = dy * gamma                                          # ∂L/∂x̂
    # x̂ зависит от x напрямую, через μ и через σ; два последних пути дают вычитаемые средние.
    dx = rstd * (g - g.mean(dim=-1, keepdim=True) - xhat * (g * xhat).mean(dim=-1, keepdim=True))
    return dx, dgamma, dbeta
