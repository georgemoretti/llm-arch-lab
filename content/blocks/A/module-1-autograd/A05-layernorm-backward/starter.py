"""LayerNorm: прямой проход и обратный, выведенный руками.

Нормировка идёт по последней оси C, все ведущие оси (батч, позиции) — независимые строки.
Нельзя: F.layer_norm, nn.LayerNorm, .backward() и torch.autograd — тесты это проверяют.
"""

from __future__ import annotations

import torch


def layernorm_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """y = (x − μ) / sqrt(σ² + eps) · γ + β по последней оси.  x [..., C], gamma/beta [C] → y [..., C], cache."""
    # TODO 1: μ и σ² по последней оси (keepdim=True). Дисперсия — делением на C или на C − 1? (как в PyTorch)
    # TODO 2: x̂ = (x − μ) / sqrt(σ² + eps), y = x̂·γ + β.
    # TODO 3: что сохранить в cache, чтобы backward не пересчитывал статистики?
    raise NotImplementedError("layernorm_forward")


def layernorm_backward(dy: torch.Tensor, cache):
    """dy [..., C] → dx [..., C], dgamma [C], dbeta [C]."""
    # TODO 1: dgamma и dbeta — формы [C]: по каким осям суммировать?
    # TODO 2: g = ∂L/∂x̂ = dy·γ.
    # TODO 3: x̂ зависит от x тремя путями: напрямую, через μ и через σ. Выведи формулу (см. README)
    #         и проверь её свойство: сумма dx по оси C равна нулю для каждой строки.
    raise NotImplementedError("layernorm_backward")
