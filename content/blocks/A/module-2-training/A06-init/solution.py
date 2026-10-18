"""Инициализация весов: Xavier (Glorot) и Kaiming (He), и что происходит с активациями в глубокой сети.

Эталон. Веса Linear хранятся как W [fan_out, fan_in] (как в nn.Linear): y = x Wᵀ.
"""

from __future__ import annotations

import math

import torch


def gain(nonlinearity: str) -> float:
    """Множитель масштаба для нелинейности: во сколько раз она «сжимает» дисперсию сигнала."""
    return {"linear": 1.0, "tanh": 5.0 / 3.0, "relu": math.sqrt(2.0)}[nonlinearity]


def xavier_uniform_(W: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """U(−a, a), a = gain·sqrt(6 / (fan_in + fan_out)): Var = gain²·2 / (fan_in + fan_out). На месте."""
    fan_out, fan_in = W.shape
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        return W.uniform_(-a, a, generator=generator)


def kaiming_normal_(W: torch.Tensor, nonlinearity: str = "relu",
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """N(0, std²), std = gain / sqrt(fan_in): сохраняет дисперсию прямого прохода. На месте."""
    fan_in = W.shape[1]
    std = gain(nonlinearity) / math.sqrt(fan_in)
    with torch.no_grad():
        return W.normal_(0.0, std, generator=generator)


def activation_stds(init: str, act: str, depth: int = 30, width: int = 256, n: int = 512,
                    seed: int = 0) -> list[float]:
    """Std активаций после каждого из depth слоёв x ← act(x Wᵀ) (без bias) при данной инициализации.

    init: "normal-1" (N(0, 1)), "normal-0.01", "xavier" (gain = 1), "kaiming" (под act).
    act: "linear" | "tanh" | "relu". Вход x ~ N(0, 1) формы [n, width]; все случайности — из одного генератора.
    """
    g = torch.Generator().manual_seed(seed)
    f = {"linear": lambda t: t, "tanh": torch.tanh, "relu": torch.relu}[act]
    x = torch.randn(n, width, generator=g)
    stds = []
    for _ in range(depth):
        W = torch.empty(width, width)
        if init == "normal-1":
            W.normal_(0.0, 1.0, generator=g)
        elif init == "normal-0.01":
            W.normal_(0.0, 0.01, generator=g)
        elif init == "xavier":
            xavier_uniform_(W, 1.0, generator=g)
        elif init == "kaiming":
            kaiming_normal_(W, act, generator=g)
        else:
            raise ValueError(init)
        x = f(x @ W.T)
        stds.append(x.std().item())
    return stds
