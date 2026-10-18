"""Инициализация весов: Xavier (Glorot) и Kaiming (He), и что происходит с активациями в глубокой сети.

Веса Linear хранятся как W [fan_out, fan_in] (как в nn.Linear): y = x Wᵀ.
Готовые torch.nn.init.* в этой задаче запрещены — тесты это проверяют.
"""

from __future__ import annotations

import math

import torch


def gain(nonlinearity: str) -> float:
    """Множитель масштаба для нелинейности: во сколько раз она «сжимает» дисперсию сигнала."""
    # TODO: "linear" → 1; "relu" → ? (ReLU обнуляет половину входов: что это делает с E[x²]?);
    #       "tanh" → 5/3 (эмпирическое значение, как в PyTorch).
    raise NotImplementedError("gain")


def xavier_uniform_(W: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """U(−a, a), a = gain·sqrt(6 / (fan_in + fan_out)): Var = gain²·2 / (fan_in + fan_out). На месте."""
    # TODO 1: fan_in и fan_out из формы W (какая ось какая?).
    # TODO 2: W.uniform_(-a, a, generator=generator) внутри torch.no_grad(); верни W.
    raise NotImplementedError("xavier_uniform_")


def kaiming_normal_(W: torch.Tensor, nonlinearity: str = "relu",
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """N(0, std²), std = gain / sqrt(fan_in): сохраняет дисперсию прямого прохода. На месте."""
    # TODO: std из gain(nonlinearity) и fan_in; W.normal_(0, std, generator=generator) без графа; верни W.
    raise NotImplementedError("kaiming_normal_")


def activation_stds(init: str, act: str, depth: int = 30, width: int = 256, n: int = 512,
                    seed: int = 0) -> list[float]:
    """Std активаций после каждого из depth слоёв x ← act(x Wᵀ) (без bias) при данной инициализации.

    init: "normal-1" (N(0, 1)), "normal-0.01", "xavier" (gain = 1), "kaiming" (под act).
    act: "linear" | "tanh" | "relu". Вход x ~ N(0, 1) формы [n, width]; все случайности — из одного генератора.
    """
    # TODO 1: g = torch.Generator().manual_seed(seed); x = torch.randn(n, width, generator=g).
    # TODO 2: depth раз: W = torch.empty(width, width), заполнить по init (тем же g), x = act(x @ W.T),
    #         запомнить x.std().item(). Порядок вызовов генератора важен: тесты сверяют числа с эталоном.
    raise NotImplementedError("activation_stds")
