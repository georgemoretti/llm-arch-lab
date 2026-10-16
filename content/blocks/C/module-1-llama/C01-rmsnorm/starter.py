"""C01: RMSNorm, как в Llama.

Реализуй:
1. функцию rms_norm(x, weight, eps) — нормировка каждого вектора по его RMS;
2. метод RMSNorm.forward — вызов rms_norm со своими параметрами.

Нельзя использовать F.rms_norm и nn.RMSNorm — формулу нужно написать руками.
"""

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm по последней оси.

    x: [..., C], weight: [C] → выход [..., C] того же dtype, что и x.
    """
    # TODO 1: в каком dtype безопасно возводить в квадрат? (подумай про fp16 и значения ~300)
    # TODO 2: RMS — корень из среднего квадрата по оси C. Где в этой формуле место для eps?
    #         Какая форма должна быть у делителя, чтобы он транслировался на [..., C]?
    # TODO 3: раздели, верни dtype входа и умножь на weight.
    raise NotImplementedError("rms_norm")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # Конструктор готов: имя и форма параметра проверяются тестами — не меняй их.
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))      # γ, инициализация единицами

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., C] → [..., C]."""
        # TODO: одна строка.
        raise NotImplementedError("RMSNorm.forward")
