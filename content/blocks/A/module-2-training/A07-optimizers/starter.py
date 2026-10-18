"""Оптимизаторы руками: SGD с momentum, Adam и AdamW.

Интерфейс как у torch.optim: конструктор получает список параметров (листовые тензоры с requires_grad),
step() обновляет их на месте по .grad, zero_grad() сбрасывает градиенты. Параметры без .grad пропускаются.
torch.optim в этой задаче не используется — тесты это проверяют. Конструкторы готовы.
"""

from __future__ import annotations

import torch


class SGD:
    """θ ← θ − lr·buf, buf ← momentum·buf + g (при momentum = 0 — обычный градиентный спуск)."""

    def __init__(self, params, lr: float, momentum: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.bufs = [torch.zeros_like(p) for p in self.params]     # «скорость» для каждого параметра

    def zero_grad(self) -> None:
        # TODO: сбрось .grad у каждого параметра (можно в None — как делает PyTorch по умолчанию).
        raise NotImplementedError("SGD.zero_grad")

    @torch.no_grad()
    def step(self) -> None:
        # TODO: для каждого параметра с градиентом: обнови buf на месте, затем сдвинь p на месте.
        #       Почему на месте (mul_, add_, sub_), а не p = p − …?
        raise NotImplementedError("SGD.step")


class Adam:
    """Адаптивный шаг: m — среднее градиента, v — среднее его квадрата, обе оценки с поправкой на старт с нуля."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0                                                  # номер шага (для bias correction)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        # TODO: как у SGD.
        raise NotImplementedError("Adam.zero_grad")

    @torch.no_grad()
    def step(self) -> None:
        # TODO 1: увеличь self.t (один раз на шаг, а не на параметр).
        # TODO 2: m ← β1·m + (1 − β1)·g;  v ← β2·v + (1 − β2)·g²  (на месте).
        # TODO 3: поправка на старт с нуля: m̂ = m / (1 − β1^t), v̂ = v / (1 − β2^t).
        # TODO 4: p ← p − lr · m̂ / (√v̂ + eps). Где именно стоит eps — под корнем или после?
        raise NotImplementedError("Adam.step")


class AdamW(Adam):
    """Adam + decoupled weight decay: θ ← θ·(1 − lr·wd) отдельно от адаптивного шага."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self) -> None:
        # TODO: тот же Adam, но перед адаптивным шагом каждый параметр умножается на (1 − lr·wd).
        #       Почему нельзя просто прибавить wd·θ к градиенту? (README, раздел «Формула»)
        raise NotImplementedError("AdamW.step")
