"""Оптимизаторы руками: SGD с momentum, Adam и AdamW.

Интерфейс как у torch.optim: конструктор получает список параметров (листовые тензоры с requires_grad),
step() обновляет их на месте по .grad, zero_grad() сбрасывает градиенты. Параметры без .grad пропускаются.
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
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for p, buf in zip(self.params, self.bufs):
            if p.grad is None:
                continue
            buf.mul_(self.momentum).add_(p.grad)
            p.sub_(self.lr * buf)


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
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            self._update(p, p.grad, m, v)

    def _update(self, p: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor) -> None:
        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
        m_hat = m / (1 - self.beta1**self.t)                        # без поправки первые шаги в разы меньше
        v_hat = v / (1 - self.beta2**self.t)
        p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))


class AdamW(Adam):
    """Adam + decoupled weight decay: θ ← θ·(1 − lr·wd) отдельно от адаптивного шага."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p
            self._update(p, g, m, v)
