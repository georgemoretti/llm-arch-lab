"""Проверка градиентов: численная производная центральными разностями и сравнение с autograd.

Эталон. Любой новый backward (свой autograd, ручной backprop в A04–A05, кастомная операция в PyTorch)
сначала сверяют с конечными разностями: это медленно, но не требует знать производную.
"""

from __future__ import annotations

from typing import Callable

import torch

from llmlab.core.autograd import Value


def numerical_grad(f: Callable[[torch.Tensor], float], x: torch.Tensor, h: float = 1e-5) -> torch.Tensor:
    """Градиент скалярной функции f в точке x центральными разностями.

    x: тензор любой формы (float64) → тензор той же формы: ∂f/∂x_i ≈ (f(x + h·e_i) − f(x − h·e_i)) / 2h.
    f возвращает число или тензор из одного элемента. x после вызова не меняется.
    """
    x = x.detach().clone()
    grad = torch.zeros_like(x)
    flat, gflat = x.view(-1), grad.view(-1)       # view: правим элементы x на месте
    for i in range(flat.numel()):
        old = flat[i].item()
        flat[i] = old + h
        f_plus = float(f(x))
        flat[i] = old - h
        f_minus = float(f(x))
        flat[i] = old
        gflat[i] = (f_plus - f_minus) / (2 * h)
    return grad


def rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Максимальная по элементам относительная ошибка |a − b| / max(|a|, |b|, eps)."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    denom = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.full_like(a, eps))
    return ((a - b).abs() / denom).max().item()


def check_value_grads(fn: Callable[..., Value], xs: list[float], h: float = 1e-6,
                      tol: float = 1e-6) -> tuple[bool, float]:
    """Сверяет градиенты Value-функции fn(*values) с численными.

    Возвращает (ok, err): err — rel_error между градиентами backward и numerical_grad, ok = err < tol.
    """
    vs = [Value(x) for x in xs]
    fn(*vs).backward()
    analytic = torch.tensor([v.grad for v in vs], dtype=torch.float64)

    def f(t: torch.Tensor) -> float:
        return fn(*[Value(v) for v in t.tolist()]).data

    numeric = numerical_grad(f, torch.tensor(xs, dtype=torch.float64), h)
    err = rel_error(analytic, numeric)
    return err < tol, err
