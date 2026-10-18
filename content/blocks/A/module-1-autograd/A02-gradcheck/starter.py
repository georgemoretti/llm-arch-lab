"""Проверка градиентов: численная производная и сравнение с autograd.

Реализуй:
1. numerical_grad(f, x, h) — градиент скалярной функции центральными разностями;
2. rel_error(a, b, eps) — максимальная относительная ошибка между двумя тензорами;
3. check_value_grads(fn, xs, h, tol) — проверка backward у Value-функции.

torch.autograd здесь не нужен: функции f могут быть написаны на чистом Python.
"""

from __future__ import annotations

from typing import Callable

import torch

from llmlab.core.autograd import Value   # готовый Value (эталон задачи «Autograd: Value»)


def numerical_grad(f: Callable[[torch.Tensor], float], x: torch.Tensor, h: float = 1e-5) -> torch.Tensor:
    """Градиент скалярной функции f в точке x центральными разностями.

    x: тензор любой формы (float64) → тензор той же формы: ∂f/∂x_i ≈ (f(x + h·e_i) − f(x − h·e_i)) / 2h.
    f возвращает число или тензор из одного элемента. x после вызова не меняется.
    """
    # TODO 1: работай с копией x — исходный тензор менять нельзя.
    # TODO 2: перебери элементы по одному (удобно через плоский view той же памяти: .view(-1)).
    #         Для каждого: сдвинь на +h, посчитай f, сдвинь на −h, посчитай f, верни старое значение.
    # TODO 3: почему разность центральная, а не (f(x + h) − f(x)) / h? Сравни ошибки (см. README).
    raise NotImplementedError("numerical_grad")


def rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Максимальная по элементам относительная ошибка |a − b| / max(|a|, |b|, eps)."""
    # TODO: поэлементно, затем максимум. Зачем eps в знаменателе? Что будет, если a = b = 0?
    raise NotImplementedError("rel_error")


def check_value_grads(fn: Callable[..., Value], xs: list[float], h: float = 1e-6,
                      tol: float = 1e-6) -> tuple[bool, float]:
    """Сверяет градиенты Value-функции fn(*values) с численными.

    Возвращает (ok, err): err — rel_error между градиентами backward и numerical_grad, ok = err < tol.
    """
    # TODO 1: аналитика — Value из каждого xs, fn(...).backward(), собери .grad в тензор float64.
    # TODO 2: численно — оберни fn в функцию от тензора: новые Value из элементов, вернуть .data.
    # TODO 3: сравни через rel_error.
    raise NotImplementedError("check_value_grads")
