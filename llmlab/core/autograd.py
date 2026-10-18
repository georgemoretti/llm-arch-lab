"""Скалярный autograd — эталон задачи A01 (блок A, модуль 1).

Задачи A02 и A03 импортируют ``Value`` отсюда, а не из кода пользователя: ошибка в A01 не должна
ломать следующие задачи. Совпадение с ``solution.py`` задачи A01 проверяет ``tests/test_core_autograd.py``.
"""

from __future__ import annotations

import math


class Value:
    """Скаляр, который помнит, из каких значений и какой операцией он получен.

    ``_children`` — входы операции, ``_local_grads`` — производные результата по каждому входу
    (∂self/∂child), посчитанные в момент прямого прохода. ``backward`` применяет chain rule.
    """

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = (), op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._children = tuple(children)
        self._local_grads = tuple(local_grads)
        self._op = op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _wrap(other) -> "Value":
        return other if isinstance(other, Value) else Value(other)

    # --- примитивы: значение + локальные производные ---

    def __add__(self, other) -> "Value":
        other = Value._wrap(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0), "+")

    def __mul__(self, other) -> "Value":
        other = Value._wrap(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data), "*")

    def __pow__(self, k: float) -> "Value":
        assert isinstance(k, (int, float)), "степень — обычное число, не Value"
        return Value(self.data**k, (self,), (k * self.data ** (k - 1),), f"**{k}")

    def exp(self) -> "Value":
        e = math.exp(self.data)
        return Value(e, (self,), (e,), "exp")

    def log(self) -> "Value":
        return Value(math.log(self.data), (self,), (1.0 / self.data,), "log")

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        return Value(t, (self,), (1.0 - t * t,), "tanh")

    def relu(self) -> "Value":
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0 else 0.0,), "relu")

    # --- производные операции: выражаются через примитивы, своих производных не нужно ---

    def __neg__(self) -> "Value":
        return self * -1.0

    def __radd__(self, other) -> "Value":
        return self + other

    def __sub__(self, other) -> "Value":
        return self + (-Value._wrap(other))

    def __rsub__(self, other) -> "Value":
        return Value._wrap(other) + (-self)

    def __rmul__(self, other) -> "Value":
        return self * other

    def __truediv__(self, other) -> "Value":
        return self * Value._wrap(other) ** -1

    def __rtruediv__(self, other) -> "Value":
        return Value._wrap(other) * self**-1

    # --- обратный проход ---

    def backward(self) -> None:
        """Градиенты всех узлов графа по self: ∂self/∂node копится в node.grad."""
        # Топологический порядок без рекурсии: глубина графа может быть больше лимита рекурсии Python.
        order: list[Value] = []
        visited: set[int] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                order.append(node)          # все входы узла уже в order
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((c, False) for c in node._children if id(c) not in visited)

        self.grad = 1.0
        for node in reversed(order):        # от выхода к входам: node.grad уже полный
            for child, local in zip(node._children, node._local_grads):
                child.grad += local * node.grad   # +=: узел может быть входом нескольких операций
