"""Value: скаляр с графом вычислений и обратный проход.

Реализуй:
1. примитивы __add__, __mul__, __pow__, exp, log, tanh, relu: каждый считает значение и
   локальные производные результата по своим входам;
2. backward(): обход графа от выхода к входам и chain rule.

Конструктор, __repr__ и производные операции (минус, деление, операции с числом слева) уже готовы:
они выражены через примитивы, поэтому заработают, как только заработают примитивы.
torch в этой задаче не нужен — только модуль math.
"""

from __future__ import annotations

import math


class Value:
    """Скаляр, который помнит, из каких значений и какой операцией он получен.

    _children — входы операции, _local_grads — производные результата по каждому входу
    (∂self/∂child), посчитанные в момент прямого прохода. backward применяет chain rule.
    """

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = (), op: str = ""):
        # Конструктор готов — тесты на него опираются, не меняй.
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
    # Образец ответа: Value(<значение>, (входы...), (∂результат/∂вход для каждого входа...), "<имя>")

    def __add__(self, other) -> "Value":
        other = Value._wrap(other)
        # TODO: сумма. Чему равны ∂(a+b)/∂a и ∂(a+b)/∂b?
        raise NotImplementedError("Value.__add__")

    def __mul__(self, other) -> "Value":
        other = Value._wrap(other)
        # TODO: произведение. ∂(a·b)/∂a = ?  ∂(a·b)/∂b = ?
        raise NotImplementedError("Value.__mul__")

    def __pow__(self, k: float) -> "Value":
        assert isinstance(k, (int, float)), "степень — обычное число, не Value"
        # TODO: x**k для любого числа k (k = 2, -1, 0.5 ...). Один вход — self.
        raise NotImplementedError("Value.__pow__")

    def exp(self) -> "Value":
        # TODO: производная exp — она сама. Посчитай math.exp один раз.
        raise NotImplementedError("Value.exp")

    def log(self) -> "Value":
        # TODO: натуральный логарифм и его производная.
        raise NotImplementedError("Value.log")

    def tanh(self) -> "Value":
        # TODO: производную tanh удобно выразить через сам результат t = tanh(x).
        raise NotImplementedError("Value.tanh")

    def relu(self) -> "Value":
        # TODO: max(0, x). Производная в нуле — 0 (так договорились и PyTorch, и тесты).
        raise NotImplementedError("Value.relu")

    # --- производные операции: готовы, выражены через примитивы ---

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
        # TODO 1: упорядочь узлы графа так, чтобы каждый узел шёл ПОСЛЕ всех своих входов
        #         (топологическая сортировка). Граф может быть глубиной в тысячи узлов:
        #         рекурсивный обход упрётся в лимит рекурсии Python — нужен явный стек.
        # TODO 2: ∂self/∂self = 1.
        # TODO 3: иди по узлам от выхода к входам и передавай градиент каждому входу:
        #         вклад = локальная производная × градиент узла. Что делать, если у входа
        #         несколько потребителей (x * x, или x используется в двух ветках)?
        raise NotImplementedError("Value.backward")
