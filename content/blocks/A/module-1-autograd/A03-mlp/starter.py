"""MLP на скалярах: нейрон → слой → сеть, MSE-лосс и шаг градиентного спуска.

Реализуй прямой проход и parameters() у Neuron, Layer, MLP, затем mse_loss, zero_grad, sgd_step
и цикл обучения train. Конструкторы готовы: порядок создания весов важен для воспроизводимости.
"""

from __future__ import annotations

import random

from llmlab.core.autograd import Value   # готовый Value (эталон задачи «Autograd: Value»)


class Neuron:
    """y = tanh(w · x + b) (или без tanh в выходном слое)."""

    def __init__(self, nin: int, rng: random.Random, nonlin: bool = True):
        # Готово, не меняй: веса из U(−1, 1), bias = 0.
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list) -> Value:
        # TODO: b + Σ wᵢ·xᵢ (x — числа или Value), затем tanh, если self.nonlin.
        raise NotImplementedError("Neuron.__call__")

    def parameters(self) -> list[Value]:
        # TODO: сначала веса по порядку входов, затем bias.
        raise NotImplementedError("Neuron.parameters")


class Layer:
    """nout независимых нейронов на одном входе."""

    def __init__(self, nin: int, nout: int, rng: random.Random, nonlin: bool = True):
        self.neurons = [Neuron(nin, rng, nonlin) for _ in range(nout)]

    def __call__(self, x: list) -> list[Value]:
        # TODO: выход каждого нейрона на одном и том же x.
        raise NotImplementedError("Layer.__call__")

    def parameters(self) -> list[Value]:
        # TODO: параметры нейронов подряд, в порядке нейронов.
        raise NotImplementedError("Layer.parameters")


class MLP:
    """Слои nin → nouts[0] → … → nouts[-1]; tanh во всех слоях, кроме последнего."""

    def __init__(self, nin: int, nouts: list[int], seed: int = 0):
        rng = random.Random(seed)
        sizes = [nin] + list(nouts)
        self.layers = [Layer(sizes[i], sizes[i + 1], rng, nonlin=i < len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x: list) -> list[Value]:
        # TODO: выход слоя — вход следующего. Возвращай список (даже если выход один).
        raise NotImplementedError("MLP.__call__")

    def parameters(self) -> list[Value]:
        # TODO: параметры слоёв подряд.
        raise NotImplementedError("MLP.parameters")


def mse_loss(preds: list[Value], ys: list[float]) -> Value:
    """Среднее (pred − y)² по примерам."""
    # TODO: сумма квадратов разностей, делённая на число примеров. Результат — Value.
    raise NotImplementedError("mse_loss")


def zero_grad(params: list[Value]) -> None:
    # TODO: зачем обнулять? (вспомни, что backward делает с .grad — присваивает или прибавляет?)
    raise NotImplementedError("zero_grad")


def sgd_step(params: list[Value], lr: float) -> None:
    """Шаг против градиента: θ ← θ − lr · ∂L/∂θ."""
    # TODO: меняй .data, а не создавай новые Value — модель держит ссылки на эти объекты.
    raise NotImplementedError("sgd_step")


def train(model: MLP, xs: list[list[float]], ys: list[float], steps: int, lr: float) -> list[float]:
    """Полный батч: forward по всем примерам → MSE → zero_grad → backward → SGD. Возвращает лосс каждого шага."""
    # TODO: на каждом шаге — предсказание model(x)[0] для всех x, лосс, обнуление градиентов,
    #       backward, шаг SGD; сохрани loss.data.
    raise NotImplementedError("train")
