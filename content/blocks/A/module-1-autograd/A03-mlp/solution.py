"""MLP на скалярах: нейрон → слой → сеть, MSE-лосс и шаг градиентного спуска.

Эталон. Всё строится на готовом Value: прямой проход собирает граф, loss.backward() даёт градиенты
всех весов, шаг SGD сдвигает их против градиента.
"""

from __future__ import annotations

import random

from llmlab.core.autograd import Value


class Neuron:
    """y = tanh(w · x + b) (или без tanh в выходном слое)."""

    def __init__(self, nin: int, rng: random.Random, nonlin: bool = True):
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list) -> Value:
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """nout независимых нейронов на одном входе."""

    def __init__(self, nin: int, nout: int, rng: random.Random, nonlin: bool = True):
        self.neurons = [Neuron(nin, rng, nonlin) for _ in range(nout)]

    def __call__(self, x: list) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Слои nin → nouts[0] → … → nouts[-1]; tanh во всех слоях, кроме последнего."""

    def __init__(self, nin: int, nouts: list[int], seed: int = 0):
        rng = random.Random(seed)
        sizes = [nin] + list(nouts)
        self.layers = [Layer(sizes[i], sizes[i + 1], rng, nonlin=i < len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x: list) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def mse_loss(preds: list[Value], ys: list[float]) -> Value:
    """Среднее (pred − y)² по примерам."""
    total = Value(0.0)
    for p, y in zip(preds, ys):
        total = total + (p - y) ** 2
    return total / len(preds)


def zero_grad(params: list[Value]) -> None:
    for p in params:
        p.grad = 0.0


def sgd_step(params: list[Value], lr: float) -> None:
    """Шаг против градиента: θ ← θ − lr · ∂L/∂θ."""
    for p in params:
        p.data -= lr * p.grad


def train(model: MLP, xs: list[list[float]], ys: list[float], steps: int, lr: float) -> list[float]:
    """Полный батч: forward по всем примерам → MSE → zero_grad → backward → SGD. Возвращает лосс каждого шага."""
    losses = []
    for _ in range(steps):
        preds = [model(x)[0] for x in xs]
        loss = mse_loss(preds, ys)
        zero_grad(model.parameters())
        loss.backward()
        sgd_step(model.parameters(), lr)
        losses.append(loss.data)
    return losses
