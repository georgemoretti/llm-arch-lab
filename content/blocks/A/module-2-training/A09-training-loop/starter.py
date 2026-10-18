"""Тренировочный цикл: батчи, шаг обучения в правильном порядке, оценка на валидации, smoke-тест переобучением.

Работает с любым nn.Module, функцией потерь и оптимизатором из torch.optim (их можно использовать).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def get_batch(X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator | None = None):
    """Случайный батч с возвращением: индексы torch.randint(0, N, (batch_size,), generator=…)."""
    # TODO: ровно один вызов torch.randint с generator — тесты сверяют последовательность батчей с эталоном.
    raise NotImplementedError("get_batch")


@torch.no_grad()
def evaluate(model: nn.Module, loss_fn: Callable, X: torch.Tensor, y: torch.Tensor) -> float:
    """Лосс на всём наборе в режиме eval (без dropout) и без графа; режим модели восстанавливается."""
    # TODO: запомни model.training, переключи в eval, посчитай лосс (.item()), верни прежний режим.
    raise NotImplementedError("evaluate")


def train_loop(model: nn.Module, loss_fn: Callable, optimizer: torch.optim.Optimizer,
               X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, *,
               steps: int, batch_size: int, lr_fn: Callable[[int], float] | None = None,
               max_grad_norm: float | None = None, eval_every: int = 50,
               generator: torch.Generator | None = None) -> dict:
    """Обучение на случайных батчах. Возвращает {"train": [лосс каждого шага], "val": [(шаг, лосс), …]}.

    Валидация — после шагов s, где (s + 1) % eval_every == 0, и после последнего шага.
    """
    # TODO: на каждом шаге в правильном порядке:
    #   1) если есть lr_fn — выставить lr во всех optimizer.param_groups;
    #   2) батч (get_batch) → лосс;
    #   3) zero_grad → backward → (clip_grad_norm_, если задан max_grad_norm) → step;
    #   4) записать loss.item(); при необходимости — evaluate на валидации.
    #   Какие из шагов нельзя переставить и почему?
    raise NotImplementedError("train_loop")


def overfit_one_batch(model: nn.Module, loss_fn: Callable, optimizer: torch.optim.Optimizer,
                      xb: torch.Tensor, yb: torch.Tensor, steps: int) -> list[float]:
    """Smoke-тест: много шагов на ОДНОМ батче. Исправный код доводит лосс почти до нуля."""
    # TODO: тот же шаг обучения, но всегда на xb, yb. Верни список лоссов.
    raise NotImplementedError("overfit_one_batch")
