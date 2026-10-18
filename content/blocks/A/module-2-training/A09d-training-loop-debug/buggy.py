"""Тренировочный цикл: батчи, шаг обучения в правильном порядке, оценка на валидации, smoke-тест переобучением.

Работает с любым nn.Module, функцией потерь и оптимизатором из torch.optim.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def get_batch(X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator | None = None):
    """Случайный батч с возвращением: индексы torch.randint(0, N, (batch_size,), generator=…)."""
    idx = torch.randint(0, X.shape[0], (batch_size,), generator=generator)
    return X[idx], y[idx]


@torch.no_grad()
def evaluate(model: nn.Module, loss_fn: Callable, X: torch.Tensor, y: torch.Tensor) -> float:
    """Лосс на всём наборе в режиме eval (без dropout) и без графа; режим модели восстанавливается."""
    was_training = model.training
    model.eval()
    loss = loss_fn(model(X), y).item()
    model.train(was_training)
    return loss


def train_loop(model: nn.Module, loss_fn: Callable, optimizer: torch.optim.Optimizer,
               X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, *,
               steps: int, batch_size: int, lr_fn: Callable[[int], float] | None = None,
               max_grad_norm: float | None = None, eval_every: int = 50,
               generator: torch.Generator | None = None) -> dict:
    """Обучение на случайных батчах. Возвращает {"train": [лосс каждого шага], "val": [(шаг, лосс), …]}.

    Валидация — после шагов s, где (s + 1) % eval_every == 0, и после последнего шага.
    """
    history: dict = {"train": [], "val": []}
    model.train()
    for step in range(steps):
        if lr_fn is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_fn(step)
        xb, yb = get_batch(X_train, y_train, batch_size, generator)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        history["train"].append(loss.item())
        if (step + 1) % eval_every == 0 or step == steps - 1:
            history["val"].append((step, evaluate(model, loss_fn, X_val, y_val)))
    return history


def overfit_one_batch(model: nn.Module, loss_fn: Callable, optimizer: torch.optim.Optimizer,
                      xb: torch.Tensor, yb: torch.Tensor, steps: int) -> list[float]:
    """Smoke-тест: много шагов на ОДНОМ батче. Исправный код доводит лосс почти до нуля."""
    model.train()
    losses = []
    for _ in range(steps):
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
