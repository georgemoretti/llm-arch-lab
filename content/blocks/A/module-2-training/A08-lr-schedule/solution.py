"""Расписание learning rate (линейный warmup + cosine decay) и gradient clipping по глобальной норме.

Эталон. Обе детали стоят в цикле обучения почти любой LLM: lr_at(step) задаёт lr перед каждым шагом,
clip_grad_norm_ вызывается между backward() и step().
"""

from __future__ import annotations

import math

import torch


def lr_at(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    """lr на шаге step (с нуля): линейный рост до max_lr за warmup_steps, затем cosine до min_lr к total_steps."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps        # шаг 0 уже с ненулевым lr
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)   # 0 → 1
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def clip_grad_norm_(params, max_norm: float, eps: float = 1e-6) -> float:
    """Если глобальная L2-норма всех градиентов больше max_norm — масштабирует их все на одно число.

    Возвращает норму ДО обрезки (её удобно логировать). Параметры без .grad пропускаются.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)).item()   # норма склеенного вектора
    coef = max_norm / (total + eps)
    if coef < 1.0:
        for g in grads:
            g.mul_(coef)                                  # направление сохраняется, меняется только длина
    return total
