"""Расписание learning rate (линейный warmup + cosine decay) и gradient clipping по глобальной норме.

Обе детали стоят в цикле обучения почти любой LLM: lr_at(step) задаёт lr перед каждым шагом,
clip_grad_norm_ вызывается между backward() и step(). Готовые torch.nn.utils.clip_grad_norm_ и
torch.optim.lr_scheduler в этой задаче запрещены.
"""

from __future__ import annotations

import math

import torch


def lr_at(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    """lr на шаге step (с нуля): линейный рост до max_lr за warmup_steps, затем cosine до min_lr к total_steps."""
    # TODO 1: warmup: step = 0 → max_lr / warmup_steps, step = warmup_steps − 1 → max_lr (линейно).
    # TODO 2: после total_steps — min_lr.
    # TODO 3: между ними — доля пройденного пути progress ∈ [0, 1] и косинус от max_lr к min_lr (формула в README).
    #         Что будет при warmup_steps = 0? Деления на ноль быть не должно.
    raise NotImplementedError("lr_at")


@torch.no_grad()
def clip_grad_norm_(params, max_norm: float, eps: float = 1e-6) -> float:
    """Если глобальная L2-норма всех градиентов больше max_norm — масштабирует их все на одно число.

    Возвращает норму ДО обрезки (её удобно логировать). Параметры без .grad пропускаются.
    """
    # TODO 1: глобальная норма — корень из суммы квадратов ВСЕХ элементов ВСЕХ градиентов (не по каждому отдельно).
    # TODO 2: coef = max_norm / (norm + eps); если coef < 1 — умножь каждый градиент на coef на месте.
    # TODO 3: верни норму до обрезки числом float.
    raise NotImplementedError("clip_grad_norm_")
