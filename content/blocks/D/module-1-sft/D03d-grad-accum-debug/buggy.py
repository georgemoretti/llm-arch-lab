"""Шаг SFT: cross-entropy только по меченым токенам, накопление градиентов с честной нормировкой, clipping, lr.

Код написал коллега; в нём одна ошибка. Метки — из D02 (``IGNORE = -100`` у всего, кроме ответа). ``lr_at`` (A08)
и ``collate`` — готовые, из ``llmlab.core.sft``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmlab.core.sft import IGNORE, collate, lr_at  # noqa: F401  (collate — для своих экспериментов)


def masked_ce_sum(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Сумма cross-entropy по позициям с меткой (≠ IGNORE) и число таких позиций. logits [B, T, V], targets [B, T]."""
    loss = F.cross_entropy(logits.flatten(0, -2).float(), targets.flatten(), ignore_index=IGNORE, reduction="sum")
    return loss, int((targets != IGNORE).sum())


def accumulate_gradients(model, micro_batches: list[tuple]) -> tuple[float, int]:
    """backward по микро-батчам ``(input_ids, targets)`` так, чтобы .grad == градиент среднего по ВСЕМ токенам ответа.

    Каждый микро-батч вносит ``сумма CE / N``, где N — число меченых токенов во всех микро-батчах шага. Возвращает
    (средний лосс на токен, N).
    """
    n_total = sum(int((t != IGNORE).sum()) for _, t in micro_batches)
    if n_total == 0:
        raise ValueError("в шаге нет ни одного токена ответа: все метки = IGNORE")
    total = 0.0
    for ids, targets in micro_batches:
        loss_sum, n = masked_ce_sum(model(ids), targets)
        if n == 0:
            continue
        (loss_sum / n / len(micro_batches)).backward()
        total += float(loss_sum.detach())
    return total / n_total, n_total


def sft_step(model, optimizer, micro_batches: list[tuple], step: int, *, max_lr: float, warmup_steps: int,
             total_steps: int, min_lr: float = 0.0, max_norm: float = 1.0) -> dict:
    """Один шаг оптимизатора: lr по расписанию → zero_grad → накопление → clip → step. Возвращает статистику."""
    lr = lr_at(step, max_lr, warmup_steps, total_steps, min_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss, n_tokens = accumulate_gradients(model, micro_batches)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    optimizer.step()
    return {"loss": loss, "tokens": n_tokens, "lr": lr, "grad_norm": grad_norm}
