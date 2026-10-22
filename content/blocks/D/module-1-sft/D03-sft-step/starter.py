"""Шаг SFT: cross-entropy только по меченым токенам, накопление градиентов с честной нормировкой, clipping, lr.

Реализуй три функции. Метки — из D02 (``IGNORE = -100`` у всего, кроме ответа). ``lr_at`` (A08) и ``collate`` —
готовые, из ``llmlab.core.sft``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmlab.core.sft import IGNORE, collate, lr_at  # noqa: F401  (collate — для своих экспериментов)


def masked_ce_sum(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Сумма cross-entropy по позициям с меткой (≠ IGNORE) и число таких позиций. logits [B, T, V], targets [B, T]."""
    # TODO 1: F.cross_entropy по сплющенным logits [B·T, V] и targets [B·T] с ignore_index=IGNORE и reduction="sum".
    # TODO 2: число позиций с меткой ≠ IGNORE — как int.
    raise NotImplementedError("masked_ce_sum")


def accumulate_gradients(model, micro_batches: list[tuple]) -> tuple[float, int]:
    """backward по микро-батчам ``(input_ids, targets)`` так, чтобы .grad == градиент среднего по ВСЕМ токенам ответа.

    Возвращает (средний лосс на токен, N), где N — число меченых токенов во всех микро-батчах шага.
    """
    # TODO 1: N = число меток ≠ IGNORE во ВСЕХ микро-батчах (посчитай до прогона). N == 0 → ValueError.
    # TODO 2: для каждого микро-батча: сумма CE (masked_ce_sum по model(input_ids)), backward от (сумма / N).
    #         Градиенты складываются в .grad сами.
    # TODO 3: верни (сумма всех сумм / N как float, N).
    raise NotImplementedError("accumulate_gradients")


def sft_step(model, optimizer, micro_batches: list[tuple], step: int, *, max_lr: float, warmup_steps: int,
             total_steps: int, min_lr: float = 0.0, max_norm: float = 1.0) -> dict:
    """Один шаг оптимизатора. Возвращает dict: loss (средний на токен), tokens (N), lr, grad_norm (до обрезки)."""
    # TODO 1: lr = lr_at(step, max_lr, warmup_steps, total_steps, min_lr) → в каждую группу optimizer.param_groups.
    # TODO 2: optimizer.zero_grad() → accumulate_gradients → torch.nn.utils.clip_grad_norm_(…, max_norm) (она
    #         возвращает норму ДО обрезки) → optimizer.step().
    raise NotImplementedError("sft_step")
