"""Маска лосса SFT: метки только у токенов ответа, сдвиг на один токен, упаковка диалогов в одну строку.

Реализуй четыре функции. Соглашение: ``targets[t]`` — токен, который модель должна предсказать на позиции ``t``
(``ids[t + 1]``), или ``IGNORE = -100`` (так ``F.cross_entropy(..., ignore_index=-100)`` пропускает позицию).
"""

from __future__ import annotations

import torch

IGNORE = -100


def assistant_mask(ids, header: list[int], end_id: int) -> torch.Tensor:
    """bool [T]: True у токенов ответов ассистента — всё ПОСЛЕ заголовка ``header`` до ``end_id`` включительно.

    ``ids`` — list[int] или тензор [T]; ``header`` — id строки ``<|im_start|>assistant\\n``; ``end_id`` — id ``<|im_end|>``.
    """
    # TODO 1: пройди по ids; там, где начинается подпоследовательность, равная header, ответ начинается СРАЗУ ПОСЛЕ неё.
    # TODO 2: отмечай True до первого end_id включительно (на нём модель учится останавливаться); если end_id так и
    #         не встретился — до конца строки.
    # TODO 3: продолжай поиск после end_id: в диалоге может быть несколько ответов. Верни torch.bool [T].
    raise NotImplementedError("assistant_mask")


def sft_targets(ids, mask: torch.Tensor) -> torch.Tensor:
    """Метки со сдвигом: ``targets[t] = ids[t + 1]``, если токен ``t + 1`` — часть ответа, иначе ``IGNORE``."""
    # TODO: targets — long [T], заполненный IGNORE; для t < T − 1: ids[t + 1], если mask[t + 1], иначе IGNORE.
    #       Последняя позиция — IGNORE (следующего токена нет).
    raise NotImplementedError("sft_targets")


def pack(examples: list[tuple], max_len: int) -> list[dict]:
    """Жадно уложить примеры ``(ids, targets)`` по порядку в строки длиной ≤ ``max_len``.

    Строка — dict: ``input_ids`` [L], ``targets`` [L], ``position_ids`` [L] (с нуля у каждого примера),
    ``cu_seqlens`` [n + 1] int32 — границы примеров. Пример длиннее ``max_len`` — ValueError.
    """
    # TODO 1: иди по примерам; если очередной не помещается в текущую строку — закрой её и начни новую.
    # TODO 2: строка: склеенные ids и targets (targets — как есть, без пересчёта), position_ids = arange(len) у каждого
    #         примера, cu_seqlens = [0, L0, L0 + L1, …] dtype int32.
    raise NotImplementedError("pack")


def packed_causal_mask(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """bool [L, L], True = «можно смотреть»: causal внутри своего примера, соседей по строке не видно."""
    # TODO: номер примера у каждой позиции (searchsorted по cu_seqlens или цикл); маска = tril & «один и тот же пример».
    raise NotImplementedError("packed_causal_mask")
