"""T5: относительный позиционный bias — бакеты расстояний и добавка к scores attention.

Реализуй relative_position_bucket и relative_position_bias.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32,
                             max_distance: int = 128) -> torch.Tensor:
    """Номер бакета для относительной позиции key − query (тензор long любой формы → такой же формы).

    Двунаправленный режим: половина бакетов — для ключей слева (и на месте), половина — для ключей справа.
    Causal: ключи справа запрещены маской, их бакет — 0, все бакеты отданы прошлому.
    Внутри половины: расстояния 0 … max_exact − 1 — каждое в своём бакете, дальше — логарифмически до
    max_distance; всё, что дальше, — в последнем бакете.
    """
    # TODO 1: двунаправленный режим — num_buckets //= 2; ключам справа (relative_position > 0) прибавь num_buckets
    #         (вторая половина); расстояние n = |relative_position|.
    #         Causal — n = max(−relative_position, 0): расстояние назад, у будущих ключей 0.
    # TODO 2: max_exact = num_buckets // 2. При n < max_exact бакет = n.
    # TODO 3: иначе max_exact + log(n / max_exact) / log(max_distance / max_exact) · (num_buckets − max_exact),
    #         округление вниз (.long()), не больше num_buckets − 1. Осторожно с log(0) — torch.where выберет
    #         ветку n, но NaN/−inf лучше не создавать (n.clamp(min=1) внутри log).
    # TODO 4: вернуть смещение половины + бакет внутри половины.
    raise NotImplementedError("relative_position_bucket")


def relative_position_bias(bias: nn.Embedding, query_length: int, key_length: int, bidirectional: bool = True,
                           num_buckets: int = 32, max_distance: int = 128, offset: int = 0) -> torch.Tensor:
    """Добавка к scores: таблица bias [num_buckets, H] → [1, H, T, S].

    offset — абсолютная позиция первого запроса (при генерации с кешем запрос один, а ключей много).
    """
    # TODO 1: позиции запросов [T, 1] (arange + offset) и ключей [1, S] на устройстве bias.weight.
    # TODO 2: относительная позиция — КЛЮЧ минус ЗАПРОС: [T, S]; бакеты — relative_position_bucket.
    # TODO 3: bias(бакеты) → [T, S, H] → [1, H, T, S] (permute + unsqueeze).
    raise NotImplementedError("relative_position_bias")
