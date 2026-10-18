"""T5: относительный позиционный bias — бакеты расстояний и добавка к scores attention.

Эталон. Совпадает с transformers T5Attention._relative_position_bucket и T5Attention.compute_bias.
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
    buckets = torch.zeros_like(relative_position)
    if bidirectional:
        num_buckets //= 2
        buckets = buckets + (relative_position > 0).long() * num_buckets     # ключ справа → вторая половина
        n = relative_position.abs()
    else:
        n = (-relative_position).clamp(min=0)                                # расстояние назад; будущее → 0
    max_exact = num_buckets // 2
    is_small = n < max_exact
    log_ratio = torch.log(n.clamp(min=1).float() / max_exact) / math.log(max_distance / max_exact)
    large = max_exact + (log_ratio * (num_buckets - max_exact)).long()
    large = large.clamp(max=num_buckets - 1)
    return buckets + torch.where(is_small, n, large)


def relative_position_bias(bias: nn.Embedding, query_length: int, key_length: int, bidirectional: bool = True,
                           num_buckets: int = 32, max_distance: int = 128, offset: int = 0) -> torch.Tensor:
    """Добавка к scores: таблица bias [num_buckets, H] → [1, H, T, S].

    offset — абсолютная позиция первого запроса (при генерации с кешем запрос один, а ключей много).
    """
    device = bias.weight.device
    q_pos = torch.arange(query_length, device=device)[:, None] + offset    # [T, 1]
    k_pos = torch.arange(key_length, device=device)[None, :]               # [1, S]
    buckets = relative_position_bucket(k_pos - q_pos, bidirectional, num_buckets, max_distance)   # [T, S]
    return bias(buckets).permute(2, 0, 1).unsqueeze(0)                     # [T, S, H] → [1, H, T, S]
