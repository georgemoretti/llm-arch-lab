"""Контрастивное обучение: InfoNCE с in-batch негативами и температурой, симметричный вариант, recall@k.

Реализуй три функции. Эмбеддинги приходят уже L2-нормированными (BiEncoder из B18), так что q · dᵀ — косинусы.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """InfoNCE с in-batch негативами: q [N, C], d [M, C] (M ≥ N) — эмбеддинги (обычно после L2).

    Для запроса i позитив — d[i]; все остальные документы батча (и лишние d[N:], hard negatives) — негативы.
    Лосс — cross-entropy по строкам матрицы q · dᵀ / τ с правильным классом i, среднее по запросам.
    """
    # TODO 1: logits = q · dᵀ, ДЕЛЁННЫЕ на temperature → [N, M].
    # TODO 2: правильный «класс» запроса i — документ i: labels = arange(N).
    # TODO 3: cross-entropy по строкам (каждый запрос выбирает свой документ среди M), среднее по N.
    raise NotImplementedError("info_nce")


def symmetric_info_nce(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Среднее двух направлений: запрос → документы (строки) и документ → запросы (столбцы). M = N."""
    # TODO: те же logits; CE по строкам (запрос ищет документ) и CE по столбцам (logits.T: документ ищет запрос),
    #       среднее двух.
    raise NotImplementedError("symmetric_info_nce")


def recall_at_k(scores: torch.Tensor, positives: torch.Tensor, k: int) -> float:
    """Доля запросов, у которых правильный документ среди k лучших: scores [N, M], positives [N] (номер документа)."""
    # TODO: индексы k лучших документов каждого запроса (topk по последней оси; k не больше M); запрос «попал», если
    #       positives[i] среди них. Вернуть долю попаданий как float.
    raise NotImplementedError("recall_at_k")
