"""Контрастивное обучение: InfoNCE с in-batch негативами и температурой, симметричный вариант, recall@k.

Эталон.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """InfoNCE с in-batch негативами: q [N, C], d [M, C] (M ≥ N) — эмбеддинги (обычно после L2).

    Для запроса i позитив — d[i]; все остальные документы батча (и лишние d[N:], hard negatives) — негативы.
    Лосс — cross-entropy по строкам матрицы q · dᵀ / τ с правильным классом i, среднее по запросам.
    """
    logits = q @ d.T / temperature                                      # [N, M]
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)


def symmetric_info_nce(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Среднее двух направлений: запрос → документы (строки) и документ → запросы (столбцы). M = N."""
    logits = q @ d.T / temperature
    labels = torch.arange(q.shape[0], device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def recall_at_k(scores: torch.Tensor, positives: torch.Tensor, k: int) -> float:
    """Доля запросов, у которых правильный документ среди k лучших: scores [N, M], positives [N] (номер документа)."""
    topk = scores.topk(min(k, scores.shape[1]), dim=-1).indices                      # [N, k]
    return float((topk == positives[:, None]).any(-1).float().mean())
