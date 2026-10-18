"""Sentence-эмбеддинги: mean / CLS pooling, L2-нормировка, косинус; bi-encoder и cross-encoder.

Реализуй четыре функции и forward у BiEncoder и CrossEncoder. Энкодер — любой из курса: ModernBertModel (B16) или
BertModel (B10); _hidden уже умеет достать из них скрытые состояния [B, T, C].
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Среднее по настоящим токенам: [B, T, C], [B, T] → [B, C]. PAD в среднее не входит."""
    # TODO: маску → [B, T, 1] в dtype hidden; сумма hidden · маска по T, делённая на ЧИСЛО НАСТОЯЩИХ токенов
    #       (сумма маски, clamp(min=1e-9) на случай пустого текста). Делить на T нельзя: PAD попадёт в среднее.
    raise NotImplementedError("mean_pool")


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Вектор первого токена ([CLS]): [B, T, C] → [B, C]."""
    # TODO: скрытое состояние позиции 0.
    raise NotImplementedError("cls_pool")


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """x / ‖x‖ по последней оси; нулевой вектор остаётся нулевым."""
    # TODO: норма по последней оси (keepdim), clamp(min=eps), деление. F.normalize не используй — напиши руками.
    raise NotImplementedError("l2_normalize")


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Косинусы всех пар: [N, C], [M, C] → [N, M]."""
    # TODO: нормировать оба набора и перемножить матрицы (без цикла по парам).
    raise NotImplementedError("cosine_matrix")


def _hidden(encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Скрытые состояния [B, T, C] любого энкодера курса: ModernBertModel возвращает тензор, BertModel — (h, pooled)."""
    kw = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
    out = encoder(input_ids, attention_mask=attention_mask, **kw)
    return out[0] if isinstance(out, tuple) else out


class BiEncoder(nn.Module):
    """Bi-encoder: каждый текст кодируется отдельно в один вектор; близость — косинус (скалярное после L2)."""

    def __init__(self, encoder: nn.Module, pooling: str = "mean", normalize: bool = True):
        super().__init__()
        # Готово, не меняй.
        assert pooling in ("mean", "cls"), pooling
        self.encoder, self.pooling, self.normalize = encoder, pooling, normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # TODO: h = _hidden(...); pooling "mean" → mean_pool, "cls" → cls_pool; при self.normalize — l2_normalize.
        raise NotImplementedError("BiEncoder.forward")


class CrossEncoder(nn.Module):
    """Cross-encoder: запрос и документ — один вход [CLS] q [SEP] d [SEP]; оценка — Linear по вектору [CLS]."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        # Готово, не меняй.
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: h = _hidden(..., token_type_ids); вектор [CLS] → self.classifier → одна оценка на пару, форма [B].
        raise NotImplementedError("CrossEncoder.forward")
