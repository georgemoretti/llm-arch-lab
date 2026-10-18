"""Sentence-эмбеддинги: mean / CLS pooling, L2-нормировка, косинус; bi-encoder и cross-encoder.

Код для отладки: в нём одна ошибка. Энкодер — любой из курса: ModernBertModel (B16) или BertModel (B10).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Среднее по настоящим токенам: [B, T, C], [B, T] → [B, C]. PAD в среднее не входит."""
    return hidden.mean(1)


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Вектор первого токена ([CLS]): [B, T, C] → [B, C]."""
    return hidden[:, 0]


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """x / ‖x‖ по последней оси; нулевой вектор остаётся нулевым."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Косинусы всех пар: [N, C], [M, C] → [N, M]."""
    return l2_normalize(a) @ l2_normalize(b).T


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
        assert pooling in ("mean", "cls"), pooling
        self.encoder, self.pooling, self.normalize = encoder, pooling, normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = _hidden(self.encoder, input_ids, attention_mask)
        e = mean_pool(h, attention_mask) if self.pooling == "mean" else cls_pool(h)
        return l2_normalize(e) if self.normalize else e


class CrossEncoder(nn.Module):
    """Cross-encoder: запрос и документ — один вход [CLS] q [SEP] d [SEP]; оценка — Linear по вектору [CLS]."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        h = _hidden(self.encoder, input_ids, attention_mask, token_type_ids)
        return self.classifier(cls_pool(h)).squeeze(-1)
