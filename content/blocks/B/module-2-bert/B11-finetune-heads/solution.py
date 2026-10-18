"""Головы дообучения BERT: классификация текста, разметка токенов, extractive QA.

Эталон. Имена — как в Hugging Face BertForSequenceClassification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.bert import BertConfig, BertModel   # эталоны B08, B10


class SequenceClassifier(nn.Module):
    """Классификация текста: pooled-выход BERT → dropout → Linear(H, n_labels). Имена как в BertForSequenceClassification."""

    def __init__(self, cfg: BertConfig, num_labels: int):
        super().__init__()
        self.bert = BertModel(cfg)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(self.dropout(pooled))


def align_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    """Метки слов → метки токенов: метка слова на его первом подтокене, −100 на остальных подтокенах и спецтокенах."""
    labels, prev = [], None
    for w in word_ids:
        labels.append(-100 if w is None or w == prev else word_labels[w])
        prev = w
    return labels


def qa_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor,
            end_positions: torch.Tensor) -> torch.Tensor:
    """Extractive QA: среднее двух cross-entropy — по позиции начала и по позиции конца ответа. Логиты [B, T]."""
    return (F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2


def best_span(start_logits: torch.Tensor, end_logits: torch.Tensor, context_mask: torch.Tensor,
              max_answer_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Лучший ответ: argmax s_i + e_j при i ≤ j < i + max_answer_len, оба внутри контекста. [B, T] → (start [B], end [B])."""
    B, T = start_logits.shape
    scores = start_logits[:, :, None] + end_logits[:, None, :]                       # [B, T(start), T(end)]
    i = torch.arange(T, device=start_logits.device)
    allowed = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < max_answer_len)   # j ≥ i и длина ≤ L
    allowed = allowed & context_mask[:, :, None] & context_mask[:, None, :]
    flat = scores.masked_fill(~allowed, float("-inf")).view(B, -1).argmax(-1)
    return flat // T, flat % T
