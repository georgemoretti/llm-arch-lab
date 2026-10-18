"""Головы дообучения BERT: классификация текста, разметка токенов, extractive QA.

Реализуй SequenceClassifier.forward, align_labels, qa_loss и best_span. Конструктор готов: имена — как в
Hugging Face BertForSequenceClassification. BertModel — эталон B10.
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
        # Готово, не меняй.
        self.bert = BertModel(cfg)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: pooled-выход BERT (второй элемент результата self.bert) → dropout → classifier. Логиты [B, num_labels].
        raise NotImplementedError("SequenceClassifier.forward")


def align_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    """Метки слов → метки токенов: метка слова на его первом подтокене, −100 на остальных подтокенах и спецтокенах."""
    # TODO: word_ids — номер слова для каждого токена (None у спецтокенов), как у fast-токенизаторов HF.
    #       Первый подтокен слова получает метку слова, остальные подтокены и спецтокены — −100.
    #       Как понять, что токен — первый в слове? Сравни его word_id с предыдущим.
    raise NotImplementedError("align_labels")


def qa_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor,
            end_positions: torch.Tensor) -> torch.Tensor:
    """Extractive QA: среднее двух cross-entropy — по позиции начала и по позиции конца ответа. Логиты [B, T]."""
    # TODO: логиты начала и конца [B, T] — это две «классификации по позициям». Cross-entropy для каждой, среднее.
    raise NotImplementedError("qa_loss")


def best_span(start_logits: torch.Tensor, end_logits: torch.Tensor, context_mask: torch.Tensor,
              max_answer_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Лучший ответ: argmax s_i + e_j при i ≤ j < i + max_answer_len, оба внутри контекста. [B, T] → (start [B], end [B])."""
    # TODO 1: оценки всех пар (начало i, конец j): s_i + e_j → [B, T, T] (broadcasting).
    # TODO 2: допустимы только j ≥ i, j − i < max_answer_len, и оба токена внутри контекста (context_mask [B, T]).
    # TODO 3: недопустимым −inf; argmax по сплющенным парам → (i, j) для каждого примера. Вернуть (start [B], end [B]).
    raise NotImplementedError("best_span")
