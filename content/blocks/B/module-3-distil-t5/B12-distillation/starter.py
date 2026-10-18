"""Дистилляция: мягкие метки с температурой, смесь с hard loss, cosine-loss скрытых состояний, ученик DistilBERT.

Реализуй пять функций. Учитель и ученик — BertForMaskedLM из B10 (готовый эталон).
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from llmlab.core.bert import BertForMaskedLM   # эталон B10: BERT с MLM-головой


def soft_targets(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Мягкие метки: softmax(logits / T) по последней оси. T > 1 — распределение площе."""
    # TODO: поделить логиты на температуру и взять softmax по последней оси.
    raise NotImplementedError("soft_targets")


def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """T² · KL(p_учителя ‖ p_ученика) при температуре T; среднее по строкам [..., V]. Учитель не получает градиент."""
    # TODO 1: логиты [..., V] → строки [N, V] (reshape). Логиты учителя — detach(): учитель заморожен.
    # TODO 2: log q = log_softmax(ученик / T), log p = log_softmax(учитель / T) — оба с температурой.
    # TODO 3: KL(p ‖ q) = Σ_v p · (log p − log q) по классам, затем среднее по строкам (это reduction="batchmean").
    # TODO 4: умножить на T² — иначе с ростом T градиент мягкой части уменьшается как 1/T².
    raise NotImplementedError("distill_kl")


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
                      temperature: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """alpha · T²·KL по всем строкам + (1 − alpha) · CE(ученик, labels) по строкам с labels ≠ −100."""
    # TODO 1: soft = distill_kl(…) — по ВСЕМ позициям: учитель даёт мягкие метки везде.
    # TODO 2: hard = cross_entropy(ученик, labels, ignore_index=−100) — только где метка есть (у MLM — закрытые токены).
    #         Если меток в батче нет совсем, hard = 0 (а не nan).
    # TODO 3: alpha · soft + (1 − alpha) · hard.
    raise NotImplementedError("distillation_loss")


def hidden_cosine_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
    """Среднее (1 − cos) между скрытыми состояниями ученика и учителя [B, T, H] по настоящим токенам (mask = 1)."""
    # TODO: выбери настоящие токены (attention_mask.bool()) у обоих тензоров → [N, H]; F.cosine_similarity по оси H;
    #       вернуть mean(1 − cos). Состояния учителя — detach().
    raise NotImplementedError("hidden_cosine_loss")


def student_from_teacher(teacher: BertForMaskedLM, layers: list[int] | None = None) -> BertForMaskedLM:
    """Ученик DistilBERT: те же эмбеддинги и голова, слои — копии слоёв учителя с номерами layers (по умолчанию 0, 2, 4, …)."""
    # TODO 1: layers по умолчанию — каждый второй слой учителя, начиная с 0.
    # TODO 2: новый BertForMaskedLM с конфигом учителя, где num_hidden_layers = len(layers)
    #         (teacher.bert.cfg — dataclass, поможет dataclasses.replace).
    # TODO 3: скопируй веса (load_state_dict(… .state_dict())): эмбеддинги, выбранные слои по порядку, голову cls.
    #         Не присваивай модули учителя напрямую: тогда обучение ученика изменит учителя.
    raise NotImplementedError("student_from_teacher")
