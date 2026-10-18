"""Дистилляция: мягкие метки с температурой, смесь с hard loss, cosine-loss скрытых состояний, ученик DistilBERT.

Эталон. Лоссы — как в Hinton et al. (2015) и в скрипте обучения DistilBERT (KLDivLoss(batchmean) · T²,
CosineEmbeddingLoss по настоящим токенам).
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from llmlab.core.bert import BertForMaskedLM   # эталон B10: BERT с MLM-головой


def soft_targets(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Мягкие метки: softmax(logits / T) по последней оси. T > 1 — распределение площе."""
    return F.softmax(logits / temperature, dim=-1)


def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """T² · KL(p_учителя ‖ p_ученика) при температуре T; среднее по строкам [..., V]. Учитель не получает градиент."""
    V = student_logits.shape[-1]
    log_q = F.log_softmax(student_logits.reshape(-1, V) / temperature, dim=-1)
    log_p = F.log_softmax(teacher_logits.detach().reshape(-1, V) / temperature, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()          # = F.kl_div(log_q, log_p, log_target=True, "batchmean")
    return kl * temperature**2                                     # T²: градиент мягкой части не тает с ростом T


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
                      temperature: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """alpha · T²·KL по всем строкам + (1 − alpha) · CE(ученик, labels) по строкам с labels ≠ −100."""
    V = student_logits.shape[-1]
    soft = distill_kl(student_logits, teacher_logits, temperature)
    s, y = student_logits.reshape(-1, V), labels.reshape(-1)
    hard = F.cross_entropy(s, y, ignore_index=-100) if bool((y != -100).any()) else s.sum() * 0.0
    return alpha * soft + (1 - alpha) * hard


def hidden_cosine_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
    """Среднее (1 − cos) между скрытыми состояниями ученика и учителя [B, T, H] по настоящим токенам (mask = 1)."""
    m = attention_mask.bool()
    cos = F.cosine_similarity(student_hidden[m], teacher_hidden.detach()[m], dim=-1)
    return (1 - cos).mean()


def student_from_teacher(teacher: BertForMaskedLM, layers: list[int] | None = None) -> BertForMaskedLM:
    """Ученик DistilBERT: те же эмбеддинги и голова, слои — копии слоёв учителя с номерами layers (по умолчанию 0, 2, 4, …)."""
    n = len(teacher.bert.encoder.layer)
    layers = list(range(0, n, 2)) if layers is None else list(layers)
    student = BertForMaskedLM(replace(teacher.bert.cfg, num_hidden_layers=len(layers)))
    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())
    for i, j in enumerate(layers):
        student.bert.encoder.layer[i].load_state_dict(teacher.bert.encoder.layer[j].state_dict())
    student.cls.load_state_dict(teacher.cls.state_dict())        # decoder.weight связан с эмбеддингами ученика
    return student
