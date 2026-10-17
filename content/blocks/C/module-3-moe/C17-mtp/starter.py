"""Multi-Token Prediction (DeepSeek-V3) — модули предсказания t+2, …, t+D+1 и их loss.

Реализуй:
1. MTPModule.forward — смешать скрытое состояние предыдущей глубины с эмбеддингом следующего
   токена и прогнать через блок трансформера;
2. mtp_logits — цепочка модулей глубин 1..D с ОБЩИМИ эмбеддингом и выходной головой;
3. mtp_loss — основной loss + λ/D · Σ loss глубин, с правильным сдвигом целей.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.gemma_olmo import PreNormBlock   # блок трансформера (Pre-Norm, как Llama), эталон «Место нормализации»
from llmlab.core.llama import RMSNorm             # эталон «RMSNorm»


class MTPModule(nn.Module):
    """Один модуль Multi-Token Prediction DeepSeek-V3: смешать h предыдущей глубины и эмбеддинг
    следующего токена, прогнать через блок трансформера."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_hidden: int):
        super().__init__()
        self.enorm = RMSNorm(d_model)
        self.hnorm = RMSNorm(d_model)
        self.eh_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = PreNormBlock(d_model, n_heads, n_kv_heads, d_hidden)
        self.norm = RMSNorm(d_model)                     # норма перед общей выходной головой

    def forward(self, h_prev: torch.Tensor, emb_next: torch.Tensor) -> torch.Tensor:
        """h_prev, emb_next [B, T', C] → h [B, T', C]. Порядок конкатенации — как в vLLM: [emb; h]."""
        # TODO: нормировать оба входа (enorm — эмбеддинг, hnorm — скрытое состояние), склеить по
        #       последней оси в порядке [эмбеддинг ; h] → eh_proj (2C → C) → self.block.
        raise NotImplementedError("MTPModule.forward")


def mtp_logits(h_main: torch.Tensor, tokens: torch.Tensor, embed: nn.Embedding, lm_head: nn.Module,
               modules: nn.ModuleList) -> list[torch.Tensor]:
    """h_main [B, T, C] (выход основной модели), tokens [B, T] → логиты глубин k = 1..D, [B, T−k, V].

    Позиция i глубины k видит токены ≤ i + k и предсказывает токен i + k + 1.
    """
    # TODO: для k = 1..D: модуль k получает первые T − k позиций h предыдущей глубины (для k = 1 — h_main)
    #       и эмбеддинги токенов, сдвинутых на k: embed(tokens[:, k:]). Новый h — вход следующей глубины.
    #       Логиты глубины k = lm_head(module.norm(h)), форма [B, T − k, V].
    raise NotImplementedError("mtp_logits")


def mtp_loss(main_logits: torch.Tensor, mtp_logits_list: list[torch.Tensor], tokens: torch.Tensor, lam: float):
    """→ (L_main + λ/D · Σ L_k, L_main, [L_k]). Основная голова: i → i+1; глубина k: i → i+k+1."""
    # TODO 1: L_main — cross-entropy основной головы: позиция i предсказывает токен i + 1.
    # TODO 2: L_k — для глубины k позиция i предсказывает токен i + k + 1. Какие логиты остаются без цели?
    # TODO 3: total = L_main + λ/D · Σ_k L_k (при D = 0 — просто L_main). Вернуть (total, L_main, [L_k]).
    raise NotImplementedError("mtp_loss")
