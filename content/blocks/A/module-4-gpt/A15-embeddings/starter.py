"""Эмбеддинги токенов и обучаемых позиций, weight tying, языковая модель без attention.

Реализуй TokenPositionEmbedding.forward, tie_weights, count_params и EmbeddingLM.forward.
Конструкторы готовы: имена слоёв — как в llmlab/core/gpt2.py (тесты загружают веса эталона).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPositionEmbedding(nn.Module):
    """Эмбеддинг токена + обучаемый эмбеддинг позиции: [B, T] → [B, T, C]."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        # Готово, не меняй.
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)   # токены
        self.wpe = nn.Embedding(block_size, n_embd)   # позиции 0 … block_size − 1
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [B, T] (long) → [B, T, C]; T > block_size → ValueError."""
        # TODO 1: T > block_size → ValueError (для таких позиций нет строки в wpe).
        # TODO 2: номера позиций 0 … T − 1 (на том же устройстве, что idx) → wpe → [T, C].
        # TODO 3: wte(idx) [B, T, C] + позиции [T, C]. Нужно ли копировать позиции для каждого примера батча?
        raise NotImplementedError("TokenPositionEmbedding.forward")


def tie_weights(lm_head: nn.Linear, wte: nn.Embedding) -> None:
    """Связать выходной слой с эмбеддингами: один и тот же тензор весов [V, C] в обоих модулях."""
    # TODO: у nn.Linear(C, V) вес имеет форму [V, C] — ровно как у nn.Embedding(V, C).
    #       Нужен не копия значений, а ОДИН объект-параметр в двух модулях. Чем отличается
    #       lm_head.weight.data.copy_(wte.weight) от присваивания самого параметра?
    raise NotImplementedError("tie_weights")


def count_params(model: nn.Module) -> int:
    """Число обучаемых чисел модели; связанные веса считаются один раз."""
    # TODO: сколько раз связанный вес встречается в model.state_dict()? А в model.parameters()?
    raise NotImplementedError("count_params")


class EmbeddingLM(nn.Module):
    """Языковая модель без attention: (токен + позиция) → выходной слой, связанный с эмбеддингами."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        # Готово, не меняй.
        self.emb = TokenPositionEmbedding(vocab_size, block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        tie_weights(self.lm_head, self.emb.wte)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx [B, T] → (logits [B, T, V], loss — средняя cross-entropy по всем B·T позициям или None)."""
        # TODO 1: эмбеддинги → lm_head → логиты [B, T, V]. Softmax не нужен.
        # TODO 2: если targets [B, T] переданы — F.cross_entropy ждёт логиты [N, V] и цели [N]:
        #         как превратить [B, T, V] и [B, T] в такие формы? Иначе loss = None.
        raise NotImplementedError("EmbeddingLM.forward")
