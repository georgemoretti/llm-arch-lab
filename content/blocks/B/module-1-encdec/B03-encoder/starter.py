"""Энкодер Transformer: FFN, слой Post-LN / Pre-LN, стопка слоёв.

Реализуй FeedForward.forward, EncoderLayer.forward и Encoder.forward. Конструкторы готовы.
MultiHeadAttention, TransformerEmbedding и padding_mask — эталоны B01–B02, импортированы готовыми.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.encdec import MultiHeadAttention, TransformerEmbedding, padding_mask   # эталоны B01–B02


class FeedForward(nn.Module):
    """FFN статьи: Linear C → d_ff, ReLU, Linear d_ff → C."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Готово, не меняй.
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: linear1 → ReLU → linear2. Применяется к каждой позиции отдельно.
        raise NotImplementedError("FeedForward.forward")


class EncoderLayer(nn.Module):
    """Слой энкодера: двунаправленный self-attention и FFN; Post-LN (как в статье) или Pre-LN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_first: bool = False):
        super().__init__()
        # Готово, не меняй.
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO 1 (Post-LN, norm_first=False — как в статье): подслой, остаток, затем LayerNorm суммы:
        #         x = norm1(x + self_attn(x, mask=mask)), потом то же с ffn и norm2.
        # TODO 2 (Pre-LN, norm_first=True): LayerNorm перед подслоем, остаток — от ненормированного x.
        #         Где тогда стоит norm1 и где norm2?
        raise NotImplementedError("EncoderLayer.forward")


class Encoder(nn.Module):
    """ids источника [B, S] → память [B, S, C]: эмбеддинги + n_layers слоёв (+ финальный LayerNorm у Pre-LN)."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int,
                 pad_id: int, norm_first: bool = False):
        super().__init__()
        # Готово, не меняй.
        self.pad_id = pad_id
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, pad_id)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, norm_first) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # TODO 1: padding-маска по src (одна на все слои).
        # TODO 2: эмбеддинги, затем все слои по очереди с этой маской.
        # TODO 3: в конце self.norm (у Pre-LN это LayerNorm, у Post-LN — Identity: выход уже нормирован).
        raise NotImplementedError("Encoder.forward")
