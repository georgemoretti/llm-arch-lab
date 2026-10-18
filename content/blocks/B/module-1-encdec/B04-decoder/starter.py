"""Декодер Transformer: маска цели, cross-attention к памяти энкодера, стопка слоёв.

Реализуй causal_mask, target_mask, DecoderLayer.forward и Decoder.forward. Конструкторы готовы.
FeedForward, MultiHeadAttention, TransformerEmbedding и padding_mask — эталоны B01–B03.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmlab.core.encdec import FeedForward, MultiHeadAttention, TransformerEmbedding, padding_mask   # B01–B03


def causal_mask(T: int) -> torch.Tensor:
    """bool [T, T]: True на диагонали и ниже."""
    # TODO: как в A16.
    raise NotImplementedError("causal_mask")


def target_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Маска self-attention декодера: causal И не PAD-ключ. [B, T] → bool [B, 1, T, T]."""
    # TODO: padding_mask цели [B, 1, 1, T] и causal [T, T] объединяются логическим И — broadcasting даст
    #       [B, 1, T, T]. Не забудь положить causal-маску на то же устройство, что и tgt.
    raise NotImplementedError("target_mask")


class DecoderLayer(nn.Module):
    """Слой декодера: masked self-attention → cross-attention к памяти энкодера → FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_first: bool = False):
        super().__init__()
        # Готово, не меняй.
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO 1: три подслоя, у каждого остаток и своя норма (norm1, norm2, norm3):
        #         self_attn по x с tgt_mask → cross_attn → ffn.
        # TODO 2: в cross_attn запросы — из x (декодер), ключи и значения — из memory (энкодер).
        #         Какая маска нужна здесь: что декодеру запрещено видеть в источнике?
        # TODO 3: режим norm_first=True (Pre-LN), как в B03.
        raise NotImplementedError("DecoderLayer.forward")


class Decoder(nn.Module):
    """ids цели [B, T] + память [B, S, C] → скрытые состояния [B, T, C]."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int,
                 pad_id: int, norm_first: bool = False):
        super().__init__()
        # Готово, не меняй.
        self.pad_id = pad_id
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, pad_id)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, norm_first) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: маска цели по tgt, эмбеддинги, все слои (память и src_mask — одни на все слои), self.norm.
        raise NotImplementedError("Decoder.forward")
