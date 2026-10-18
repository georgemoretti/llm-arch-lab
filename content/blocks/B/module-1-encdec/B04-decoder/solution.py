"""Декодер Transformer: маска цели, cross-attention к памяти энкодера, стопка слоёв.

Эталон. Имена слоёв — как в nn.TransformerDecoderLayer (self_attn, multihead_attn → cross_attn, linear1,
linear2, norm1–norm3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmlab.core.encdec import FeedForward, MultiHeadAttention, TransformerEmbedding, padding_mask   # B01–B03


def causal_mask(T: int) -> torch.Tensor:
    """bool [T, T]: True на диагонали и ниже."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def target_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Маска self-attention декодера: causal И не PAD-ключ. [B, T] → bool [B, 1, T, T]."""
    return padding_mask(tgt, pad_id) & causal_mask(tgt.shape[1]).to(tgt.device)


class DecoderLayer(nn.Module):
    """Слой декодера: masked self-attention → cross-attention к памяти энкодера → FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_first: bool = False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
            x = x + self.cross_attn(self.norm2(x), memory, mask=src_mask)
            return x + self.ffn(self.norm3(x))
        x = self.norm1(x + self.self_attn(x, mask=tgt_mask))
        x = self.norm2(x + self.cross_attn(x, memory, mask=src_mask))   # Q из декодера, K и V из памяти
        return self.norm3(x + self.ffn(x))


class Decoder(nn.Module):
    """ids цели [B, T] + память [B, S, C] → скрытые состояния [B, T, C]."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int,
                 pad_id: int, norm_first: bool = False):
        super().__init__()
        self.pad_id = pad_id
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, pad_id)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, norm_first) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        tgt_mask = target_mask(tgt, self.pad_id)     # [B, 1, T, T]
        x = self.embed(tgt)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)
