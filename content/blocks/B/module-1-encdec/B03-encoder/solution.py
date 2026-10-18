"""Энкодер Transformer: FFN, слой Post-LN / Pre-LN, стопка слоёв.

Эталон. Имена слоёв — как в nn.TransformerEncoderLayer (self_attn, linear1, linear2, norm1, norm2).
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
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """Слой энкодера: двунаправленный self-attention и FFN; Post-LN (как в статье) или Pre-LN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_first: bool = False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), mask=mask)
            return x + self.ffn(self.norm2(x))
        x = self.norm1(x + self.self_attn(x, mask=mask))
        return self.norm2(x + self.ffn(x))


class Encoder(nn.Module):
    """ids источника [B, S] → память [B, S, C]: эмбеддинги + n_layers слоёв (+ финальный LayerNorm у Pre-LN)."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int,
                 pad_id: int, norm_first: bool = False):
        super().__init__()
        self.pad_id = pad_id
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, pad_id)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, norm_first) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask = padding_mask(src, self.pad_id)        # [B, 1, 1, S]
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
