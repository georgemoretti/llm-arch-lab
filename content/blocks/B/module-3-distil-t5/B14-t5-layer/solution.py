"""Слой T5: RMSNorm без центрирования, Pre-LN, attention без 1/√d с относительным bias, FFN без bias.

Эталон. Имена параметров — как в Hugging Face T5Model: его state_dict загружается без переименований.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.t5 import T5Config, relative_position_bias   # T5Config — размеры; relative_position_bias — эталон B13


class T5LayerNorm(nn.Module):
    """RMSNorm: x / √(mean(x²) + eps) · weight — без вычитания среднего и без bias. Статистика — в fp32."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * (x.float() * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)


class T5Attention(nn.Module):
    """Multi-head attention T5: q / k / v / o без bias, scores без 1/√D, плюс относительный bias (только в слое 0)."""

    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.n_heads, self.d_kv = cfg.num_heads, cfg.d_kv
        inner = cfg.num_heads * cfg.d_kv
        self.q = nn.Linear(cfg.d_model, inner, bias=False)
        self.k = nn.Linear(cfg.d_model, inner, bias=False)
        self.v = nn.Linear(cfg.d_model, inner, bias=False)
        self.o = nn.Linear(inner, cfg.d_model, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(cfg.relative_attention_num_buckets, cfg.num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, kv: torch.Tensor | None = None,
                position_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, T, C]; kv [B, S, C] для cross-attention (иначе kv = x) → (выход [B, T, C], bias [1, H, T, S])."""
        B, T, _ = x.shape
        src = x if kv is None else kv
        S = src.shape[1]
        split = lambda t, L: t.view(B, L, self.n_heads, self.d_kv).transpose(1, 2)   # [B, H, L, D]
        q, k, v = split(self.q(x), T), split(self.k(src), S), split(self.v(src), S)
        scores = q @ k.transpose(-2, -1)                                    # без 1/√D: масштаб — в инициализации
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = relative_position_bias(
                    self.relative_attention_bias, T, S, bidirectional=not self.is_decoder,
                    num_buckets=self.cfg.relative_attention_num_buckets,
                    max_distance=self.cfg.relative_attention_max_distance)
            else:
                position_bias = scores.new_zeros(1, self.n_heads, T, S)    # cross-attention: позиций нет
        scores = scores + position_bias
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        out = (weights @ v).transpose(1, 2).reshape(B, T, self.n_heads * self.d_kv)
        return self.o(out), position_bias


class T5Block(nn.Module):
    """Блок T5 (Pre-LN): x + SelfAttention(LN(x)) → [x + EncDecAttention(LN(x), enc)] → x + FFN(LN(x)).

    Имена как в HF: layer[0].SelfAttention / layer_norm, у декодера layer[1].EncDecAttention / layer_norm,
    последний — layer[-1].DenseReluDense.wi / wo / layer_norm. FFN: wo(ReLU(wi(x))), без bias.
    """

    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        eps = cfg.layer_norm_epsilon
        self.layer = nn.ModuleList()
        sa = nn.Module()
        sa.SelfAttention = T5Attention(cfg, has_relative_attention_bias, is_decoder)
        sa.layer_norm = T5LayerNorm(cfg.d_model, eps)
        self.layer.append(sa)
        if is_decoder:
            ca = nn.Module()
            ca.EncDecAttention = T5Attention(cfg, False, is_decoder)
            ca.layer_norm = T5LayerNorm(cfg.d_model, eps)
            self.layer.append(ca)
        ff = nn.Module()
        ff.DenseReluDense = nn.Module()
        ff.DenseReluDense.wi = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        ff.DenseReluDense.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        ff.layer_norm = T5LayerNorm(cfg.d_model, eps)
        self.layer.append(ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, position_bias: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        sa = self.layer[0]
        a, position_bias = sa.SelfAttention(sa.layer_norm(x), mask, position_bias=position_bias)
        x = x + a
        if self.is_decoder and encoder_hidden_states is not None:
            ca = self.layer[1]
            c, _ = ca.EncDecAttention(ca.layer_norm(x), encoder_mask, kv=encoder_hidden_states)
            x = x + c
        ff = self.layer[-1]
        x = x + ff.DenseReluDense.wo(F.relu(ff.DenseReluDense.wi(ff.layer_norm(x))))
        return x, position_bias


class T5Stack(nn.Module):
    """Энкодер или декодер T5: эмбеддинги (без позиций и без масштаба) → блоки с общим bias → финальная норма."""

    def __init__(self, cfg: T5Config, embed_tokens: nn.Embedding, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        n = (cfg.num_decoder_layers or cfg.num_layers) if is_decoder else cfg.num_layers
        self.embed_tokens = embed_tokens
        self.block = nn.ModuleList(T5Block(cfg, i == 0, is_decoder) for i in range(n))
        self.final_layer_norm = T5LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.bool()[:, None, None, :]                      # [B, 1, 1, T]: PAD-ключи запрещены
        if self.is_decoder:
            mask = mask & torch.tril(torch.ones(T, T, dtype=torch.bool, device=input_ids.device))
        enc_mask = None
        if encoder_attention_mask is not None:
            enc_mask = encoder_attention_mask.bool()[:, None, None, :]
        x = self.embed_tokens(input_ids)
        position_bias = None                                                # считает слой 0, остальные берут его
        for block in self.block:
            x, position_bias = block(x, mask, position_bias, encoder_hidden_states, enc_mask)
        return self.final_layer_norm(x)


class T5Model(nn.Module):
    """Энкодер + декодер с общей матрицей эмбеддингов shared. forward → выход декодера [B, T, C]."""

    def __init__(self, cfg: T5Config):
        super().__init__()
        self.cfg = cfg
        self.shared = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = T5Stack(cfg, self.shared, is_decoder=False)
        self.decoder = T5Stack(cfg, self.shared, is_decoder=True)

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                decoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        memory = self.encoder(input_ids, attention_mask)
        return self.decoder(decoder_input_ids, decoder_attention_mask, memory, attention_mask)
