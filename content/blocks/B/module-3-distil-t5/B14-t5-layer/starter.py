"""Слой T5: RMSNorm без центрирования, Pre-LN, attention без 1/√d с относительным bias, FFN без bias.

Реализуй forward у T5LayerNorm, T5Attention, T5Block и T5Stack. Конструкторы и T5Model готовы: имена параметров —
как в Hugging Face, тесты загружают в модель state_dict из transformers.
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
        # Готово, не меняй.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: variance = mean(x²) по последней оси, посчитанное в fp32 (x.float()); x · rsqrt(variance + eps),
        #       обратно в dtype входа; умножить на self.weight. Среднее НЕ вычитается, bias нет.
        raise NotImplementedError("T5LayerNorm.forward")


class T5Attention(nn.Module):
    """Multi-head attention T5: q / k / v / o без bias, scores без 1/√D, плюс относительный bias (только в слое 0)."""

    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        # Готово, не меняй.
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
        # TODO 1: src = kv, если передан (cross-attention), иначе x. q из x, k и v из src; головы [B, H, L, D].
        # TODO 2: scores = q · kᵀ — БЕЗ деления на √D.
        # TODO 3: position_bias не передан → если у слоя есть таблица (has_relative_attention_bias), посчитай его
        #         через relative_position_bias (эталон B13; энкодер — двунаправленный, декодер — causal, числа из
        #         self.cfg); если таблицы нет — нули [1, H, T, S]. Передан → используй как есть.
        # TODO 4: scores + position_bias, затем маска (masked_fill(~mask, −inf)), softmax по ключам (в fp32).
        # TODO 5: веса @ v → склей головы [B, T, H·D] → self.o. Верни (выход, position_bias).
        raise NotImplementedError("T5Attention.forward")


class T5Block(nn.Module):
    """Блок T5 (Pre-LN): x + SelfAttention(LN(x)) → [x + EncDecAttention(LN(x), enc)] → x + FFN(LN(x)).

    Имена как в HF: layer[0].SelfAttention / layer_norm, у декодера layer[1].EncDecAttention / layer_norm,
    последний — layer[-1].DenseReluDense.wi / wo / layer_norm. FFN: wo(ReLU(wi(x))), без bias.
    """

    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        # Готово, не меняй.
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
        # TODO 1: self-attention: sa = self.layer[0]; x = x + sa.SelfAttention(sa.layer_norm(x), mask,
        #         position_bias=position_bias)[0]. Запомни возвращённый position_bias.
        # TODO 2: у декодера (self.is_decoder) при encoder_hidden_states: ca = self.layer[1];
        #         x = x + ca.EncDecAttention(ca.layer_norm(x), encoder_mask, kv=encoder_hidden_states)[0].
        # TODO 3: FFN: ff = self.layer[-1]; x = x + wo(ReLU(wi(ff.layer_norm(x)))). Нормы ДО подслоя (Pre-LN).
        # TODO 4: вернуть (x, position_bias).
        raise NotImplementedError("T5Block.forward")


class T5Stack(nn.Module):
    """Энкодер или декодер T5: эмбеддинги (без позиций и без масштаба) → блоки с общим bias → финальная норма."""

    def __init__(self, cfg: T5Config, embed_tokens: nn.Embedding, is_decoder: bool = False):
        super().__init__()
        # Готово, не меняй.
        self.is_decoder = is_decoder
        n = (cfg.num_decoder_layers or cfg.num_layers) if is_decoder else cfg.num_layers
        self.embed_tokens = embed_tokens
        self.block = nn.ModuleList(T5Block(cfg, i == 0, is_decoder) for i in range(n))
        self.final_layer_norm = T5LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO 1: attention_mask (1 / 0, по умолчанию единицы) → bool [B, 1, 1, T]; у декодера ещё И causal [T, T].
        #         encoder_attention_mask (если есть) → bool [B, 1, 1, S] для cross-attention.
        # TODO 2: x = embed_tokens(input_ids) — без позиций и без масштаба: позиции живут в bias.
        # TODO 3: блоки по очереди; position_bias сначала None — его посчитает блок 0, следующие получают его же.
        # TODO 4: final_layer_norm.
        raise NotImplementedError("T5Stack.forward")


class T5Model(nn.Module):
    """Энкодер + декодер с общей матрицей эмбеддингов shared. forward → выход декодера [B, T, C]."""

    def __init__(self, cfg: T5Config):
        super().__init__()
        # Готово целиком (и forward тоже): склейка энкодера и декодера.
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
