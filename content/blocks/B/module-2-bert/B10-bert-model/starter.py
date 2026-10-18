"""Модель BERT: слой Post-LN с GELU, стопка, pooler, MLM-голова со связанной матрицей эмбеддингов.

Реализуй forward у BertSelfAttention, BertLayer, BertModel и BertForMaskedLM. Конструкторы готовы: имена
параметров — как в Hugging Face, тесты загружают в модель state_dict из transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.bert import BertConfig, BertEmbeddings   # эталон B08
from llmlab.core.encdec import attention                  # эталон B01: softmax(q·kᵀ/√D + маска) @ v


class BertSelfAttention(nn.Module):
    """Multi-head self-attention с проекциями query / key / value (без выходной: она в BertSelfOutput)."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        # Готово, не меняй.
        self.n_heads = cfg.num_attention_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: q, k, v — проекции query / key / value; разбей на головы [B, heads, T, D]; attention(q, k, v, mask)
        #       из B01; склей головы обратно в [B, T, H]. Выходной проекции здесь нет — она в attention.output.dense.
        raise NotImplementedError("BertSelfAttention.forward")


class BertLayer(nn.Module):
    """Слой BERT: attention → dense → LN(x + ·), затем GELU-FFN → LN(h + ·). Post-LN, как в исходном Transformer."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        # Готово, не меняй.
        H, I, eps = cfg.hidden_size, cfg.intermediate_size, cfg.layer_norm_eps
        self.attention = nn.Module()
        self.attention.self = BertSelfAttention(cfg)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(H, H)
        self.attention.output.LayerNorm = nn.LayerNorm(H, eps=eps)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(H, I)
        self.output = nn.Module()
        self.output.dense = nn.Linear(I, H)
        self.output.LayerNorm = nn.LayerNorm(H, eps=eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO 1: a = attention.self(x, mask); h = attention.output.LayerNorm(x + attention.output.dense(a)).
        # TODO 2: FFN: intermediate.dense → GELU (точный, через erf: F.gelu без approximate) → output.dense.
        # TODO 3: output.LayerNorm(h + FFN(h)). Нормы стоят ПОСЛЕ сложения (Post-LN), как в исходном Transformer.
        raise NotImplementedError("BertLayer.forward")


class BertModel(nn.Module):
    """Эмбеддинги → слои → (pooler: tanh(dense(h[CLS]))). Возвращает (последовательность [B, T, H], pooled [B, H] | None)."""

    def __init__(self, cfg: BertConfig, add_pooling_layer: bool = True):
        super().__init__()
        # Готово, не меняй.
        self.cfg = cfg
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(BertLayer(cfg) for _ in range(cfg.num_hidden_layers))
        if add_pooling_layer:
            self.pooler = nn.Module()
            self.pooler.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        else:
            self.pooler = None

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None):
        # TODO 1: attention_mask (1 / 0, [B, T]; по умолчанию все единицы) → bool-маска ключей [B, 1, 1, T].
        # TODO 2: эмбеддинги (B08), затем все слои encoder.layer с этой маской.
        # TODO 3: pooled = tanh(pooler.dense(x[:, 0])) — вектор [CLS]; если pooler нет (None) — pooled = None.
        # TODO 4: вернуть (x, pooled).
        raise NotImplementedError("BertModel.forward")


class BertForMaskedLM(nn.Module):
    """BERT + MLM-голова: dense → GELU → LayerNorm → декодер (матрица = word_embeddings) + bias → логиты [B, T, V]."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        # Готово, не меняй.
        H = cfg.hidden_size
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.cls = nn.Module()
        self.cls.predictions = nn.Module()
        self.cls.predictions.transform = nn.Module()
        self.cls.predictions.transform.dense = nn.Linear(H, H)
        self.cls.predictions.transform.LayerNorm = nn.LayerNorm(H, eps=cfg.layer_norm_eps)
        self.cls.predictions.decoder = nn.Linear(H, cfg.vocab_size)
        self.cls.predictions.bias = self.cls.predictions.decoder.bias              # один bias под двумя именами (как в HF)
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight   # weight tying

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: последовательность из self.bert → transform: dense → GELU → LayerNorm → decoder (его матрица —
        #       word_embeddings, bias — свой). Вернуть логиты [B, T, V].
        raise NotImplementedError("BertForMaskedLM.forward")
