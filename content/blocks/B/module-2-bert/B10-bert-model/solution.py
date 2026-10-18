"""Модель BERT: слой Post-LN с GELU, стопка, pooler, MLM-голова со связанной матрицей эмбеддингов.

Эталон. Имена параметров — как в Hugging Face BertModel / BertForMaskedLM: их state_dict загружается без
переименований.
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
        self.n_heads = cfg.num_attention_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, H = x.shape
        split = lambda t: t.view(B, T, self.n_heads, H // self.n_heads).transpose(1, 2)   # [B, heads, T, D]
        out, _ = attention(split(self.query(x)), split(self.key(x)), split(self.value(x)), mask)
        return out.transpose(1, 2).reshape(B, T, H)


class BertLayer(nn.Module):
    """Слой BERT: attention → dense → LN(x + ·), затем GELU-FFN → LN(h + ·). Post-LN, как в исходном Transformer."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
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
        a = self.attention.self(x, mask)
        h = self.attention.output.LayerNorm(x + self.attention.output.dense(a))
        f = self.output.dense(F.gelu(self.intermediate.dense(h)))       # GELU точный (через erf), как в BERT
        return self.output.LayerNorm(h + f)


class BertModel(nn.Module):
    """Эмбеддинги → слои → (pooler: tanh(dense(h[CLS]))). Возвращает (последовательность [B, T, H], pooled [B, H] | None)."""

    def __init__(self, cfg: BertConfig, add_pooling_layer: bool = True):
        super().__init__()
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
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.bool()[:, None, None, :]                   # [B, 1, 1, T]: запрет PAD-ключей
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.encoder.layer:
            x = layer(x, mask)
        pooled = torch.tanh(self.pooler.dense(x[:, 0])) if self.pooler is not None else None
        return x, pooled


class BertForMaskedLM(nn.Module):
    """BERT + MLM-голова: dense → GELU → LayerNorm → декодер (матрица = word_embeddings) + bias → логиты [B, T, V]."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
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
        x, _ = self.bert(input_ids, token_type_ids, attention_mask)
        t = self.cls.predictions.transform
        h = t.LayerNorm(F.gelu(t.dense(x)))
        return self.cls.predictions.decoder(h)
