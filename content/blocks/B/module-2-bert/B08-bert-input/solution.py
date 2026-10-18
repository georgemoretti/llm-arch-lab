"""Вход BERT: [CLS] A [SEP] B [SEP], обрезка пары, token_type_ids, attention_mask, эмбеддинги токена + позиции + сегмента.

Эталон. Имена слоёв — как в Hugging Face BertEmbeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def truncate_pair(a: list[int], b: list[int], max_tokens: int) -> tuple[list[int], list[int]]:
    """longest_first (как в быстрых токенизаторах Hugging Face): короткий сегмент не трогаем, если он не длиннее
    половины бюджета; иначе бюджет делится пополам, и лишний токен нечётного бюджета достаётся более длинному
    (при равных длинах длинным считается b)."""
    if len(a) + len(b) <= max_tokens:
        return list(a), list(b)
    a_longer = len(a) > len(b)
    n_long, n_short = (len(a), len(b)) if a_longer else (len(b), len(a))
    half = max_tokens // 2
    if n_short > half:
        n_short, n_long = half, max_tokens - half
    else:
        n_long = max_tokens - n_short
    n_a, n_b = (n_long, n_short) if a_longer else (n_short, n_long)
    return list(a[:n_a]), list(b[:n_b])


def build_inputs(a: list[int], b: list[int] | None, cls_id: int, sep_id: int,
                 max_len: int) -> tuple[list[int], list[int]]:
    """[CLS] a [SEP] (b [SEP]) с обрезкой до max_len и token_type_ids: 0 — первый сегмент, 1 — второй."""
    if b is None:
        a = a[: max_len - 2]
        return [cls_id] + a + [sep_id], [0] * (len(a) + 2)
    a, b = truncate_pair(a, b, max_len - 3)
    ids = [cls_id] + a + [sep_id] + b + [sep_id]
    return ids, [0] * (len(a) + 2) + [1] * (len(b) + 1)


def pad_batch(examples: list[tuple[list[int], list[int]]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """[(input_ids, token_type_ids), …] → input_ids, token_type_ids, attention_mask [B, T_max] (PAD справа, long)."""
    T = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    types = torch.zeros(len(examples), T, dtype=torch.long)
    mask = torch.zeros(len(examples), T, dtype=torch.long)
    for i, (x, t) in enumerate(examples):
        ids[i, : len(x)] = torch.tensor(x)
        types[i, : len(t)] = torch.tensor(t)
        mask[i, : len(x)] = 1
    return {"input_ids": ids, "token_type_ids": types, "attention_mask": mask}


@dataclass
class BertConfig:
    vocab_size: int
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 64
    max_position_embeddings: int = 32
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    hidden_dropout_prob: float = 0.0


class BertEmbeddings(nn.Module):
    """Сумма трёх эмбеддингов — токен, позиция, сегмент — и LayerNorm: [B, T] → [B, T, H]."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        T = input_ids.shape[1]
        positions = torch.arange(T, device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.word_embeddings(input_ids) + self.position_embeddings(positions) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(x))
