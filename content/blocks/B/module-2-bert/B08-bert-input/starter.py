"""Вход BERT: [CLS] A [SEP] B [SEP], обрезка пары, token_type_ids, attention_mask, эмбеддинги токена + позиции + сегмента.

Реализуй truncate_pair, build_inputs, pad_batch и BertEmbeddings.forward. Конфиг и конструктор готовы.
Имена слоёв — как в Hugging Face BertEmbeddings: тесты загружают в модуль веса из transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def truncate_pair(a: list[int], b: list[int], max_tokens: int) -> tuple[list[int], list[int]]:
    """longest_first (как в быстрых токенизаторах Hugging Face): короткий сегмент не трогаем, если он не длиннее
    половины бюджета; иначе бюджет делится пополам, и лишний токен нечётного бюджета достаётся более длинному
    (при равных длинах длинным считается b)."""
    # TODO 1: влезает — вернуть копии без изменений.
    # TODO 2: иначе выясни, какой сегмент длиннее (при равенстве длинным считается b).
    # TODO 3: если короткий не длиннее половины бюджета (max_tokens // 2) — он остаётся целым, длинный получает остаток.
    #         Иначе короткому — половина, длинному — max_tokens − половина (лишний токен нечётного бюджета — длинному).
    # TODO 4: обрезай с конца: a[:n_a], b[:n_b].
    raise NotImplementedError("truncate_pair")


def build_inputs(a: list[int], b: list[int] | None, cls_id: int, sep_id: int,
                 max_len: int) -> tuple[list[int], list[int]]:
    """[CLS] a [SEP] (b [SEP]) с обрезкой до max_len и token_type_ids: 0 — первый сегмент, 1 — второй."""
    # TODO 1: один сегмент: [CLS] a [SEP], a обрезан так, чтобы всё влезло в max_len; типы — нули.
    # TODO 2: пара: обрезка truncate_pair до max_len − 3 (три служебных токена), затем [CLS] a [SEP] b [SEP].
    #         token_type_ids: 0 у [CLS], a и первого [SEP]; 1 у b и последнего [SEP].
    raise NotImplementedError("build_inputs")


def pad_batch(examples: list[tuple[list[int], list[int]]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """[(input_ids, token_type_ids), …] → input_ids, token_type_ids, attention_mask [B, T_max] (PAD справа, long)."""
    # TODO: тензоры [B, T_max] типа long: input_ids (PAD справа), token_type_ids (нули на PAD),
    #       attention_mask (1 у настоящих токенов, 0 у PAD). Вернуть dict с этими тремя ключами.
    raise NotImplementedError("pad_batch")


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
        # Готово, не меняй: имена — как в Hugging Face.
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        # TODO 1: позиции 0 … T − 1; token_type_ids по умолчанию — нули той же формы, что input_ids.
        # TODO 2: сумма трёх эмбеддингов → LayerNorm → dropout.
        raise NotImplementedError("BertEmbeddings.forward")
