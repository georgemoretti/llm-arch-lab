"""Языковая модель на символах: словарь, датасет «окно → следующий символ», MLP-модель в духе Bengio 2003, перплексия.

Реализуй build_vocab, encode, decode, make_dataset, MLPLM.forward и perplexity.
Конструктор MLPLM и generate уже готовы.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    """Словарь символов: itos — уникальные символы текста по возрастанию, stoi — обратное отображение."""
    # TODO: уникальные символы, отсортированные (тогда словарь не зависит от порядка в тексте).
    raise NotImplementedError("build_vocab")


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    # TODO: строка → тензор индексов dtype=torch.long (nn.Embedding и cross_entropy ждут long).
    raise NotImplementedError("encode")


def decode(ids, itos: list[str]) -> str:
    # TODO: индексы (тензор или список) → строка.
    raise NotImplementedError("decode")


def make_dataset(ids: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Все окна длины context и символ сразу после каждого: X [N − context, context], Y [N − context]."""
    # TODO: X[i] = ids[i : i + context], Y[i] = ids[i + context]. Можно циклом + torch.stack,
    #       можно одной операцией ids.unfold(0, context, 1) — сколько окон она даст и какое лишнее?
    raise NotImplementedError("make_dataset")


class MLPLM(nn.Module):
    """Эмбеддинги последних context символов → склейка → tanh-слой → логиты по словарю."""

    def __init__(self, vocab_size: int, context: int, emb_dim: int = 16, hidden: int = 128):
        super().__init__()
        # Готово, не меняй: имена слоёв проверяют тесты (они загружают веса эталона).
        self.context = context
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.hidden = nn.Linear(context * emb_dim, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X [B, context] (индексы символов) → логиты [B, V]."""
        # TODO 1: эмбеддинги окна: [B, context] → [B, context, E].
        # TODO 2: склей эмбеддинги окна в один вектор [B, context·E] (порядок: сначала первый символ окна).
        # TODO 3: tanh(hidden(...)) → out(...). Softmax НЕ нужен: cross_entropy принимает логиты.
        raise NotImplementedError("MLPLM.forward")


@torch.no_grad()
def perplexity(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    """exp(средняя cross-entropy) на наборе, в режиме eval; режим модели восстанавливается."""
    # TODO: как evaluate из прошлой задачи, но вернуть exp от средней cross-entropy.
    raise NotImplementedError("perplexity")


@torch.no_grad()
def generate(model: MLPLM, start: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Готовая функция: дописывает n символов, сэмплируя из softmax(логитов). start — не короче окна."""
    model.eval()
    ids = start.clone()
    for _ in range(n):
        probs = F.softmax(model(ids[-model.context:][None]), dim=-1)[0]
        ids = torch.cat([ids, torch.multinomial(probs, 1, generator=generator)])
    return ids
