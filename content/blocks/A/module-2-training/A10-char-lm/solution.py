"""Языковая модель на символах: словарь, датасет «окно → следующий символ», MLP-модель в духе Bengio 2003, перплексия.

Эталон.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    """Словарь символов: itos — уникальные символы текста по возрастанию, stoi — обратное отображение."""
    itos = sorted(set(text))
    return {c: i for i, c in enumerate(itos)}, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids, itos: list[str]) -> str:
    return "".join(itos[int(i)] for i in ids)


def make_dataset(ids: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Все окна длины context и символ сразу после каждого: X [N − context, context], Y [N − context]."""
    X = ids.unfold(0, context, 1)[:-1]          # окна ids[i : i + context] для i = 0 … N − context − 1
    Y = ids[context:]
    return X, Y


class MLPLM(nn.Module):
    """Эмбеддинги последних context символов → склейка → tanh-слой → логиты по словарю."""

    def __init__(self, vocab_size: int, context: int, emb_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.hidden = nn.Linear(context * emb_dim, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X [B, context] (индексы символов) → логиты [B, V]."""
        e = self.emb(X)                          # [B, context, E]
        h = torch.tanh(self.hidden(e.flatten(1)))   # склейка эмбеддингов окна: [B, context·E] → [B, H]
        return self.out(h)                       # [B, V]


@torch.no_grad()
def perplexity(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    """exp(средняя cross-entropy) на наборе, в режиме eval; режим модели восстанавливается."""
    was = model.training
    model.eval()
    ce = F.cross_entropy(model(X), Y).item()
    model.train(was)
    return math.exp(ce)


@torch.no_grad()
def generate(model: MLPLM, start: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Готовая функция: дописывает n символов, сэмплируя из softmax(логитов). start — не короче окна."""
    model.eval()
    ids = start.clone()
    for _ in range(n):
        probs = F.softmax(model(ids[-model.context:][None]), dim=-1)[0]
        ids = torch.cat([ids, torch.multinomial(probs, 1, generator=generator)])
    return ids
