"""Seq2seq Transformer целиком: shift right, label smoothing, общая матрица эмбеддингов, жадная генерация.

Реализуй shift_right, label_smoothed_ce, Seq2Seq.encode / decode / forward и greedy_decode.
Конфиг и конструктор Seq2Seq готовы. Encoder, Decoder и padding_mask — эталоны B01–B04.
F.cross_entropy и nn.CrossEntropyLoss в этой задаче запрещены: тесты используют их как эталон.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.encdec import Decoder, Encoder, padding_mask   # эталоны B01–B04


def shift_right(tgt: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Вход декодера при teacher forcing: [y0, y1, …, y_{T−1}] → [BOS, y0, …, y_{T−2}]."""
    # TODO: столбец BOS [B, 1] слева, последний токен цели отбросить. Длина не меняется. Вход не портить.
    raise NotImplementedError("shift_right")


def label_smoothed_ce(logits: torch.Tensor, target: torch.Tensor, eps: float = 0.1,
                      ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy с label smoothing: цель (1 − ε)·one-hot + ε/V; среднее по позициям, где target ≠ ignore_index."""
    # TODO 1: log-вероятности (log_softmax по словарю; для устойчивости — во float32).
    # TODO 2: −log p(y) в правильном классе (gather). Осторожно: ignore_index может быть отрицательным.
    # TODO 3: «равномерная» часть: −(1/V)·Σ_v log p(v).
    # TODO 4: (1 − ε)·первое + ε·второе; усреднить только по позициям, где target ≠ ignore_index.
    raise NotImplementedError("label_smoothed_ce")


@dataclass
class Seq2SeqConfig:
    vocab_size: int
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    n_layers: int = 2
    max_len: int = 32
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    norm_first: bool = False


class Seq2Seq(nn.Module):
    """Энкодер-декодер с общим словарём: один эмбеддинг на энкодер, декодер и выходной слой (§3.4 статьи)."""

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        # Готово, не меняй: имена и общая матрица проверяются тестами.
        self.cfg = cfg
        args = (cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.max_len, cfg.pad_id,
                cfg.norm_first)
        self.encoder = Encoder(*args)
        self.decoder = Decoder(*args)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.decoder.embed.tok.weight = self.encoder.embed.tok.weight
        self.lm_head.weight = self.encoder.embed.tok.weight

    def encode(self, src: torch.Tensor):
        """[B, S] → (память [B, S, C], src_mask [B, 1, 1, S])."""
        # TODO: энкодер + маска источника (она понадобится декодеру).
        raise NotImplementedError("Seq2Seq.encode")

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """[B, T] → логиты [B, T, V]."""
        # TODO: декодер, затем lm_head.
        raise NotImplementedError("Seq2Seq.decode")

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        # TODO: encode, затем decode.
        raise NotImplementedError("Seq2Seq.forward")


@torch.no_grad()
def greedy_decode(model: Seq2Seq, src: torch.Tensor, max_len: int) -> torch.Tensor:
    """Жадная генерация: энкодер один раз, декодер — на растущем префиксе. → [B, L ≤ max_len] без BOS;
    после EOS — PAD; остановка, когда EOS есть в каждой строке."""
    # TODO 1: encode — один раз, до цикла.
    # TODO 2: начать с [BOS] в каждой строке; не больше max_len шагов: логиты ПОСЛЕДНЕЙ позиции → argmax.
    # TODO 3: строкам, где EOS уже был, дописывать PAD; остановиться, когда закончили все.
    # TODO 4: вернуть без столбца BOS.
    raise NotImplementedError("greedy_decode")
