"""Seq2seq Transformer целиком: shift right, label smoothing, общая матрица эмбеддингов, жадная генерация.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.encdec import Decoder, Encoder, padding_mask   # эталоны B01–B04


def shift_right(tgt: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Вход декодера при teacher forcing: [y0, y1, …, y_{T−1}] → [BOS, y0, …, y_{T−2}]."""
    return torch.roll(tgt, shifts=1, dims=1)          # сдвиг вправо на одну позицию


def label_smoothed_ce(logits: torch.Tensor, target: torch.Tensor, eps: float = 0.1,
                      ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy с label smoothing: цель (1 − ε)·one-hot + ε/V; среднее по позициям, где target ≠ ignore_index."""
    logp = F.log_softmax(logits.float(), dim=-1)                      # [..., V]
    keep = target != ignore_index
    safe = target.masked_fill(~keep, 0)                               # gather не любит отрицательные индексы
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)            # −log p(y)
    uniform = -logp.mean(dim=-1)                                      # −(1/V) Σ_v log p(v)
    loss = (1 - eps) * nll + eps * uniform
    return loss[keep].sum() / keep.sum().clamp(min=1)


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
        return self.encoder(src), padding_mask(src, self.cfg.pad_id)

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """[B, T] → логиты [B, T, V]."""
        return self.lm_head(self.decoder(tgt_in, memory, src_mask))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)


@torch.no_grad()
def greedy_decode(model: Seq2Seq, src: torch.Tensor, max_len: int) -> torch.Tensor:
    """Жадная генерация: энкодер один раз, декодер — на растущем префиксе. → [B, L ≤ max_len] без BOS;
    после EOS — PAD; остановка, когда EOS есть в каждой строке."""
    cfg = model.cfg
    memory, src_mask = model.encode(src)
    out = torch.full((src.shape[0], 1), cfg.bos_id, dtype=torch.long, device=src.device)
    done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        next_tok = model.decode(out, memory, src_mask)[:, -1].argmax(-1)   # логиты последней позиции
        next_tok = next_tok.masked_fill(done, cfg.pad_id)                  # закончившим — PAD
        out = torch.cat([out, next_tok[:, None]], dim=1)
        done |= next_tok == cfg.eos_id
        if done.all():
            break
    return out[:, 1:]
