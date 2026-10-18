"""Transformer encoder-decoder — эталоны задач B01–B05 (блок B, модуль 1).

Задачи B03–B06 импортируют готовые части отсюда, а не из кода пользователя: ошибка в B01 не должна ломать
следующие задачи. Совпадение функций и классов с ``solution.py`` задач проверяет ``tests/test_core_encdec.py``.

Архитектура — как в Vaswani et al. (2017): синусоидальные позиции, Post-LN (по флагу — Pre-LN), ReLU-FFN, общие
эмбеддинги энкодера, декодера и выходного слоя. Имена слоёв — как в ``nn.TransformerEncoderLayer`` /
``nn.TransformerDecoderLayer`` (self_attn, multihead_attn → cross_attn, linear1, linear2, norm1–norm3).
Маски — bool, ``True`` = «можно смотреть» (как в A16; у ``key_padding_mask`` PyTorch наоборот).
Обозначения: B — batch, T — длина цели (запросов), S — длина источника (ключей), C — d_model, H — головы.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- B01: padding-маска и multi-head attention


def padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B, S] → bool [B, 1, 1, S]: True у настоящих токенов. Оси 1 и 2 — головы и запросы (растянутся)."""
    return (ids != pad_id)[:, None, None, :]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """q [B, H, T, D], k [B, H, S, D], v [B, H, S, Dv] → (out [B, H, T, Dv], weights [B, H, T, S]).

    mask — bool, растягивается до [B, H, T, S]; True — ключ разрешён.
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])      # [B, H, T, S]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))          # запрещённые ключи — до softmax
    weights = F.softmax(scores, dim=-1)                            # распределение по ключам
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention: запросы из x [B, T, C], ключи и значения из context [B, S, C] (по умолчанию x)."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, C] → [B, H, L, C / H]."""
        B, L, C = x.shape
        return x.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                need_weights: bool = False):
        context = x if context is None else context
        q = self.split_heads(self.q_proj(x))                           # [B, H, T, D]
        k = self.split_heads(self.k_proj(context))                     # [B, H, S, D]
        v = self.split_heads(self.v_proj(context))                     # [B, H, S, D]
        out, weights = attention(q, k, v, mask)                        # [B, H, T, D]
        B, H, T, D = out.shape
        out = self.out_proj(out.transpose(1, 2).reshape(B, T, H * D))  # склеить головы → [B, T, C]
        return (out, weights) if need_weights else out


# ---------------------------------------------------------------- B02: синусоидальные позиции


def sinusoidal_positions(n_pos: int, d_model: int) -> torch.Tensor:
    """Таблица [n_pos, d_model]: PE[p, 2i] = sin(p·ω_i), PE[p, 2i+1] = cos(p·ω_i), ω_i = 10000^(−2i/d_model)."""
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]                        # [n_pos, 1]
    omega = 10000.0 ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)  # [d_model / 2]
    angle = pos * omega                                                            # [n_pos, d_model / 2]
    pe = torch.zeros(n_pos, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


def shift_matrix(k: int, d_model: int) -> torch.Tensor:
    """Матрица M [d_model, d_model], для которой PE[p + k] = M @ PE[p] при любом p: поворот каждой пары (sin, cos)."""
    omega = 10000.0 ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    c, s = torch.cos(k * omega), torch.sin(k * omega)
    m = torch.zeros(d_model, d_model)
    m[0::2, 0::2] = torch.diag(c)      # sin(a + b) = sin a · cos b + cos a · sin b
    m[0::2, 1::2] = torch.diag(s)
    m[1::2, 0::2] = torch.diag(-s)     # cos(a + b) = cos a · cos b − sin a · sin b
    m[1::2, 1::2] = torch.diag(c)
    return m


class TransformerEmbedding(nn.Module):
    """Эмбеддинг токена × √d_model + синусоидальная позиция: [B, T] → [B, T, C]."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_id: int):
        super().__init__()
        self.d_model, self.max_len = d_model, max_len
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # Матрица общая с выходным слоем (B05), поэтому маленькая: std = 1/√C. Умножение на √C в forward
        # возвращает эмбеддингам масштаб ~1 — тот же, что у синусоид.
        nn.init.normal_(self.tok.weight, std=d_model ** -0.5)
        with torch.no_grad():
            self.tok.weight[pad_id].zero_()
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        if T > self.max_len:
            raise ValueError(f"длина {T} больше max_len = {self.max_len}")
        return self.tok(ids) * math.sqrt(self.d_model) + self.pe[:T]


# ---------------------------------------------------------------- B03: энкодер


class FeedForward(nn.Module):
    """FFN статьи: Linear C → d_ff, ReLU, Linear d_ff → C."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """Слой энкодера: двунаправленный self-attention и FFN; Post-LN (как в статье) или Pre-LN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_first: bool = False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), mask=mask)
            return x + self.ffn(self.norm2(x))
        x = self.norm1(x + self.self_attn(x, mask=mask))
        return self.norm2(x + self.ffn(x))


class Encoder(nn.Module):
    """ids источника [B, S] → память [B, S, C]: эмбеддинги + n_layers слоёв (+ финальный LayerNorm у Pre-LN)."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int,
                 pad_id: int, norm_first: bool = False):
        super().__init__()
        self.pad_id = pad_id
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, pad_id)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, norm_first) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask = padding_mask(src, self.pad_id)        # [B, 1, 1, S]
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ---------------------------------------------------------------- B04: декодер и cross-attention


def causal_mask(T: int) -> torch.Tensor:
    """bool [T, T]: True на диагонали и ниже."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def target_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Маска self-attention декодера: causal И не PAD-ключ. [B, T] → bool [B, 1, T, T]."""
    return padding_mask(tgt, pad_id) & causal_mask(tgt.shape[1]).to(tgt.device)


class DecoderLayer(nn.Module):
    """Слой декодера: masked self-attention → cross-attention к памяти энкодера → FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_first: bool = False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
            x = x + self.cross_attn(self.norm2(x), memory, mask=src_mask)
            return x + self.ffn(self.norm3(x))
        x = self.norm1(x + self.self_attn(x, mask=tgt_mask))
        x = self.norm2(x + self.cross_attn(x, memory, mask=src_mask))   # Q из декодера, K и V из памяти
        return self.norm3(x + self.ffn(x))


class Decoder(nn.Module):
    """ids цели [B, T] + память [B, S, C] → скрытые состояния [B, T, C]."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int,
                 pad_id: int, norm_first: bool = False):
        super().__init__()
        self.pad_id = pad_id
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, pad_id)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, norm_first) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        tgt_mask = target_mask(tgt, self.pad_id)     # [B, 1, T, T]
        x = self.embed(tgt)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


# ---------------------------------------------------------------- B05: seq2seq целиком


def shift_right(tgt: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Вход декодера при teacher forcing: [y0, y1, …, y_{T−1}] → [BOS, y0, …, y_{T−2}]."""
    bos = torch.full_like(tgt[:, :1], bos_id)
    return torch.cat([bos, tgt[:, :-1]], dim=1)


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
