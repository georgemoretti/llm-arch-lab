"""T5 — эталоны задач B13–B15 (блок B, модуль 3).

Задача B14 импортирует отсюда относительные позиции (B13), а не из кода пользователя. Совпадение функций и
классов с ``solution.py`` задач проверяет ``tests/test_core_t5.py``.

Имена параметров — как в Hugging Face ``T5Model``: state_dict из ``transformers`` загружается без
переименований. Маски — bool, True = «можно смотреть». Обозначения: B — batch, T — длина запросов, S — длина
ключей, H — головы, D — d_kv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- B13: относительный позиционный bias


def relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32,
                             max_distance: int = 128) -> torch.Tensor:
    """Номер бакета для относительной позиции key − query (тензор long любой формы → такой же формы).

    Двунаправленный режим: половина бакетов — для ключей слева (и на месте), половина — для ключей справа.
    Causal: ключи справа запрещены маской, их бакет — 0, все бакеты отданы прошлому.
    Внутри половины: расстояния 0 … max_exact − 1 — каждое в своём бакете, дальше — логарифмически до
    max_distance; всё, что дальше, — в последнем бакете.
    """
    buckets = torch.zeros_like(relative_position)
    if bidirectional:
        num_buckets //= 2
        buckets = buckets + (relative_position > 0).long() * num_buckets     # ключ справа → вторая половина
        n = relative_position.abs()
    else:
        n = (-relative_position).clamp(min=0)                                # расстояние назад; будущее → 0
    max_exact = num_buckets // 2
    is_small = n < max_exact
    log_ratio = torch.log(n.clamp(min=1).float() / max_exact) / math.log(max_distance / max_exact)
    large = max_exact + (log_ratio * (num_buckets - max_exact)).long()
    large = large.clamp(max=num_buckets - 1)
    return buckets + torch.where(is_small, n, large)


def relative_position_bias(bias: nn.Embedding, query_length: int, key_length: int, bidirectional: bool = True,
                           num_buckets: int = 32, max_distance: int = 128, offset: int = 0) -> torch.Tensor:
    """Добавка к scores: таблица bias [num_buckets, H] → [1, H, T, S].

    offset — абсолютная позиция первого запроса (при генерации с кешем запрос один, а ключей много).
    """
    device = bias.weight.device
    q_pos = torch.arange(query_length, device=device)[:, None] + offset    # [T, 1]
    k_pos = torch.arange(key_length, device=device)[None, :]               # [1, S]
    buckets = relative_position_bucket(k_pos - q_pos, bidirectional, num_buckets, max_distance)   # [T, S]
    return bias(buckets).permute(2, 0, 1).unsqueeze(0)                     # [T, S, H] → [1, H, T, S]


# ---------------------------------------------------------------- B14: слой T5


@dataclass
class T5Config:
    vocab_size: int
    d_model: int = 32
    d_kv: int = 8
    d_ff: int = 64
    num_layers: int = 2
    num_decoder_layers: int | None = None
    num_heads: int = 4
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6
    pad_token_id: int = 0
    decoder_start_token_id: int = 0


class T5LayerNorm(nn.Module):
    """RMSNorm: x / √(mean(x²) + eps) · weight — без вычитания среднего и без bias. Статистика — в fp32."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * (x.float() * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)


class T5Attention(nn.Module):
    """Multi-head attention T5: q / k / v / o без bias, scores без 1/√D, плюс относительный bias (только в слое 0)."""

    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
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
        B, T, _ = x.shape
        src = x if kv is None else kv
        S = src.shape[1]
        split = lambda t, L: t.view(B, L, self.n_heads, self.d_kv).transpose(1, 2)   # [B, H, L, D]
        q, k, v = split(self.q(x), T), split(self.k(src), S), split(self.v(src), S)
        scores = q @ k.transpose(-2, -1)                                    # без 1/√D: масштаб — в инициализации
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = relative_position_bias(
                    self.relative_attention_bias, T, S, bidirectional=not self.is_decoder,
                    num_buckets=self.cfg.relative_attention_num_buckets,
                    max_distance=self.cfg.relative_attention_max_distance)
            else:
                position_bias = scores.new_zeros(1, self.n_heads, T, S)    # cross-attention: позиций нет
        scores = scores + position_bias
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        out = (weights @ v).transpose(1, 2).reshape(B, T, self.n_heads * self.d_kv)
        return self.o(out), position_bias


class T5Block(nn.Module):
    """Блок T5 (Pre-LN): x + SelfAttention(LN(x)) → [x + EncDecAttention(LN(x), enc)] → x + FFN(LN(x)).

    Имена как в HF: layer[0].SelfAttention / layer_norm, у декодера layer[1].EncDecAttention / layer_norm,
    последний — layer[-1].DenseReluDense.wi / wo / layer_norm. FFN: wo(ReLU(wi(x))), без bias.
    """

    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
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
        sa = self.layer[0]
        a, position_bias = sa.SelfAttention(sa.layer_norm(x), mask, position_bias=position_bias)
        x = x + a
        if self.is_decoder and encoder_hidden_states is not None:
            ca = self.layer[1]
            c, _ = ca.EncDecAttention(ca.layer_norm(x), encoder_mask, kv=encoder_hidden_states)
            x = x + c
        ff = self.layer[-1]
        x = x + ff.DenseReluDense.wo(F.relu(ff.DenseReluDense.wi(ff.layer_norm(x))))
        return x, position_bias


class T5Stack(nn.Module):
    """Энкодер или декодер T5: эмбеддинги (без позиций и без масштаба) → блоки с общим bias → финальная норма."""

    def __init__(self, cfg: T5Config, embed_tokens: nn.Embedding, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        n = (cfg.num_decoder_layers or cfg.num_layers) if is_decoder else cfg.num_layers
        self.embed_tokens = embed_tokens
        self.block = nn.ModuleList(T5Block(cfg, i == 0, is_decoder) for i in range(n))
        self.final_layer_norm = T5LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.bool()[:, None, None, :]                      # [B, 1, 1, T]: PAD-ключи запрещены
        if self.is_decoder:
            mask = mask & torch.tril(torch.ones(T, T, dtype=torch.bool, device=input_ids.device))
        enc_mask = None
        if encoder_attention_mask is not None:
            enc_mask = encoder_attention_mask.bool()[:, None, None, :]
        x = self.embed_tokens(input_ids)
        position_bias = None                                                # считает слой 0, остальные берут его
        for block in self.block:
            x, position_bias = block(x, mask, position_bias, encoder_hidden_states, enc_mask)
        return self.final_layer_norm(x)


class T5Model(nn.Module):
    """Энкодер + декодер с общей матрицей эмбеддингов shared. forward → выход декодера [B, T, C]."""

    def __init__(self, cfg: T5Config):
        super().__init__()
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


# ---------------------------------------------------------------- B15: span corruption


def random_segmentation(num_items: int, num_segments: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Случайное разбиение num_items на num_segments непустых кусков: длины [num_segments], сумма = num_items."""
    cuts = torch.randperm(num_items - 1, generator=generator)[: num_segments - 1].sort().values + 1
    bounds = torch.cat([torch.tensor([0]), cuts, torch.tensor([num_items])])
    return bounds.diff()


def random_spans_noise_mask(length: int, noise_density: float = 0.15, mean_noise_span_length: float = 3.0,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Маска шума T5 (bool [length]): round(length · density) токенов в round(·/mean) спанах.

    Спаны текста и шума чередуются; первым идёт текст, последним — шум (как в T5).
    """
    num_noise = min(max(int(round(length * noise_density)), 1), length - 1)
    num_spans = max(int(round(num_noise / mean_noise_span_length)), 1)
    noise_lengths = random_segmentation(num_noise, num_spans, generator)
    text_lengths = random_segmentation(length - num_noise, num_spans, generator)
    lengths = torch.stack([text_lengths, noise_lengths], dim=1).reshape(-1)   # текст, шум, текст, шум, …
    flags = torch.tensor([False, True]).repeat(num_spans)
    return flags.repeat_interleave(lengths)


def noise_span_to_unique_sentinel(ids: torch.Tensor, noise_mask: torch.Tensor, sentinels: list[int]) -> torch.Tensor:
    """Каждый спан шума → один sentinel (i-й спан → sentinels[i]); остальные токены — как есть. [L] → [L']."""
    prev_is_noise = torch.cat([noise_mask.new_zeros(1), noise_mask[:-1]])
    first = noise_mask & ~prev_is_noise                 # первый токен спана — станет sentinel
    rest = noise_mask & prev_is_noise                   # остальные токены спана — выкидываем
    span = torch.cumsum(first.long(), 0) - 1            # номер спана у каждого токена
    if int(first.sum()) > len(sentinels):
        raise ValueError(f"спанов {int(first.sum())}, а sentinel-токенов только {len(sentinels)}")
    table = torch.tensor(sentinels, dtype=ids.dtype, device=ids.device)
    out = torch.where(first, table[span.clamp(min=0, max=len(sentinels) - 1)], ids)
    return out[~rest]


def span_corruption(ids: torch.Tensor, noise_mask: torch.Tensor, sentinels: list[int],
                    eos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Вход и цель T5: во входе спаны шума заменены sentinel-ами, в цели — наоборот, спаны текста; обе с EOS.

    Выравнивание sentinel-ов держится на том, что маска начинается с текста (так строит random_spans_noise_mask).
    """
    eos = torch.tensor([eos_id], dtype=ids.dtype, device=ids.device)
    inputs = torch.cat([noise_span_to_unique_sentinel(ids, noise_mask, sentinels), eos])
    targets = torch.cat([noise_span_to_unique_sentinel(ids, ~noise_mask, sentinels), eos])
    return inputs, targets
