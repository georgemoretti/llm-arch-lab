"""BERT — эталоны задач B07–B11 (блок B, модуль 2).

Задачи B08–B11 импортируют готовые части отсюда, а не из кода пользователя. Совпадение функций и классов с
``solution.py`` задач проверяет ``tests/test_core_bert.py``.

Имена параметров — как в Hugging Face ``BertModel`` / ``BertForMaskedLM`` / ``BertForSequenceClassification``:
state_dict из ``transformers`` загружается без переименований. Маски — bool, True = «можно смотреть».
Обозначения: B — batch, T — длина, H — hidden_size, V — словарь.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.encdec import attention   # эталон B01: softmax(q·kᵀ/√D + маска) @ v

# ---------------------------------------------------------------- B07: WordPiece


def is_punctuation(ch: str) -> bool:
    """Пунктуация в смысле BERT: все не-буквенно-цифровые ASCII-символы и всё из категорий Unicode P*."""
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def clean_text(text: str) -> str:
    """Убрать управляющие символы (и U+0000, U+FFFD), любые пробельные — заменить обычным пробелом."""
    out = []
    for ch in text:
        if ch in "\x00�":
            continue
        if ch in " \t\n\r" or unicodedata.category(ch) == "Zs":
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def strip_accents(text: str) -> str:
    """NFD-разложение и удаление комбинируемых знаков (категория Mn): «é» → «e», но и «й» → «и»."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def basic_tokenize(text: str, lower: bool = True) -> list[str]:
    """Пре-токенизация BERT: очистка, (lower + strip_accents), разрез по пробелам, каждый знак пунктуации — отдельно."""
    text = clean_text(text)
    if lower:
        text = strip_accents(text.lower())
    words = []
    for chunk in text.split():
        current = ""
        for ch in chunk:
            if is_punctuation(ch):
                if current:
                    words.append(current)
                words.append(ch)
                current = ""
            else:
                current += ch
        if current:
            words.append(current)
    return words


def wordpiece(word: str, vocab: dict[str, int], unk: str = "[UNK]", max_chars: int = 100) -> list[str]:
    """Жадный longest-match-first: самый длинный префикс из словаря, продолжения — с «##». Не разобралось — [unk]."""
    if len(word) > max_chars:
        return [unk]
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        while end > start:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                break
            end -= 1
        if end == start:              # ни один префикс не нашёлся → всё слово неизвестно
            return [unk]
        pieces.append(piece)
        start = end
    return pieces


def tokenize(text: str, vocab: dict[str, int], lower: bool = True) -> list[str]:
    """Текст → WordPiece-токены: basic_tokenize, затем wordpiece для каждого слова."""
    return [p for w in basic_tokenize(text, lower) for p in wordpiece(w, vocab)]


# ---------------------------------------------------------------- B08: вход BERT


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


# ---------------------------------------------------------------- B09: MLM-маскирование


def mlm_mask(input_ids: torch.Tensor, special_ids: list[int], mask_id: int, vocab_size: int, prob: float = 0.15,
             word_ids: torch.Tensor | None = None, generator: torch.Generator | None = None):
    """Маскирование для MLM → (inputs, labels).

    Выбирается доля prob обычных токенов (спецтокены и PAD не выбираются никогда). При word_ids [B, T] (номер слова
    токена, −1 у спецтокенов) выбираются целые слова: все их подтокены сразу. Из выбранных токенов 80 % → [MASK],
    10 % → случайный токен словаря, 10 % остаются. labels = исходный токен у выбранных, −100 у остальных.
    """
    special = torch.isin(input_ids, torch.tensor(special_ids, device=input_ids.device))
    if word_ids is None:
        chosen = torch.rand(input_ids.shape, generator=generator) < prob
    else:
        n_words = int(word_ids.max()) + 1
        word_chosen = torch.rand(input_ids.shape[0], max(n_words, 1), generator=generator) < prob   # [B, W]
        chosen = word_chosen.gather(1, word_ids.clamp(min=0)) & (word_ids >= 0)
    chosen &= ~special
    labels = input_ids.masked_fill(~chosen, -100)
    roll = torch.rand(input_ids.shape, generator=generator)                       # одна «кость» на токен
    to_mask = chosen & (roll < 0.8)
    to_random = chosen & (roll >= 0.8) & (roll < 0.9)
    random_tokens = torch.randint(vocab_size, input_ids.shape, generator=generator)
    inputs = input_ids.masked_fill(to_mask, mask_id)
    inputs = torch.where(to_random, random_tokens, inputs)
    return inputs, labels


# ---------------------------------------------------------------- B10: модель BERT


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


# ---------------------------------------------------------------- B11: головы дообучения


class SequenceClassifier(nn.Module):
    """Классификация текста: pooled-выход BERT → dropout → Linear(H, n_labels). Имена как в BertForSequenceClassification."""

    def __init__(self, cfg: BertConfig, num_labels: int):
        super().__init__()
        self.bert = BertModel(cfg)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(self.dropout(pooled))


def align_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    """Метки слов → метки токенов: метка слова на его первом подтокене, −100 на остальных подтокенах и спецтокенах."""
    labels, prev = [], None
    for w in word_ids:
        labels.append(-100 if w is None or w == prev else word_labels[w])
        prev = w
    return labels


def qa_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor,
            end_positions: torch.Tensor) -> torch.Tensor:
    """Extractive QA: среднее двух cross-entropy — по позиции начала и по позиции конца ответа. Логиты [B, T]."""
    return (F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2


def best_span(start_logits: torch.Tensor, end_logits: torch.Tensor, context_mask: torch.Tensor,
              max_answer_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Лучший ответ: argmax s_i + e_j при i ≤ j < i + max_answer_len, оба внутри контекста. [B, T] → (start [B], end [B])."""
    B, T = start_logits.shape
    scores = start_logits[:, :, None] + end_logits[:, None, :]                       # [B, T(start), T(end)]
    i = torch.arange(T, device=start_logits.device)
    allowed = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < max_answer_len)   # j ≥ i и длина ≤ L
    allowed = allowed & context_mask[:, :, None] & context_mask[:, None, :]
    flat = scores.masked_fill(~allowed, float("-inf")).view(B, -1).argmax(-1)
    return flat // T, flat % T
