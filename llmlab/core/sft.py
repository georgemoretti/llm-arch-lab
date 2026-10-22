"""SFT — эталоны задач D01–D03 (блок D, модуль 1) и общие детали блока D.

Задачи D02–D04 (и дальше D10, D11, D15) импортируют готовые части отсюда, а не из кода пользователя. Совпадение
функций с ``solution.py`` задач проверяет ``tests/test_core_sft.py``.

Что здесь лежит:
- ``CharTokenizer`` — посимвольный токенизатор игрушечных задач со служебными токенами в стиле Qwen3
  (``<|im_start|>``, ``<|im_end|>``, ``<think>``, ``</think>``, ``<|endoftext|>`` = PAD);
- генератор синтетических инструкций (арифметика словами, «заглавными буквами», JSON по шаблону) и тот же материал
  сплошным текстом для «предобучения» базовой модели;
- D01: ``render_chat`` (правила chat-шаблона Qwen3 без tools), ``split_special``, ``encode_chat``;
- D02: ``assistant_mask``, ``sft_targets``, ``pack``, ``packed_causal_mask``;
- D03: ``masked_ce_sum``, ``accumulate_gradients``, ``sft_step`` (+ ``lr_at`` из A08 и ``collate``).

Соглашение о метках: ``targets[t]`` — токен, который модель должна предсказать на позиции ``t`` (то есть
``ids[t + 1]``), или ``IGNORE = -100``. Сдвиг сделан при подготовке данных, а не внутри модели, поэтому лосс —
обычная cross-entropy ``logits[:, t]`` против ``targets[:, t]``.
"""

from __future__ import annotations

import math
import random
import re

import torch
import torch.nn.functional as F

from llmlab.core.model import ModelConfig

IGNORE = -100

# ---------------------------------------------------------------- токенизатор игрушечных задач

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
ALPHABET = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \n.,:;?!-—{}[]\"'()+=*/<>|_"
)


class CharTokenizer:
    """Посимвольный токенизатор: id 0–4 — служебные токены, дальше — символы ``ALPHABET``.

    Интерфейс тот же, что нужен ``encode_chat`` (D01) от настоящего токенизатора: ``special_tokens``,
    ``token_to_id``, ``encode`` (текст БЕЗ разбора служебных токенов), ``decode``.
    """

    def __init__(self, alphabet: str = ALPHABET, special_tokens: list[str] = SPECIAL_TOKENS):
        self.special_tokens = list(special_tokens)
        self.itos = self.special_tokens + list(alphabet)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi["<|endoftext|>"]
        self.eos_id = self.stoi["<|im_end|>"]          # конец хода: на нём генерация ответа останавливается

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def token_to_id(self, token: str) -> int:
        return self.stoi[token]

    def encode(self, text: str) -> list[int]:
        """Обычный текст → id по символу. Служебные токены здесь НЕ распознаются: «<|im_end|>» в тексте — 10 символов."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"символа {exc.args[0]!r} нет в алфавите игрушечного токенизатора") from None

    def decode(self, ids, skip_special: bool = False) -> str:
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        return "".join(self.itos[i] for i in ids if not (skip_special and i < len(self.special_tokens)))


# ---------------------------------------------------------------- синтетические инструкции

NUMBER_WORDS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
                "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
                "семнадцать", "восемнадцать", "девятнадцать", "двадцать"]
CONSONANTS = "бвгдзклмнпрстфхчшж"
VOWELS = "аеиоуяю"
ARITH_TEMPLATES = ["Сколько будет {a} {op} {b}?", "Посчитай: {a} {op} {b}", "{a} {op} {b} — это сколько?"]
UPPER_TEMPLATES = ["Напиши заглавными буквами: {w}", "Сделай слово заглавным: {w}"]
JSON_TEMPLATES = ["Сделай JSON: имя {name}, возраст {age}", "Запиши в JSON: {name}, {age} лет"]
TASKS = ("arith", "upper", "json")


def _held_out(key: str) -> bool:
    """Детерминированное разбиение train/test: ~20 % комбинаций только в тесте (не зависит от PYTHONHASHSEED)."""
    return sum(map(ord, key)) * 2654435761 % 5 == 0


def _arith(rng: random.Random):
    while True:
        op = rng.choice(["плюс", "минус"])
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        if op == "минус" and b > a:
            a, b = b, a
        if op == "плюс" and a + b > 20:
            continue                                            # ответы — до двадцати, как и числа в вопросе
        yield (f"{a}{op}{b}", rng.choice(ARITH_TEMPLATES).format(a=NUMBER_WORDS[a], op=op, b=NUMBER_WORDS[b]),
               str(a + b if op == "плюс" else a - b))


def pseudo_word(rng: random.Random, min_syl: int = 1, max_syl: int = 3) -> str:
    """Псевдослово из слогов «согласная + гласная» («мокату»): их тысячи, запомнить ответы списком нельзя —
    нужен навык (поднять регистр, скопировать имя), а не таблица фактов."""
    word = "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(rng.randint(min_syl, max_syl)))
    return word + (rng.choice(CONSONANTS) if rng.random() < 0.3 else "")


def _upper(rng: random.Random):
    while True:
        w = pseudo_word(rng)
        yield w, rng.choice(UPPER_TEMPLATES).format(w=w), w.upper()


def _json(rng: random.Random):
    while True:
        name, age = pseudo_word(rng, 2, 3).capitalize(), rng.randint(18, 80)
        yield (name, rng.choice(JSON_TEMPLATES).format(name=name, age=age),
               '{"name": "%s", "age": %d}' % (name, age))


def make_instructions(n: int, seed: int = 0, split: str = "train", tasks=TASKS) -> list[dict]:
    """n примеров ``{"task", "prompt", "answer"}`` по кругу из ``tasks``. ``split="test"`` — только отложенные
    комбинации (пары чисел, слова, имена), которых нет в ``train``; ``"all"`` — без разбиения (материал
    «предобучения» в D04: база знает все факты, SFT видит только часть)."""
    if split not in ("train", "test", "all"):
        raise ValueError("split — 'train', 'test' или 'all'")
    rng = random.Random(seed)
    gens = {"arith": _arith(rng), "upper": _upper(rng), "json": _json(rng)}
    out = []
    while len(out) < n:
        task = tasks[len(out) % len(tasks)]
        key, prompt, answer = next(gens[task])
        if split == "all" or _held_out(key) == (split == "test"):
            out.append({"task": task, "prompt": prompt, "answer": answer})
    return out


def plain_text(example: dict) -> str:
    """Тот же факт сплошным текстом, без ролей и chat-шаблона, — материал «предобучения» базовой модели в D04."""
    return f"{example['prompt']} Ответ: {example['answer']}.\n"


def to_messages(example: dict, with_answer: bool = True) -> list[dict]:
    msgs = [{"role": "user", "content": example["prompt"]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": example["answer"]})
    return msgs


def toy_config(vocab_size: int, d_model: int = 64, n_layers: int = 2) -> ModelConfig:
    """Llama-подобная модель (``llama-like``, C06) игрушечного размера под словарь ``CharTokenizer``."""
    base = ModelConfig.named("llama-like")
    return base.model_copy(update={"name": "sft-toy", "vocab_size": vocab_size, "d_model": d_model,
                                   "n_layers": n_layers, "tie_embeddings": True})


# ---------------------------------------------------------------- D01: chat-шаблон


def render_chat(messages: list[dict], add_generation_prompt: bool = False, enable_thinking: bool = True) -> str:
    """Список ходов → строка chat-шаблона Qwen3 (без tools) == ``tokenizer.apply_chat_template(..., tokenize=False)``.

    Каждый ход — ``<|im_start|>{role}\\n{content}<|im_end|>\\n``. У ассистента рассуждение ``<think>…</think>``
    вырезается из ходов ДО последнего вопроса пользователя; последний ход ассистента после этого вопроса получает
    блок ``<think>\\n{рассуждение}\\n</think>\\n\\n`` (пустой, если рассуждения нет).
    """
    last_query = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_query = i
            break
    out = []
    for i, msg in enumerate(messages):
        role, content = msg["role"], msg["content"]
        if role in ("system", "user"):
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif role == "assistant":
            reasoning = msg.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = ""
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = content.split("</think>")[-1].lstrip("\n")
            if i > last_query and (i == len(messages) - 1 or reasoning):
                think, answer = reasoning.strip("\n"), content.lstrip("\n")
                out.append(f"<|im_start|>assistant\n<think>\n{think}\n</think>\n\n{answer}")
            else:
                out.append(f"<|im_start|>assistant\n{content}")
            out.append("<|im_end|>\n")
        else:
            raise ValueError(f"роль {role!r} не поддержана: только system, user, assistant")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        if not enable_thinking:
            out.append("<think>\n\n</think>\n\n")
    return "".join(out)


def split_special(text: str, special_tokens: list[str]) -> list[str]:
    """Разрезать строку по служебным токенам: служебный токен — отдельный кусок, пустых кусков нет."""
    if not special_tokens:
        return [text] if text else []
    pattern = "(" + "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)) + ")"
    return [piece for piece in re.split(pattern, text) if piece]


def encode_chat(text: str, tok) -> list[int]:
    """Строка шаблона → id: служебный токен — один id (``tok.token_to_id``), текст между ними — ``tok.encode``."""
    ids: list[int] = []
    for piece in split_special(text, tok.special_tokens):
        if piece in tok.special_tokens:
            ids.append(tok.token_to_id(piece))
        else:
            ids.extend(tok.encode(piece))
    return ids


class HFChatTokenizer:
    """Обёртка над ``tokenizers.Tokenizer`` настоящей модели с тем же интерфейсом, что у ``CharTokenizer``.

    ``special_tokens`` — все добавленные токены (у Qwen3 это и ``<|im_start|>``, и ``<think>``): быстрый
    токенизатор HF вырезает их из текста до BPE, так же поступает ``encode_chat``.
    """

    def __init__(self, raw):
        self.raw = raw
        self.special_tokens = [t.content for t in raw.get_added_tokens_decoder().values()]
        self.eos_id = raw.token_to_id("<|im_end|>")
        self.pad_id = raw.token_to_id("<|endoftext|>")

    @property
    def vocab_size(self) -> int:
        return self.raw.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self.raw.token_to_id(token)

    def encode(self, text: str) -> list[int]:
        return self.raw.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special: bool = False) -> str:
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        return self.raw.decode(ids, skip_special_tokens=skip_special)


def assistant_header(tok) -> list[int]:
    """id заголовка хода ассистента ``<|im_start|>assistant\\n`` — по нему D02 ищет ответы."""
    return encode_chat("<|im_start|>assistant\n", tok)


# ---------------------------------------------------------------- D02: маска лосса и упаковка


def assistant_mask(ids, header: list[int], end_id: int) -> torch.Tensor:
    """bool [T]: True у токенов ответов ассистента — всё ПОСЛЕ заголовка ``header`` до ``end_id`` включительно.

    Сам заголовок (``<|im_start|>assistant\\n``) — контекст: его вставляет шаблон, модели его не генерировать.
    ``<|im_end|>`` ответа — в маске: на нём модель учится останавливаться. Ход, обрезанный до ``end_id``, —
    в маске до конца строки.
    """
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    mask = [False] * len(ids)
    h, i = len(header), 0
    while i < len(ids):
        if h and ids[i:i + h] == header:
            j = i + h
            while j < len(ids) and ids[j] != end_id:
                mask[j] = True
                j += 1
            if j < len(ids):
                mask[j] = True
            i = j + 1
        else:
            i += 1
    return torch.tensor(mask, dtype=torch.bool)


def sft_targets(ids, mask: torch.Tensor) -> torch.Tensor:
    """Метки со сдвигом: ``targets[t] = ids[t + 1]``, если токен ``t + 1`` — часть ответа, иначе ``IGNORE``.

    Позиция ``t`` видит токены ``0…t`` и предсказывает следующий. У последней позиции следующего нет → ``IGNORE``.
    """
    ids = torch.as_tensor(ids, dtype=torch.long)
    targets = torch.full_like(ids, IGNORE)
    targets[:-1] = torch.where(mask[1:], ids[1:], torch.full_like(ids[1:], IGNORE))
    return targets


def pack(examples: list[tuple], max_len: int) -> list[dict]:
    """Жадно уложить примеры ``(ids, targets)`` по порядку в строки длиной ≤ ``max_len``.

    Строка: ``input_ids`` [L], ``targets`` [L], ``position_ids`` [L] (с нуля у каждого примера), ``cu_seqlens``
    [n + 1] int32 — границы примеров, как в B17. Пример длиннее ``max_len`` — ValueError (обрезать до упаковки).
    """
    rows, cur = [], []

    def flush():
        if cur:
            ids = torch.cat([torch.as_tensor(x, dtype=torch.long) for x, _ in cur])
            tgt = torch.cat([torch.as_tensor(t, dtype=torch.long) for _, t in cur])
            lens = [len(x) for x, _ in cur]
            pos = torch.cat([torch.arange(n) for n in lens])
            cu = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
            rows.append({"input_ids": ids, "targets": tgt, "position_ids": pos, "cu_seqlens": cu})
            cur.clear()

    used = 0
    for ids, tgt in examples:
        n = len(ids)
        if n > max_len:
            raise ValueError(f"пример длиной {n} не помещается в max_len = {max_len}: обрежь его до упаковки")
        if used + n > max_len:
            flush()
            used = 0
        cur.append((ids, tgt))
        used += n
    flush()
    return rows


def packed_causal_mask(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """bool [L, L], True = «можно смотреть»: causal внутри своего примера, соседей по строке не видно."""
    cu = cu_seqlens.long()
    L = int(cu[-1])
    seg = torch.searchsorted(cu[1:], torch.arange(L), right=True)          # номер примера у каждой позиции
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    return causal & (seg[:, None] == seg[None, :])


# ---------------------------------------------------------------- D03: шаг SFT


def lr_at(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    """lr на шаге step (с нуля): линейный рост до max_lr за warmup_steps, затем cosine до min_lr к total_steps."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps        # шаг 0 уже с ненулевым lr
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)   # 0 → 1
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def collate(examples: list[tuple], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Примеры ``(ids, targets)`` разной длины → ``input_ids`` [B, T] (PAD справа) и ``targets`` [B, T] (IGNORE).

    PAD справа безопасен для causal-модели: настоящие токены не видят того, что правее.
    """
    T = max(len(x) for x, _ in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    tgt = torch.full((len(examples), T), IGNORE, dtype=torch.long)
    for b, (x, t) in enumerate(examples):
        ids[b, :len(x)] = torch.as_tensor(x, dtype=torch.long)
        tgt[b, :len(t)] = torch.as_tensor(t, dtype=torch.long)
    return ids, tgt


def masked_ce_sum(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Сумма cross-entropy по позициям с меткой (≠ IGNORE) и число таких позиций. logits [B, T, V], targets [B, T]."""
    loss = F.cross_entropy(logits.flatten(0, -2).float(), targets.flatten(), ignore_index=IGNORE, reduction="sum")
    return loss, int((targets != IGNORE).sum())


def accumulate_gradients(model, micro_batches: list[tuple]) -> tuple[float, int]:
    """backward по микро-батчам ``(input_ids, targets)`` так, чтобы .grad == градиент среднего по ВСЕМ токенам ответа.

    Каждый микро-батч вносит ``сумма CE / N``, где N — число меченых токенов во всех микро-батчах шага. Возвращает
    (средний лосс на токен, N).
    """
    n_total = sum(int((t != IGNORE).sum()) for _, t in micro_batches)
    if n_total == 0:
        raise ValueError("в шаге нет ни одного токена ответа: все метки = IGNORE")
    total = 0.0
    for ids, targets in micro_batches:
        loss_sum, _ = masked_ce_sum(model(ids), targets)
        (loss_sum / n_total).backward()
        total += float(loss_sum.detach())
    return total / n_total, n_total


def sft_step(model, optimizer, micro_batches: list[tuple], step: int, *, max_lr: float, warmup_steps: int,
             total_steps: int, min_lr: float = 0.0, max_norm: float = 1.0) -> dict:
    """Один шаг оптимизатора: lr по расписанию → zero_grad → накопление → clip → step. Возвращает статистику."""
    lr = lr_at(step, max_lr, warmup_steps, total_steps, min_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss, n_tokens = accumulate_gradients(model, micro_batches)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    optimizer.step()
    return {"loss": loss, "tokens": n_tokens, "lr": lr, "grad_norm": grad_norm}
