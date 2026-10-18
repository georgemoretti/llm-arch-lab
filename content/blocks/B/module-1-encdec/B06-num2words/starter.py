"""Проект: seq2seq «число цифрами → число словами» (0 … 999 999, по-русски).

Реализуй encode_pair, collate, train_step и exact_match. Данные и обучение целиком готовы.
Обучение целиком (~25 с на CPU):  python workspace/B06-num2words/solution.py
Модель, лосс и генерация — эталоны B05 из llmlab.core.encdec.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from llmlab.core.encdec import Seq2Seq, Seq2SeqConfig, greedy_decode, label_smoothed_ce, shift_right  # B05

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<eos>"]
MAX_NUMBER = 999_999
MAX_WORDS = 7            # «девятьсот девяносто девять тысяч девятьсот девяносто девять»

# ---------------------------------------------------------------- данные (готово)

ONES = ("ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать "
        "четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать").split()
TENS = "_ _ двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split()
HUNDREDS = "_ сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split()


def _under_1000(n: int, feminine: bool = False) -> list[str]:
    words = []
    h, r = divmod(n, 100)
    if h:
        words.append(HUNDREDS[h])
    if r >= 20:
        t, o = divmod(r, 10)
        words.append(TENS[t])
        if o:
            words.append(ONES[o])
    elif r:
        words.append(ONES[r])
    if feminine and words and words[-1] in ("один", "два"):   # одна тысяча, две тысячи
        words[-1] = {"один": "одна", "два": "две"}[words[-1]]
    return words


def number_to_words(n: int) -> list[str]:
    """123 456 → ['сто', 'двадцать', 'три', 'тысячи', 'четыреста', 'пятьдесят', 'шесть']."""
    if not 0 <= n <= MAX_NUMBER:
        raise ValueError(f"число вне диапазона 0 … {MAX_NUMBER}: {n}")
    if n == 0:
        return ["ноль"]
    thousands, rest = divmod(n, 1000)
    words = []
    if thousands:
        words += _under_1000(thousands, feminine=True)
        last2, last = thousands % 100, thousands % 10
        if 10 < last2 < 20 or last == 0 or last >= 5:
            words.append("тысяч")
        elif last == 1:
            words.append("тысяча")
        else:
            words.append("тысячи")
    return words + _under_1000(rest)


def build_vocab() -> tuple[list[str], dict[str, int]]:
    """Общий словарь источника и цели: спецтокены, цифры, слова."""
    words = sorted({w for n in list(range(1000)) + [k * 1000 for k in range(1, 1000)] for w in number_to_words(n)})
    itos = SPECIALS + list("0123456789") + words
    return itos, {s: i for i, s in enumerate(itos)}


def split_numbers(n_train: int = 2000, n_val: int = 500, seed: int = 0) -> tuple[list[int], list[int]]:
    """Непересекающиеся случайные выборки чисел из 0 … 999 999."""
    nums = random.Random(seed).sample(range(MAX_NUMBER + 1), n_train + n_val)
    return nums[:n_train], nums[n_train:]


# ---------------------------------------------------------------- то, что пишешь ты


def encode_pair(n: int, stoi: dict[str, int]) -> tuple[list[int], list[int]]:
    """Число → (источник: id цифр, цель: id слов + EOS). Без BOS: его добавит shift_right."""
    # TODO: источник — id каждой цифры str(n); цель — id слов number_to_words(n) и EOS в конце.
    raise NotImplementedError("encode_pair")


def collate(pairs: list[tuple[list[int], list[int]]], pad_id: int = PAD) -> tuple[torch.Tensor, torch.Tensor]:
    """Список пар разной длины → (src [B, S_max], tgt [B, T_max]), PAD справа, dtype long."""
    # TODO: дополни каждый источник и каждую цель PAD справа до максимальной длины в батче.
    raise NotImplementedError("collate")


def train_step(model: Seq2Seq, opt: torch.optim.Optimizer, src: torch.Tensor, tgt: torch.Tensor,
               eps: float = 0.1, clip: float = 1.0) -> float:
    """Один шаг teacher forcing: shift right → логиты → label smoothing (PAD игнорируется) → clip → шаг."""
    # TODO 1: model.train(); вход декодера — shift_right(tgt, BOS); логиты model(src, вход декодера).
    # TODO 2: label_smoothed_ce с eps, PAD игнорируется. Цель — tgt (а не сдвинутая).
    # TODO 3: zero_grad → backward → clip_grad_norm_(…, clip) → step. Вернуть loss как float.
    raise NotImplementedError("train_step")


def exact_match(model: Seq2Seq, numbers: list[int], stoi: dict[str, int]) -> float:
    """Доля чисел, для которых жадная генерация совпала с целью целиком (слова и EOS, без хвоста PAD)."""
    # TODO 1: eval-режим на время генерации (и вернуть прежний режим в конце).
    # TODO 2: батч из encode_pair → collate; greedy_decode(model, src, MAX_WORDS + 1).
    # TODO 3: для каждой строки: цель без PAD; ответ — до первого EOS включительно (если EOS есть).
    #         Совпало целиком — попадание. Вернуть долю попаданий.
    raise NotImplementedError("exact_match")


# ---------------------------------------------------------------- обучение целиком (готово)


def make_model(vocab_size: int, seed: int = 0) -> Seq2Seq:
    torch.manual_seed(seed)
    return Seq2Seq(Seq2SeqConfig(vocab_size=vocab_size, d_model=64, n_heads=4, d_ff=256, n_layers=2, max_len=16,
                                 pad_id=PAD, bos_id=BOS, eos_id=EOS))


def cross_attention(model: Seq2Seq, n: int, stoi: dict[str, int]) -> tuple[list[int], list[list[float]]]:
    """Жадный ответ на число n и веса cross-attention последнего слоя (среднее по головам): [T_out, S]."""
    model.eval()
    src, _ = collate([encode_pair(n, stoi)])
    pred = greedy_decode(model, src, MAX_WORDS + 1)[0].tolist()
    pred = pred[: pred.index(EOS) + 1] if EOS in pred else pred
    layer = model.decoder.layers[-1].cross_attn
    captured = {}
    original = layer.forward

    def spy(x, context=None, mask=None, need_weights=False):
        out, w = original(x, context, mask, need_weights=True)
        captured["w"] = w
        return out

    layer.forward = spy
    try:
        with torch.no_grad():
            tgt_in = torch.tensor([[BOS] + pred[:-1]])
            model(src, tgt_in)
    finally:
        layer.forward = original
    return pred, captured["w"][0].mean(0).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="seq2seq: число цифрами → число словами")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--every", type=int, default=100, help="как часто считать exact match на валидации")
    ap.add_argument("--record", type=Path, help="записать прогон для визуализации (JSON)")
    args = ap.parse_args()

    itos, stoi = build_vocab()
    train_nums, val_nums = split_numbers(args.train_size)
    model = make_model(len(itos))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pairs = [encode_pair(n, stoi) for n in train_nums]
    g = torch.Generator().manual_seed(0)
    print(f"словарь {len(itos)}, параметров {sum(p.numel() for p in model.parameters()):,}, "
          f"обучение на {len(train_nums)} числах из {MAX_NUMBER + 1:,}")
    curve, t0, losses = [], time.time(), []
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(pairs), (args.batch_size,), generator=g).tolist()
        losses.append(train_step(model, opt, *collate([pairs[i] for i in idx])))
        if step % args.every == 0 or step == 1:
            em = exact_match(model, val_nums, stoi)
            curve.append({"step": step, "loss": round(sum(losses) / len(losses), 4), "val_em": round(em, 4)})
            losses = []
            print(f"шаг {step:5d}  loss {curve[-1]['loss']:.3f}  exact match (val) {em:.1%}  {time.time() - t0:.0f} с")

    words = lambda ids: " ".join(itos[i] for i in ids if i != EOS)
    errors = []
    for n in val_nums:
        pred, _ = cross_attention(model, n, stoi)
        if pred != encode_pair(n, stoi)[1]:
            errors.append({"n": n, "want": " ".join(number_to_words(n)), "got": words(pred)})
    print(f"ошибок на валидации: {len(errors)} из {len(val_nums)}")
    for e in errors[:10]:
        print(f"  {e['n']:>7}: {e['got']!r}  (надо {e['want']!r})")
    model.eval()
    for n in (7, 21000, 102345, 999999):
        print(f"  {n:>7} → {words(cross_attention(model, n, stoi)[0])}")

    if args.record:
        picks = [5, 21, 1001, 2022, 11000, 40712, 123456, 999999] + [e["n"] for e in errors[:4]]
        examples = []
        for n in picks:
            pred, attn = cross_attention(model, n, stoi)
            examples.append({"n": n, "src": list(str(n)), "out": [itos[i] for i in pred],
                             "want": number_to_words(n) + ["<eos>"],
                             "attn": [[round(v, 3) for v in row] for row in attn]})
        data = {"config": {"steps": args.steps, "train_size": args.train_size, "val_size": len(val_nums),
                           "d_model": 64, "n_layers": 2, "n_heads": 4, "lr": args.lr, "batch_size": args.batch_size},
                "curve": curve, "examples": examples, "errors": errors[:20], "val_errors": len(errors)}
        args.record.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"записано: {args.record}")


if __name__ == "__main__":
    main()
