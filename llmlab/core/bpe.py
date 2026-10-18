"""Byte-level BPE — эталоны задач A11–A13 (блок A, модуль 3 «Токенизация»).

Задачи A12–A14 импортируют готовые части отсюда, а не из кода пользователя: ошибка в A11 не должна ломать
следующие задачи. Совпадение функций с ``solution.py`` задач проверяет ``tests/test_core_bpe.py``.

Соглашение об id: токены 0..255 — отдельные байты, слияние номер i (с нуля) создаёт токен 256 + i.
Поэтому id нового токена — это и его ранг: чем меньше id, тем раньше слияние выучено.
"""

from __future__ import annotations

import regex

# ---------------------------------------------------------------- A11: обучение


def count_pairs(seqs: list[list[int]]) -> dict[tuple[int, int], int]:
    """Сколько раз каждая пара соседних id встречается внутри последовательностей (не через их границы)."""
    counts: dict[tuple[int, int], int] = {}
    for seq in seqs:
        for pair in zip(seq, seq[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Новый список, в котором каждое вхождение pair слева направо без перекрытий заменено на new_id."""
    out = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(seqs: list[list[int]], vocab_size: int) -> dict[tuple[int, int], int]:
    """Обучение BPE: vocab_size − 256 раз слить самую частую пару. Возвращает merges {пара: id нового токена}.

    При равных частотах берётся меньшая пара (сравнение кортежей), чтобы результат не зависел от порядка
    обхода словаря. Обучение останавливается раньше, если пар не осталось.
    """
    if vocab_size < 256:
        raise ValueError(f"vocab_size >= 256: первые 256 токенов — байты, а передано {vocab_size}")
    seqs = [list(s) for s in seqs]
    merges: dict[tuple[int, int], int] = {}
    for new_id in range(256, vocab_size):
        counts = count_pairs(seqs)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        merges[best] = new_id
        seqs = [merge_pair(s, best, new_id) for s in seqs]
    return merges


def build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    """id → байты токена: 256 одиночных байтов, затем склейки в порядке обучения."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


# ---------------------------------------------------------------- A12: encode / decode


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Строка → id: байты UTF-8, затем слияния по рангу — каждый раз пара с наименьшим id из merges."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pair = min(set(zip(ids, ids[1:])), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge_pair(ids, pair, merges[pair])
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """id → строка: сначала склеить байты всех токенов, потом один раз декодировать UTF-8."""
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- A13: пре-токенизация, спецтокены, byte ↔ unicode

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str) -> list[str]:
    """Нарезка текста на куски регуляркой GPT-2; слияния BPE не выходят за границы кусков."""
    return regex.findall(GPT2_PATTERN, text)


def bytes_to_unicode() -> dict[int, str]:
    """Обратимое отображение 256 байтов в печатные символы Unicode (как в vocab.json и merges.txt GPT-2).

    Видимые байты (``!``..``~``, ``¡``..``¬``, ``®``..``ÿ``) отображаются в символ с тем же кодом, остальные
    68 (управляющие, пробел, DEL, NBSP, мягкий перенос) по порядку — в символы 256, 257, …
    """
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    table = {b: chr(b) for b in visible}
    shift = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + shift)
            shift += 1
    return table


def train_regex_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    """BPE с пре-токенизацией: пары считаются только внутри кусков pretokenize(text)."""
    return train_bpe([list(chunk.encode("utf-8")) for chunk in pretokenize(text)], vocab_size)


def split_special(text: str, special: dict[str, int]) -> list[str]:
    """Разрезать текст по спецтокенам, сохранив их отдельными элементами; пустые куски выбросить."""
    if not special:
        return [text] if text else []
    # длинные спецтокены — первыми, чтобы `<|a|>` не съел начало `<|a|b|>`
    pattern = "(" + "|".join(regex.escape(s) for s in sorted(special, key=len, reverse=True)) + ")"
    return [part for part in regex.split(pattern, text) if part]


def encode_with_special(text: str, merges: dict[tuple[int, int], int], special: dict[str, int]) -> list[int]:
    """Спецтокены → свои id целиком; остальной текст → pretokenize → BPE каждого куска отдельно."""
    ids: list[int] = []
    for part in split_special(text, special):
        if part in special:
            ids.append(special[part])
        else:
            for chunk in pretokenize(part):
                ids.extend(encode(chunk, merges))
    return ids


def decode_with_special(ids: list[int], vocab: dict[int, bytes], special: dict[str, int]) -> str:
    """id (в том числе спецтокены) → строка; UTF-8 декодируется один раз для всей последовательности байтов."""
    inverse = {i: s.encode("utf-8") for s, i in special.items()}
    return b"".join(inverse[i] if i in inverse else vocab[i] for i in ids).decode("utf-8", errors="replace")
