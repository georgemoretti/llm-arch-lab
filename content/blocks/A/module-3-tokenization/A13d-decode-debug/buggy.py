"""Пре-токенизация в стиле GPT-2, спецтокены и отображение byte ↔ unicode.

merges и vocab — как в A11: {(a, b): 256 + k} и {id: bytes}; special — {строка: id}.
"""

from __future__ import annotations

import regex

from llmlab.core.bpe import encode, train_bpe   # готовые encode (A12) и обучение (A11)

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
    # escape обязателен: в `<|endoftext|>` символ | — это «или» регулярки; скобки сохраняют разделители
    pattern = "(" + "|".join(regex.escape(s) for s in special) + ")"
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
    inverse = {i: s for s, i in special.items()}
    return "".join(inverse[i] if i in inverse else vocab[i].decode("utf-8", errors="replace") for i in ids)
