"""WordPiece-токенизатор BERT: очистка текста, пре-токенизация, жадный longest-match-first с «##».

Эталон. Китайские иероглифы (BERT окружает их пробелами) не обрабатываются — в тестах их нет.
"""

from __future__ import annotations

import unicodedata


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
