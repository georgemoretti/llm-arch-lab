"""WordPiece-токенизатор BERT: очистка текста, пре-токенизация, жадный longest-match-first с «##».

Реализуй strip_accents, basic_tokenize, wordpiece и tokenize. is_punctuation и clean_text готовы.
Китайские иероглифы (BERT окружает их пробелами) не обрабатываются — в тестах их нет.
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
    # TODO: разложи строку в NFD (буква + отдельные комбинируемые знаки) и выкинь символы категории Mn.
    #       unicodedata.normalize и unicodedata.category.
    raise NotImplementedError("strip_accents")


def basic_tokenize(text: str, lower: bool = True) -> list[str]:
    """Пре-токенизация BERT: очистка, (lower + strip_accents), разрез по пробелам, каждый знак пунктуации — отдельно."""
    # TODO 1: clean_text; если lower — нижний регистр и strip_accents.
    # TODO 2: разрежь по пробельным символам (str.split()).
    # TODO 3: внутри каждого куска каждый знак пунктуации (is_punctuation) — отдельное слово, а буквы и цифры между
    #         ними склеиваются: "don't!" → ["don", "'", "t", "!"].
    raise NotImplementedError("basic_tokenize")


def wordpiece(word: str, vocab: dict[str, int], unk: str = "[UNK]", max_chars: int = 100) -> list[str]:
    """Жадный longest-match-first: самый длинный префикс из словаря, продолжения — с «##». Не разобралось — [unk]."""
    # TODO 1: слово длиннее max_chars → [unk].
    # TODO 2: с текущей позиции ищи САМЫЙ ДЛИННЫЙ кусок, который есть в словаре; у всех кусков, кроме первого,
    #         в словаре ищется вариант с префиксом "##".
    # TODO 3: если с какой-то позиции не нашлось ни одного куска — всё слово превращается в [unk] (не только хвост).
    raise NotImplementedError("wordpiece")


def tokenize(text: str, vocab: dict[str, int], lower: bool = True) -> list[str]:
    """Текст → WordPiece-токены: basic_tokenize, затем wordpiece для каждого слова."""
    # TODO: basic_tokenize, затем wordpiece для каждого слова; результат — плоский список.
    raise NotImplementedError("tokenize")
