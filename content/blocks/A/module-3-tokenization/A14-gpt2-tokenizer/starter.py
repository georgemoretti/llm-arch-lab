"""Свой токенизатор с настоящими merges GPT-2: чтение vocab.json и merges.txt, перевод id, encode и decode.

Реализуй load_gpt2_files, GPT2Tokenizer.encode и GPT2Tokenizer.decode.
Внутри работаем в своих id (байты 0..255, слияние k → 256 + k), наружу отдаём id GPT-2.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmlab.core.bpe import build_vocab, bytes_to_unicode, encode, pretokenize, split_special  # эталоны A11–A13


def load_gpt2_files(vocab_path: str | Path, merges_path: str | Path):
    """vocab.json + merges.txt → (merges в своих id, to_gpt2: свой id → id GPT-2, special: {строка: id GPT-2}).

    Строки токенов в файлах записаны через bytes_to_unicode (пробел = «Ġ»). Первая строка merges.txt —
    заголовок «#version: …», слияния идут по рангу: строка k (с нуля) создаёт свой токен 256 + k.
    """
    # TODO 1: vocab.json — словарь {строка токена: id GPT-2}; merges.txt — строки «a b», первая — заголовок.
    #         Пустые строки (обычно последняя) пропусти.
    # TODO 2: заведи str_to_ours: строка → свой id. Сначала 256 строк-байтов из bytes_to_unicode (символ → байт).
    # TODO 3: для k-й строки «a b»: merges[(свой id a, свой id b)] = 256 + k, и строка a + b получает свой id 256 + k.
    #         Почему строки a и b всегда уже есть в str_to_ours к этому моменту?
    # TODO 4: to_gpt2[свой id] = vocab.json[строка]. Спецтокены — строки vocab.json, которые не байты и не
    #         результаты слияний (у GPT-2 такой один).
    raise NotImplementedError("load_gpt2_files")


class GPT2Tokenizer:
    """Токенизатор GPT-2 из своих частей: пре-токенизация (A13), BPE по рангам (A12), перевод id."""

    def __init__(self, vocab_path: str | Path, merges_path: str | Path):
        # Готово, не меняй.
        self.merges, self.to_gpt2, self.special = load_gpt2_files(vocab_path, merges_path)
        self.from_gpt2 = {g: ours for ours, g in self.to_gpt2.items()}
        self.vocab = build_vocab(self.merges)                           # свой id → bytes

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Текст → id GPT-2. allowed_special=False: «<|endoftext|>» в тексте — обычные символы."""
        # TODO: как encode_with_special из A13 (split_special → pretokenize → encode каждого куска),
        #       только каждый свой id переведи в id GPT-2 через self.to_gpt2. Спецтокены уже в id GPT-2.
        raise NotImplementedError("GPT2Tokenizer.encode")

    def decode(self, ids: list[int]) -> str:
        """id GPT-2 → строка: байты всех токенов (спецтокены — их строкой), один decode UTF-8."""
        # TODO: id GPT-2 → свой id (self.from_gpt2) → байты (self.vocab); спецтокен → его строка в UTF-8.
        #       Склей всё и декодируй один раз с errors="replace" (A12, A13d).
        raise NotImplementedError("GPT2Tokenizer.decode")
