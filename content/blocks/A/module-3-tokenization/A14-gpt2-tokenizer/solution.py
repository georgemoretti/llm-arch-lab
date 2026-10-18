"""Свой токенизатор с настоящими merges GPT-2: чтение vocab.json и merges.txt, перевод id, encode и decode.

Эталон. Внутри работаем в своих id (байты 0..255, слияние k → 256 + k), наружу отдаём id GPT-2.
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
    gpt2_vocab: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    lines = Path(merges_path).read_text(encoding="utf-8").splitlines()
    rules = [line for line in lines if line and not line.startswith("#version")]

    str_to_ours = {ch: b for b, ch in bytes_to_unicode().items()}      # «Ġ» → 32, «a» → 97, …
    merges: dict[tuple[int, int], int] = {}
    for k, line in enumerate(rules):
        a, b = line.split(" ")
        merges[(str_to_ours[a], str_to_ours[b])] = 256 + k
        str_to_ours[a + b] = 256 + k                                    # склеенный токен — тоже строка

    to_gpt2 = {ours: gpt2_vocab[s] for s, ours in str_to_ours.items()}
    special = {s: i for s, i in gpt2_vocab.items() if s not in str_to_ours}   # <|endoftext|>
    return merges, to_gpt2, special


class GPT2Tokenizer:
    """Токенизатор GPT-2 из своих частей: пре-токенизация (A13), BPE по рангам (A12), перевод id."""

    def __init__(self, vocab_path: str | Path, merges_path: str | Path):
        # Готово, не меняй.
        self.merges, self.to_gpt2, self.special = load_gpt2_files(vocab_path, merges_path)
        self.from_gpt2 = {g: ours for ours, g in self.to_gpt2.items()}
        self.vocab = build_vocab(self.merges)                           # свой id → bytes

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Текст → id GPT-2. allowed_special=False: «<|endoftext|>» в тексте — обычные символы."""
        special = self.special if allowed_special else {}
        ids: list[int] = []
        for part in split_special(text, special):
            if part in special:
                ids.append(special[part])
            else:
                for chunk in pretokenize(part):
                    ids.extend(self.to_gpt2[i] for i in encode(chunk, self.merges))
        return ids

    def decode(self, ids: list[int]) -> str:
        """id GPT-2 → строка: байты всех токенов (спецтокены — их строкой), один decode UTF-8."""
        inverse = {i: s.encode("utf-8") for s, i in self.special.items()}
        data = b"".join(inverse[i] if i in inverse else self.vocab[self.from_gpt2[i]] for i in ids)
        return data.decode("utf-8", errors="replace")
