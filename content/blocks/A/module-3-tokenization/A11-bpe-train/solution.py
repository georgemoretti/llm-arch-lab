"""Обучение byte-level BPE: подсчёт пар, слияние, список merges с рангами, словарь «id → байты».

Эталон. Токены 0..255 — байты, слияние номер i создаёт токен 256 + i.
"""

from __future__ import annotations


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
