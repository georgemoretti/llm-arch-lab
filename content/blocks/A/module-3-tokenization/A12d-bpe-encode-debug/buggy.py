"""BPE: кодирование строки по рангам слияний и декодирование через байты.

merges и vocab — как в A11: {(a, b): 256 + k} и {id: bytes}.
"""

from __future__ import annotations

from llmlab.core.bpe import merge_pair   # готовое слияние пары (эталон задачи «BPE: обучение»)


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Строка → id: байты UTF-8, затем слияния по рангу — каждый раз пара с наименьшим id из merges."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # пары, которые токенизатор умеет сливать, и сколько раз каждая встречается
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(ids, ids[1:]):
            if pair in merges:
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        pair = max(counts, key=counts.get)
        ids = merge_pair(ids, pair, merges[pair])
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """id → строка: сначала склеить байты всех токенов, потом один раз декодировать UTF-8."""
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
