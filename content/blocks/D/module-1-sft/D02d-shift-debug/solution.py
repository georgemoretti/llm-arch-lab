"""Маска лосса SFT: метки только у токенов ответа, сдвиг на один токен, упаковка диалогов в одну строку.

Эталон. Соглашение: ``targets[t]`` — токен, который модель должна предсказать на позиции ``t`` (``ids[t + 1]``),
или ``IGNORE = -100`` (так ``F.cross_entropy(..., ignore_index=-100)`` пропускает позицию).
"""

from __future__ import annotations

import torch

IGNORE = -100


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
