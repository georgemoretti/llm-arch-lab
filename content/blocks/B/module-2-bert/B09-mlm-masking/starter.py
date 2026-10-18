"""MLM-маскирование BERT: 15 % токенов, из них 80 % → [MASK], 10 % → случайный токен, 10 % — без изменений.

Реализуй mlm_mask. Генератор случайных чисел передаётся снаружи: результат должен быть воспроизводимым.
"""

from __future__ import annotations

import torch


def mlm_mask(input_ids: torch.Tensor, special_ids: list[int], mask_id: int, vocab_size: int, prob: float = 0.15,
             word_ids: torch.Tensor | None = None, generator: torch.Generator | None = None):
    """Маскирование для MLM → (inputs, labels).

    Выбирается доля prob обычных токенов (спецтокены и PAD не выбираются никогда). При word_ids [B, T] (номер слова
    токена, −1 у спецтокенов) выбираются целые слова: все их подтокены сразу. Из выбранных токенов 80 % → [MASK],
    10 % → случайный токен словаря, 10 % остаются. labels = исходный токен у выбранных, −100 у остальных.
    """
    # TODO 1: маска спецтокенов (special_ids: [CLS], [SEP], [PAD] …) — torch.isin.
    # TODO 2: выбор: без word_ids — каждый токен независимо с вероятностью prob (torch.rand(…, generator=generator)).
    #         С word_ids — бросок на каждое СЛОВО [B, W], затем раздать решение всем его подтокенам (gather);
    #         у позиций с word_ids = −1 (спецтокены) выбора нет.
    # TODO 3: спецтокены не выбираются никогда. labels: исходный токен у выбранных, −100 у остальных.
    # TODO 4: у выбранных: 80 % → mask_id, 10 % → случайный id из [0, vocab_size), 10 % — как было.
    #         Одно равномерное число на токен решает, в какую из трёх долей он попал.
    # TODO 5: вход не менять на месте; вернуть (inputs, labels).
    raise NotImplementedError("mlm_mask")
