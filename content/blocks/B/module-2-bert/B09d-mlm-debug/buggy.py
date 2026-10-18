"""MLM-маскирование BERT: 15 % токенов, из них 80 % → [MASK], 10 % → случайный токен, 10 % — без изменений.

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
    special = torch.isin(input_ids, torch.tensor(special_ids, device=input_ids.device))
    if word_ids is None:
        chosen = torch.rand(input_ids.shape, generator=generator) < prob
    else:
        n_words = int(word_ids.max()) + 1
        word_chosen = torch.rand(input_ids.shape[0], max(n_words, 1), generator=generator) < prob   # [B, W]
        chosen = word_chosen.gather(1, word_ids.clamp(min=0)) & (word_ids >= 0)
    labels = input_ids.masked_fill(~chosen, -100)
    chosen &= ~special
    roll = torch.rand(input_ids.shape, generator=generator)                       # одна «кость» на токен
    to_mask = chosen & (roll < 0.8)
    to_random = chosen & (roll >= 0.8) & (roll < 0.9)
    random_tokens = torch.randint(vocab_size, input_ids.shape, generator=generator)
    inputs = input_ids.masked_fill(to_mask, mask_id)
    inputs = torch.where(to_random, random_tokens, inputs)
    return inputs, labels
