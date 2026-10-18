## Подсказка 1

```python
special = torch.isin(input_ids, torch.tensor(special_ids))
chosen = torch.rand(input_ids.shape, generator=generator) < prob
chosen &= ~special
labels = input_ids.masked_fill(~chosen, -100)
```

## Подсказка 2

Одно число на токен делит выбранные на три доли: `roll = torch.rand(input_ids.shape, generator=generator)`;
`to_mask = chosen & (roll < 0.8)`, `to_random = chosen & (roll >= 0.8) & (roll < 0.9)`. Случайные токены:
`torch.randint(vocab_size, input_ids.shape, generator=generator)`. Собери вход через `masked_fill` и `torch.where` —
они возвращают новый тензор.

## Подсказка 3

Whole word masking: бросок на каждое слово `torch.rand(B, n_words, generator=generator) < prob`, где
`n_words = int(word_ids.max()) + 1`. Раздать подтокенам: `word_chosen.gather(1, word_ids.clamp(min=0))`, затем
`& (word_ids >= 0)` — у спецтокенов слова нет.
