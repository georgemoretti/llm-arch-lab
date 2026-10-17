## Подсказка 1

Все три ручки работают с **логитами**, а не с вероятностями: отсечённым токенам ставят $-\infty$, и один
`softmax` в конце сам перенормирует оставшиеся. Порядок: `/ T` → top-k → top-p → `multinomial`.

## Подсказка 2

top-k: k-й по величине логит — `logits.topk(k, dim=-1).values[:, -1:]` (форма `[B, 1]`, чтобы сравнение
транслировалось по словарю); всё, что меньше, — `masked_fill(..., -inf)`. Если `top_k` больше словаря,
`topk` упадёт — возьми `min(top_k, V)`.

## Подсказка 3

top-p: `sorted_logits, order = logits.sort(dim=-1, descending=True)`; `probs = sorted_logits.softmax(-1)`;
`before = probs.cumsum(-1) - probs` — масса токенов **выше** текущего; `sorted_logits[before >= top_p] = -inf`
(через `masked_fill`); вернуть на места словаря:
`torch.full_like(logits, -inf).scatter(-1, order, sorted_logits)`. Маска `cumsum >= p` (без сдвига) выкинула
бы токен, который пересёк порог.
