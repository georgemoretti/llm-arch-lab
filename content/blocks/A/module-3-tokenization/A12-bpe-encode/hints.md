## Подсказка 1

Ранг пары — её id в `merges`. Для пар, которых нет в `merges`, удобно считать ранг бесконечным:
`merges.get(pair, float("inf"))`. Тогда «пара с наименьшим рангом» — это просто `min` по соседним парам
с таким ключом.

## Подсказка 2

Если у найденной пары ранг бесконечный, значит, ни одной пары из `merges` в последовательности не
осталось: цикл заканчивается. Иначе `ids = merge_pair(ids, pair, merges[pair])` и следующий круг.
Строки короче двух байтов в цикл не попадают.

## Подсказка 3

```python
ids = list(text.encode("utf-8"))
while len(ids) >= 2:
    pair = min(set(zip(ids, ids[1:])), key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
        break
    ids = merge_pair(ids, pair, merges[pair])
```

Decode: `b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")`.
