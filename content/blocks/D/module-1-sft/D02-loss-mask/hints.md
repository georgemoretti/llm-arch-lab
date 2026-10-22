## Подсказка 1

Поиск заголовка — обычный проход окном:
```python
ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
h, i = len(header), 0
while i < len(ids):
    if ids[i:i + h] == header:
        j = i + h                      # первый токен ответа
        ...                            # True до end_id включительно
        i = j + 1
    else:
        i += 1
```

## Подсказка 2

Сдвиг — два среза:
```python
targets = torch.full_like(ids, IGNORE)
targets[:-1] = torch.where(mask[1:], ids[1:], IGNORE)
```
Срез `[1:]` берёт «следующий» токен для каждой позиции `0 … T − 2`. Маску сдвигай так же: метка позиции `t` зависит
от того, входит ли в ответ токен `t + 1`, а не сам токен `t`.

## Подсказка 3

```python
lens = [len(x) for x, _ in cur]
position_ids = torch.cat([torch.arange(n) for n in lens])
cu_seqlens = torch.tensor([0] + lens).cumsum(0).to(torch.int32)

seg = torch.searchsorted(cu[1:].long(), torch.arange(L), right=True)    # номер примера
mask = torch.ones(L, L, dtype=torch.bool).tril() & (seg[:, None] == seg[None, :])
```
