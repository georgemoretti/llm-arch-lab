## Подсказка 1

Сначала реши, в какой половине бакет, и посчитай неотрицательное расстояние `n`:
```python
buckets = torch.zeros_like(relative_position)
if bidirectional:
    num_buckets //= 2
    buckets = buckets + (relative_position > 0).long() * num_buckets
    n = relative_position.abs()
else:
    n = (-relative_position).clamp(min=0)
```
Дальше обе ветки одинаковы: работаешь с `n` и половиной `num_buckets`.

## Подсказка 2

Точная часть и логарифмическая считаются для всех элементов, а выбирает между ними `torch.where`:
```python
max_exact = num_buckets // 2
is_small = n < max_exact
log_ratio = torch.log(n.clamp(min=1).float() / max_exact) / math.log(max_distance / max_exact)
large = (max_exact + (log_ratio * (num_buckets - max_exact)).long()).clamp(max=num_buckets - 1)
return buckets + torch.where(is_small, n, large)
```
`.long()` у положительного числа — округление вниз, как в T5.

## Подсказка 3

`relative_position_bias`:
```python
q_pos = torch.arange(query_length, device=dev)[:, None] + offset
k_pos = torch.arange(key_length, device=dev)[None, :]
b = relative_position_bucket(k_pos - q_pos, bidirectional, num_buckets, max_distance)   # [T, S]
return bias(b).permute(2, 0, 1).unsqueeze(0)                                            # [1, H, T, S]
```
Порядок вычитания важен: ключ минус запрос. Ключ слева — отрицательный сдвиг.
