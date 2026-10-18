## Подсказка 1

`ids != pad_id` даёт bool `[B, S]`. Индексирование `m[:, None, None, :]` добавляет две оси размера 1 →
`[B, 1, 1, S]`. При `masked_fill` по оценкам `[B, H, T, S]` маска растянется по головам и запросам.

В `attention` транспонируй только две последние оси ключей: `k.transpose(-2, -1)` имеет форму `[B, H, D, S]`,
поэтому `q @ k.transpose(-2, -1)` даёт `[B, H, T, S]` даже при `T ≠ S`.

## Подсказка 2

`split_heads`: сначала `x.view(B, L, H, C // H)` — каждая позиция режет свой вектор на $H$ кусков, затем
`.transpose(1, 2)` → `[B, H, L, D]`. Сразу `view(B, H, L, D)` форму даст ту же, но перемешает позиции между
головами.

Обратно: `out.transpose(1, 2)` → `[B, T, H, D]`, затем `.reshape(B, T, H * D)`. После `transpose` тензор не
непрерывен в памяти, поэтому `reshape`, а не `view`.

## Подсказка 3

```python
context = x if context is None else context
q = self.split_heads(self.q_proj(x))
k = self.split_heads(self.k_proj(context))
v = self.split_heads(self.v_proj(context))
out, weights = attention(q, k, v, mask)
B, H, T, D = out.shape
out = self.out_proj(out.transpose(1, 2).reshape(B, T, H * D))
return (out, weights) if need_weights else out
```
