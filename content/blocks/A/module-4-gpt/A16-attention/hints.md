## Подсказка 1

`torch.tril(torch.ones(T, T, dtype=torch.bool))` — нижний треугольник с диагональю. В `attention`
транспонируй только две последние оси ключей: `k.transpose(-2, -1)` имеет форму `[B, D, T]`, и
`q @ k.transpose(-2, -1)` даёт `[B, T, T]`.

## Подсказка 2

`scores.masked_fill(~mask, float("-inf"))` ставит $-\infty$ туда, где маска `False`. Маска `[T, T]` сама
растянется по батчу. Softmax по последней оси: `F.softmax(scores, dim=-1)`.

## Подсказка 3

```python
D = q.shape[-1]
scores = q @ k.transpose(-2, -1) / math.sqrt(D)
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
return weights @ v, weights
```

В голове: `T = x.shape[1]`, затем `attention(self.query(x), self.key(x), self.value(x), self.mask[:T, :T])[0]`.
