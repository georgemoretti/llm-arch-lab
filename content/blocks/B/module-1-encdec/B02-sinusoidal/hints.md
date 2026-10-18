## Подсказка 1

```python
pos = torch.arange(n_pos, dtype=torch.float32)[:, None]                 # [n_pos, 1]
two_i = torch.arange(0, d_model, 2, dtype=torch.float32)               # 0, 2, 4, …
omega = 10000.0 ** (-two_i / d_model)                                  # [d_model / 2]
```

`pos * omega` по broadcasting даёт углы `[n_pos, d_model / 2]`. Срезы `pe[:, 0::2]` и `pe[:, 1::2]` — чётные и
нечётные столбцы.

## Подсказка 2

Блок $2\times2$ пары $i$ стоит на пересечении строк и столбцов $2i, 2i+1$. Если `c = cos(k·ω)` и `s = sin(k·ω)` —
векторы длины $C/2$, то все блоки сразу заполняются четырьмя срезами:
`m[0::2, 0::2]`, `m[0::2, 1::2]`, `m[1::2, 0::2]`, `m[1::2, 1::2]` — каждый равен `torch.diag(...)`.

## Подсказка 3

```python
m[0::2, 0::2] = torch.diag(c)
m[0::2, 1::2] = torch.diag(s)
m[1::2, 0::2] = torch.diag(-s)
m[1::2, 1::2] = torch.diag(c)
```

В `forward`: `T = ids.shape[1]`, проверка длины, затем `self.tok(ids) * math.sqrt(self.d_model) + self.pe[:T]`.
