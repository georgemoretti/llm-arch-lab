## Подсказка 1

`W` имеет форму `[fan_out, fan_in]`: `fan_out, fan_in = W.shape`. Kaiming зависит только от
`fan_in`, Xavier — от суммы. `gain` для ReLU получается из того, что ReLU вдвое уменьшает $\mathbb{E}[x^2]$,
а gain умножает **std**, а не дисперсию.

## Подсказка 2

Заполнение на месте без графа:

```python
with torch.no_grad():
    return W.normal_(0.0, std, generator=generator)
```

Для Xavier — `W.uniform_(-a, a, generator=generator)`.

## Подсказка 3

Эксперимент: `g = torch.Generator().manual_seed(seed)`, сначала `x = torch.randn(n, width, generator=g)`, потом
в цикле `W = torch.empty(width, width)`, заполнение по `init` тем же `g` (для `"xavier"` — gain 1, для
`"kaiming"` — `nonlinearity=act`), `x = f(x @ W.T)`, `stds.append(x.std().item())`.
