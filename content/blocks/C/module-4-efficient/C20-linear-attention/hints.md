## Подсказка 1

Всё строится из двух сумм по ключам: $\sum \varphi(k_s) v_s^\top$ (матрица $[d, d_v]$) и $\sum \varphi(k_s)$
(вектор $[d]$). Некаузальная форма берёт их по всем $s$, causal — по $s \le t$. Числитель и знаменатель
всегда суммируются по одному и тому же набору ключей.

## Подсказка 2

Некаузальная: `kv = k.transpose(-2, -1) @ v`, `z = k.sum(dim=-2)`, `out = (q @ kv) / (q @ z.unsqueeze(-1) + eps)`
(после `q, k = feature_map(q), feature_map(k)`). Параллельная causal: `A = (q @ k.transpose(-2, -1)).tril()`,
`out = (A @ v) / (A.sum(-1, keepdim=True) + eps)`.

## Подсказка 3

Шаг рекуррентной формы:

```python
S = S + k[:, :, t, :, None] * v[:, :, t, None, :]      # [B, H, d, dv]
z = z + k[:, :, t]                                     # [B, H, d]
num = (q[:, :, t, :, None] * S).sum(dim=-2)            # [B, H, dv]
den = (q[:, :, t] * z).sum(dim=-1, keepdim=True)       # [B, H, 1]
```

Сначала обновление, потом чтение: токен $t$ учитывает и себя.
