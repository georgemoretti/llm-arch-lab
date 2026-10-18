## Подсказка 1

Положи в `cache` всё, что пригодится обратному проходу: `x̂`, `rstd = 1/sqrt(σ² + eps)` формы
`[..., 1]` и `γ`. Тогда backward не пересчитывает статистики, и `eps` используется тот же.
`dgamma` и `dbeta` суммируются по всем осям, кроме последней: `dim=tuple(range(dy.dim() - 1))`.

## Подсказка 2

Выведи производную по шагам: $\partial\mu/\partial x_j = 1/C$,
$\partial\sigma^2/\partial x_j = 2(x_j - \mu)/C$, $\partial r/\partial\sigma^2 = -r^3/2$.
Подставь в $\hat x_i = (x_i - \mu)\,r$. Получится
$\partial\hat x_i/\partial x_j = r(\delta_{ij} - 1/C - \hat x_i\hat x_j/C)$.

## Подсказка 3

Свёртка с `g = dy * gamma` по индексу `i`:
`dx = rstd * (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True))`.
Проверка: `dx.sum(-1)` должен быть ≈ 0.
