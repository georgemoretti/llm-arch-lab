## Подсказка 1

Начни со скана — он прямо повторяет формулу. Следи за осями: состояние $h$ — $[b, H, N, P]$; $B_t$ и $C_t$ имеют
только ось $N$ (общие для голов), $x_t$ — оси $H, P$, а $\Delta_t$ и $e^{\Delta_t A}$ — ось $H$. Всё склеивается
броадкастингом через `[..., None]`.

## Подсказка 2

`segsum`: `cs = torch.cumsum(a, -1)`, `out = cs[..., :, None] - cs[..., None, :]` — это $a_{s+1} + \dots + a_t$ для $s \le t$.
Затем `masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))`.

## Подсказка 3

Квадратичная форма: `L = torch.exp(segsum((dt * A).transpose(1, 2)))` → $[b, H, T, T]$;
`M = L * (C @ B.transpose(1, 2))[:, None]`; `y = torch.einsum("bhts,bshp->bthp", M, dt[..., None] * x) + D[:, None] * x`.
