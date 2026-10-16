## Подсказка 1

Состояние $S$ — это матрица $[d_k, d_v]$: «по ключу $k$ память отдаёт $S^\top k$». Каждый шаг: (забыть) →
посмотреть, что память отдаёт по новому ключу → поправить разницу → прочитать выход для запроса.
Отличие от C20 — одна строка: пишется не $v_t$, а $\beta_t(v_t - S^\top k_t)$.

## Подсказка 2

Без einsum: `v_old = (S * k_t[..., None]).sum(dim=-2)` (сумма по оси $d_k$), запись —
`S = S + k_t[..., None] * delta[..., None, :]`, выход — `(S * q_t[..., None]).sum(dim=-2)`.
Гейт: `S = S * g[:, :, t].exp()[..., None, None]` — **до** чтения `v_old`.

## Подсказка 3

В слое: `beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)`,
`g = (-self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)).transpose(1, 2)`;
`q = l2norm(self.q_proj(x).view(B, T, H, dk)).transpose(1, 2)` (так же `k`), `v` — без нормы.
