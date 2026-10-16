## Подсказка 1

Сначала `kv_cache_bytes` — это просто произведение: для каждого слоя, каждой KV-головы, каждого
токена и каждого примера в батче хранится вектор длины `head_dim` — и для K, и для V.
Потом `repeat_kv`: тебе нужна операция, повторяющая каждую голову `n_rep` раз **подряд**.

## Подсказка 2

`torch.repeat_interleave(x, n_rep, dim=1)` повторяет подряд (`[a, a, b, b]`), а
`x.repeat(1, n_rep, 1, 1)` — по кругу (`[a, b, a, b]`). Нужен первый вариант. Без копий до
последнего момента: `x[:, :, None].expand(B, H_kv, n_rep, T, D).reshape(B, H_kv * n_rep, T, D)` —
так делает HF.

## Подсказка 3

`forward` — это C00 с двумя отличиями: у k и v в `view` стоит `self.n_kv_heads`, а не
`self.n_heads`, и перед `q @ k.transpose(-2, -1)` их надо пропустить через
`repeat_kv(·, self.n_rep)`. В конце: `out.transpose(1, 2).reshape(B, T, H * D)` → `self.o_proj`.
