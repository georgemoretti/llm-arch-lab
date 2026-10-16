## Подсказка 1

Начни с `update`. При первом вызове заведи `self.k = k.new_zeros(B, H_kv, W, D)` (и `self.v`), и
`self.pos = torch.full((W,), -1, dtype=torch.long)`. Позиции новых токенов:
`new_pos = torch.arange(self.seq_len, self.seq_len + T)`. Сначала собери ответ —
`torch.cat((self.k, k), dim=2)`, то же для `v` и `torch.cat((self.pos, new_pos))`, — и только потом пиши в буфер.

## Подсказка 2

Запись: `keep = min(T, W)`, `slots = new_pos[-keep:] % W`, затем
`self.k[:, :, slots] = k[:, :, -keep:]`, то же для `v`, `self.pos[slots] = new_pos[-keep:]`,
`self.seq_len += T`. Последние `W` новых позиций дают разные ячейки, так что конфликтов нет.

## Подсказка 3

В `forward`: `start = cache.seq_len` (до `update`!), `q_pos = arange(start, start + T)`, RoPE по `q_pos`
для q и k. С кешем — `k, v, k_pos = cache.update(k, v)`, без кеша — `k_pos = q_pos`. Маска:
`dist = q_pos[:, None] - k_pos[None, :]`, `mask = (k_pos[None, :] >= 0) & (dist >= 0) & (dist < W)`.
Для `swa_kv_cache_bytes`: `tokens = sum(min(seq_len, window) if t == "sliding_attention" else seq_len ...)`,
ответ `2 * tokens * n_kv_heads * head_dim * batch * bytes_per_elem`.
