## Подсказка 1

`LatentCache.update` — копия `KVCache.update` из C05 с `dim=1` (латенты `[B, S, r]` без оси голов).
В `forward`: `c = self.kv_a_layernorm(self.kv_a_proj(x))`, затем `if cache is not None: c, _ = cache.update(c)`.
Кешируется **c**, а не `kv_b_proj(c)`.

## Подсказка 2

`kv = self.kv_b_proj(c).view(B, S, H, d + dv).transpose(1, 2)`, `k, v = kv.split([d, dv], dim=-1)`.
Дальше как в C05: `scores = q @ k.transpose(-2, -1) / sqrt(d)`, маска
`torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)`, softmax, `@ v`,
`transpose(1, 2).reshape(B, T, H * dv)`, `o_proj`.

## Подсказка 3

Поглощение: `W = self.kv_b_proj.weight.view(H, d + dv, r)`, `W_uk, W_uv = W[:, :d], W[:, d:]`.
`q_lat = torch.einsum("bhtd,hdr->bhtr", q, W_uk)`, `scores = q_lat @ c.unsqueeze(1).transpose(-2, -1) / sqrt(d)`
(латенты общие для всех голов — `unsqueeze(1)` и broadcasting), маска, softmax,
`o_lat = weights @ c.unsqueeze(1)`, `out = torch.einsum("bhtr,hvr->bhtv", o_lat, W_uv)`.
