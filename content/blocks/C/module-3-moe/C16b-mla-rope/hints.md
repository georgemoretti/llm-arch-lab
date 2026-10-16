## Подсказка 1

Позиции и запрос: `start = cache.seq_len if cache is not None else 0`,
`cos, sin = rope_cos_sin(torch.arange(start, start + T), dr, self.rope_theta)`.
`q = self.q_proj(x).view(B, T, H, dn + dr).transpose(1, 2)`, `q_nope, q_rope = q.split([dn, dr], dim=-1)`,
`q_rope = apply_rope(q_rope, cos, sin)`.

## Подсказка 2

`c, k_rope = self.kv_a_proj_with_mqa(x).split([r, dr], dim=-1)`, `c = self.kv_a_layernorm(c)`.
`apply_rope` ждёт `[..., T, dr]` с осью голов перед временем: `apply_rope(k_rope.unsqueeze(1), cos, sin).squeeze(1)`.
Затем `if cache is not None: c, k_rope = cache.update(c, k_rope)`.

## Подсказка 3

`kv = self.kv_b_proj(c).view(B, S, H, dn + dv).transpose(1, 2)`, `k_nope, v = kv.split([dn, dv], -1)`,
`k = torch.cat((k_nope, k_rope.unsqueeze(1).expand(B, H, S, dr)), dim=-1)`,
`q = torch.cat((q_nope, q_rope), dim=-1)`, `scores = q @ k.transpose(-2, -1) / math.sqrt(dn + dr)`,
маска `causal_mask(T, S, x.device)`. Память: `n_layers * (kv_lora_rank + qk_rope_head_dim) * seq_len * batch * bytes_per_elem`.
