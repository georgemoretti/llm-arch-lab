## Подсказка 1

`mlp_layer_types` — списковое включение: `"moe" if i >= first_k_dense_replace and i % moe_layer_freq == 0 else "dense"`.
`forward`: `out, logits = self.routed(x)`; если `self.shared_experts is not None` —
`out = out + self.shared_experts(x)`. Shared expert не умножается ни на какой вес роутера.

## Подсказка 2

`build_ffn_layers`: `nn.ModuleList(DeepSeekMoE(d_model, d_expert, n_routed, n_shared, top_k) if t == "moe"
else SwiGLUFFN(d_model, d_dense) for t in mlp_layer_types(...))`.

## Подсказка 3

Параметры SwiGLU шириной $F$ — `3 * d_model * F` (gate, up, down без bias). Роутер — `d_model * n_routed`.
Число MoE-слоёв — `mlp_layer_types(...).count("moe")`. Всего: `dense_слои * 3CF_d + moe_слои *
((n_routed + n_shared) * 3CF_e + C·E)`; активно — то же с `top_k` вместо `n_routed` в скобках
(роутер при этом считается целиком: он вычисляется для каждого токена).
