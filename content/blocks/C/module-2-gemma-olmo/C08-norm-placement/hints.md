## Подсказка 1

Каждый `forward` — две строки, одна на подслой. Начни с Pre-Norm: он уже был в C06 —
`x = x + self.self_attn(self.input_layernorm(x))`, затем то же для FFN с `post_attention_layernorm`.
Остальные три — перестановки тех же трёх операций: норма, подслой, `+ x`.

## Подсказка 2

Спроси себя для каждого варианта: что нормируется — вход подслоя, выход подслоя или сумма с
остатком? Post-Norm: сумма. OLMo 2: выход. Sandwich: вход и выход. Если в формуле остаток
проходит через норму, блок уже не может быть тождеством при нулевых подслоях.

## Подсказка 3

```
Post:     x = n_a(x + attn(x));            x = n_f(x + mlp(x))
OLMo 2:   x = x + n_a(attn(x));            x = x + n_f(mlp(x))
Sandwich: x = x + n_pa(attn(n_in(x)));     x = x + n_pf(mlp(n_pre_ff(x)))
```

где `n_a = post_attention_layernorm`, `n_f = post_feedforward_layernorm`, `n_in = input_layernorm`,
`n_pre_ff = pre_feedforward_layernorm`.
