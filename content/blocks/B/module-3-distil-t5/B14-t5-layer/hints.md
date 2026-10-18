## Подсказка 1

`T5LayerNorm.forward` — две строки:
```python
variance = x.float().pow(2).mean(-1, keepdim=True)
return self.weight * (x.float() * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)
```

## Подсказка 2

`T5Attention.forward`:
```python
B, T, _ = x.shape
src = x if kv is None else kv
S = src.shape[1]
split = lambda t, L: t.view(B, L, self.n_heads, self.d_kv).transpose(1, 2)
q, k, v = split(self.q(x), T), split(self.k(src), S), split(self.v(src), S)
scores = q @ k.transpose(-2, -1)                          # без / √D
if position_bias is None:
    if self.has_relative_attention_bias:
        position_bias = relative_position_bias(self.relative_attention_bias, T, S, not self.is_decoder,
                                               self.cfg.relative_attention_num_buckets,
                                               self.cfg.relative_attention_max_distance)
    else:
        position_bias = scores.new_zeros(1, self.n_heads, T, S)
```
Дальше: `scores + position_bias`, `masked_fill(~mask, -inf)`, softmax, `@ v`, склейка, `self.o`.

## Подсказка 3

`T5Block.forward`: `a, position_bias = sa.SelfAttention(sa.layer_norm(x), mask, position_bias=position_bias)`,
`x = x + a`; у декодера то же с `ca.EncDecAttention(ca.layer_norm(x), encoder_mask, kv=encoder_hidden_states)`;
FFN: `x = x + ff.DenseReluDense.wo(F.relu(ff.DenseReluDense.wi(ff.layer_norm(x))))`.

`T5Stack.forward`: маска `attention_mask.bool()[:, None, None, :]`, у декодера `& torch.tril(torch.ones(T, T,
dtype=torch.bool))`; `x = self.embed_tokens(input_ids)`; `position_bias = None`; в цикле
`x, position_bias = block(x, mask, position_bias, encoder_hidden_states, enc_mask)`; в конце
`self.final_layer_norm(x)`.
