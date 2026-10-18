## Подсказка 1

`layer_types` и маска окна — по одной строке:
```python
["full_attention" if i % global_every == 0 else "sliding_attention" for i in range(num_layers)]

pos = torch.arange(T, device=device)
(pos[:, None] - pos[None, :]).abs() <= window // 2
```
GeGLU: `inp, gate = self.Wi(x).chunk(2, dim=-1)`; `self.Wo(F.gelu(inp) * gate)`.

## Подсказка 2

`ModernBertAttention.forward`:
```python
B, T, C = x.shape
qkv = self.Wqkv(x).view(B, T, 3, self.n_heads, self.head_dim)
q, k, v = qkv.permute(2, 0, 3, 1, 4)                 # [3, B, H, T, D] → три тензора
cos, sin = rope_cos_sin(torch.arange(T, device=x.device), self.head_dim, self.rope_theta)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
```
Дальше: `masked_fill(~mask, torch.finfo(scores.dtype).min)`, softmax, `@ v`, `transpose(1, 2).reshape(B, T, C)`,
`self.Wo`.

## Подсказка 3

Слой: `x = x + self.attn(self.attn_norm(x), mask)`, затем `x + self.mlp(self.mlp_norm(x))`.

Модель:
```python
pad = attention_mask.bool()[:, None, None, :]
masks = {"full_attention": pad,
         "sliding_attention": pad & local_window_mask(T, self.cfg.local_attention, input_ids.device)}
x = self.embeddings(input_ids)
for layer in self.layers:
    x = layer(x, masks[layer.attention_type])
return self.final_norm(x)
```
