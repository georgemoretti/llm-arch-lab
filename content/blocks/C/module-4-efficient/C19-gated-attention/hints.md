## Подсказка 1

Partial RoPE — это обычный RoPE из C03 на «подголове» из первых $R$ каналов. Частоты считаются для
размера $R$ (`rope_cos_sin(pos, R)`), а не для $D$ с последующей обрезкой. Остальные каналы просто
приклеиваются обратно.

## Подсказка 2

```python
R = cos.shape[-1]
x_rot, x_pass = x[..., :R], x[..., R:]
return torch.cat((apply_rope(x_rot, cos, sin), x_pass), dim=-1)
```

В `forward` всё, кроме гейта и partial RoPE, — как `QKNormAttention` из C09 (режим `head`).

## Подсказка 3

Раскладка гейта — по головам: `q, gate = self.q_proj(x).view(B, T, H, 2 * D).chunk(2, dim=-1)`,
затем `gate = gate.reshape(B, T, H * D)`. После attention: `out.transpose(1, 2).reshape(B, T, H * D)`,
`self.o_proj(out * torch.sigmoid(gate))`.
