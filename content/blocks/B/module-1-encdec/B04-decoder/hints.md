## Подсказка 1

`padding_mask(tgt, pad_id)` имеет форму `[B, 1, 1, T]`, `causal_mask(T)` — `[T, T]`. Оператор `&` растянет их до
`[B, 1, T, T]`: строка $t$ разрешает ключи $s \le t$, которые не PAD.

## Подсказка 2

У `MultiHeadAttention` из B01 сигнатура `forward(x, context=None, mask=None)`. Self-attention:
`self.self_attn(x, mask=tgt_mask)`. Cross-attention: `self.cross_attn(x, memory, mask=src_mask)` — запросы из
`x`, ключи и значения из `memory`. Causal-маска в cross-attention не нужна.

## Подсказка 3

```python
if self.norm_first:
    x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
    x = x + self.cross_attn(self.norm2(x), memory, mask=src_mask)
    return x + self.ffn(self.norm3(x))
x = self.norm1(x + self.self_attn(x, mask=tgt_mask))
x = self.norm2(x + self.cross_attn(x, memory, mask=src_mask))
return self.norm3(x + self.ffn(x))
```

В `Decoder.forward`: `tgt_mask = target_mask(tgt, self.pad_id)`, `x = self.embed(tgt)`, цикл по слоям
`x = layer(x, memory, tgt_mask, src_mask)`, `return self.norm(x)`.
