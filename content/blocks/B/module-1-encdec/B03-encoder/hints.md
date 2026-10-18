## Подсказка 1

Маска считается один раз в `Encoder.forward`: `mask = padding_mask(src, self.pad_id)`. Слои её не строят: у них
нет `ids`, только векторы.

## Подсказка 2

Post-LN — две строки, каждая «подслой, остаток, норма»:

```python
x = self.norm1(x + self.self_attn(x, mask=mask))
return self.norm2(x + self.ffn(x))
```

В Pre-LN норма переезжает внутрь, к входу подслоя, а остаток берётся от исходного `x`.

## Подсказка 3

```python
if self.norm_first:
    x = x + self.self_attn(self.norm1(x), mask=mask)
    return x + self.ffn(self.norm2(x))
```

`Encoder.forward`: `x = self.embed(src)`, цикл `for layer in self.layers: x = layer(x, mask)`, затем
`return self.norm(x)`.
