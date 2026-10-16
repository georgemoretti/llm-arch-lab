## Подсказка 1

`forward` — это `LlamaAttention.forward` из C05 без кеша, где поворот q и k обёрнут в
`if self.use_rope:`. Маску не трогай: она нужна в любом слое декодера.

## Подсказка 2

`rope_layers`: слой `i` без RoPE, если `(i + 1) % interval == 0` — последний в каждой группе.
Значит, True там, где `(i + 1) % interval != 0`. Проверь на SmolLM3: `rope_layers(8, 4)` →
`[True, True, True, False, True, True, True, False]`.

## Подсказка 3

```python
return nn.ModuleList(
    NoPEAttention(d_model, n_heads, n_kv_heads, use_rope, rope_theta)
    for use_rope in rope_layers(n_layers, no_rope_layer_interval)
)
```

Обычный `list` вместо `nn.ModuleList` не подойдёт: параметры слоёв не попадут в `parameters()` и
`state_dict()`.
