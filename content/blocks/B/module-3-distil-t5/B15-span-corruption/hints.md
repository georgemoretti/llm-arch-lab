## Подсказка 1

`random_segmentation` — выбрать разрезы без повторов и взять разности границ:
```python
cuts = torch.randperm(num_items - 1, generator=generator)[: num_segments - 1].sort().values + 1
bounds = torch.cat([torch.tensor([0]), cuts, torch.tensor([num_items])])
return bounds.diff()
```

## Подсказка 2

`random_spans_noise_mask`: посчитай `num_noise` и `num_spans`, затем
```python
lengths = torch.stack([text_lengths, noise_lengths], dim=1).reshape(-1)   # текст₀, шум₀, текст₁, …
flags = torch.tensor([False, True]).repeat(num_spans)
return flags.repeat_interleave(lengths)
```
Порядок вызовов `random_segmentation` важен для воспроизводимости: сначала шум, потом текст.

## Подсказка 3

`noise_span_to_unique_sentinel`:
```python
prev = torch.cat([noise_mask.new_zeros(1), noise_mask[:-1]])
first, rest = noise_mask & ~prev, noise_mask & prev
span = torch.cumsum(first.long(), 0) - 1
out = torch.where(first, table[span.clamp(min=0)], ids)    # table = torch.tensor(sentinels)
return out[~rest]
```
`span_corruption` — эта функция дважды: с `noise_mask` для входа и с `~noise_mask` для цели, к обоим — EOS.
