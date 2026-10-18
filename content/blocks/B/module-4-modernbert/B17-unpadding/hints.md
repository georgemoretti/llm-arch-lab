## Подсказка 1

```python
seqlens = attention_mask.sum(-1, dtype=torch.int32)
indices = attention_mask.flatten().nonzero().flatten()
cu_seqlens = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))      # ноль в начало
```
`pack`: `x.flatten(0, 1)[indices]`. `unpack`: `out = packed.new_zeros(B * T, *packed.shape[1:])`,
`out[indices] = packed`, `out.view(B, T, *packed.shape[1:])`.

## Подсказка 2

Номер текста каждого токена упаковки:
```python
pos = torch.arange(N)
seq_id = torch.searchsorted(cu_seqlens[1:], pos, right=True)    # cu[1:] — концы текстов
```
Позиции: `pos - cu_seqlens[seq_id].long()`. Маска: `seq_id[:, None] == seq_id[None, :]`, с окном ещё
`& ((pos[:, None] - pos[None, :]).abs() <= window // 2)`.

## Подсказка 3

`varlen_attention`: головы вперёд, `q.transpose(0, 1)` → `[H, N, D]`;
`scores = q_h @ k_h.transpose(-2, -1) / D ** 0.5`;
`scores.masked_fill(~varlen_mask(cu_seqlens, window), torch.finfo(scores.dtype).min)`; softmax; `@ v_h`;
`.transpose(0, 1)` обратно.

`padding_savings`: `L = attention_mask.sum(-1).double()`; `1 - L.sum() / (B * T)` и `1 - (L ** 2).sum() / (B * T * T)`,
обернуть в `float`.
