## Подсказка 1

`shift_right`: `bos = torch.full_like(tgt[:, :1], bos_id)` — столбец `[B, 1]` того же типа; затем
`torch.cat([bos, tgt[:, :-1]], dim=1)`.

В лоссе: `logp = F.log_softmax(logits.float(), dim=-1)`. Для `gather` индексы должны быть неотрицательными, поэтому
сначала замени `ignore_index` любым допустимым классом: `safe = target.masked_fill(~keep, 0)`, а потом выбрось эти
позиции маской `keep`.

## Подсказка 2

```python
nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)    # [B, T]
uniform = -logp.mean(dim=-1)                               # [B, T]
loss = (1 - eps) * nll + eps * uniform
return loss[keep].sum() / keep.sum().clamp(min=1)
```

`encode`: `return self.encoder(src), padding_mask(src, self.cfg.pad_id)`. `decode`:
`return self.lm_head(self.decoder(tgt_in, memory, src_mask))`.

## Подсказка 3

```python
memory, src_mask = model.encode(src)
out = torch.full((B, 1), cfg.bos_id, dtype=torch.long)
done = torch.zeros(B, dtype=torch.bool)
for _ in range(max_len):
    next_tok = model.decode(out, memory, src_mask)[:, -1].argmax(-1)
    next_tok = next_tok.masked_fill(done, cfg.pad_id)
    out = torch.cat([out, next_tok[:, None]], dim=1)
    done |= next_tok == cfg.eos_id
    if done.all():
        break
return out[:, 1:]
```
