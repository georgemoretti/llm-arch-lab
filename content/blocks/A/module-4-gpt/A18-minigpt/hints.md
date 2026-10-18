## Подсказка 1

`forward` — четыре строки: `x = self.emb(idx)`, цикл `for block in self.blocks: x = block(x)`,
`logits = self.lm_head(self.ln_f(x))`, loss как в A15 (`F.cross_entropy(logits.flatten(0, 1), targets.flatten())`).

## Подсказка 2

`get_batch`: самое правое начало `s` должно оставлять место для `y`, который заканчивается на
`data[s + block_size]`. Значит, `s ≤ len(data) − block_size − 1`, и верхняя граница `torch.randint`
(не включительно) равна `len(data) − block_size`. Затем
`torch.stack([data[s:s + block_size] for s in starts])` и то же со сдвигом на 1.

## Подсказка 3

`generate`:

```python
was_training = self.training
self.eval()
for _ in range(max_new_tokens):
    logits, _ = self(idx[:, -self.config.block_size:])
    last = logits[:, -1]
    if temperature == 0:
        nxt = last.argmax(-1, keepdim=True)
    else:
        nxt = torch.multinomial(F.softmax(last / temperature, dim=-1), 1, generator=generator)
    idx = torch.cat([idx, nxt], dim=1)
self.train(was_training)
return idx
```

`train` — цикл A09 с `get_batch` вместо `DataLoader`.
