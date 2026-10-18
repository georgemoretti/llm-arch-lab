## Подсказка 1

`truncate_pair`: если `len(a) + len(b) <= max_tokens`, вернуть `list(a), list(b)`. Иначе:
`a_longer = len(a) > len(b)` (при равенстве длинным считается `b`), `half = max_tokens // 2`.

## Подсказка 2

```python
n_long, n_short = (len(a), len(b)) if a_longer else (len(b), len(a))
if n_short > half:
    n_short, n_long = half, max_tokens - half
else:
    n_long = max_tokens - n_short
```

Затем переведи `n_long / n_short` обратно в `n_a / n_b` и верни `a[:n_a], b[:n_b]`.

## Подсказка 3

`build_inputs` для пары: `ids = [cls_id] + a + [sep_id] + b + [sep_id]`,
`types = [0] * (len(a) + 2) + [1] * (len(b) + 1)`.

`BertEmbeddings.forward`:
```python
positions = torch.arange(input_ids.shape[1], device=input_ids.device)
if token_type_ids is None:
    token_type_ids = torch.zeros_like(input_ids)
x = self.word_embeddings(input_ids) + self.position_embeddings(positions) + self.token_type_embeddings(token_type_ids)
return self.dropout(self.LayerNorm(x))
```
