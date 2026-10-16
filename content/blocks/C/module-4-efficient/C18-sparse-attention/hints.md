## Подсказка 1

Индексатор — это маленькое attention без softmax: оценки «запрос × ключ» по нескольким головам, ReLU и
взвешенная сумма голов. Ключ у индексатора один на токен, поэтому `k_proj(x)` даёт `[B, T, d_I]` без оси голов.
Будущее маскируй прямо в оценках индексатора: иначе top-k честно выберет «самый полезный» ключ — из будущего.

## Подсказка 2

`logits = torch.einsum("bthd,bsd->bths", q, k) / math.sqrt(d_I)`, затем
`scores = torch.einsum("bth,bths->bts", w, F.relu(logits))`, где `w = weights_proj(x) / math.sqrt(H_I)`.
Маска будущего: `~torch.ones(T, T, dtype=torch.bool).tril()` → `masked_fill(…, float("-inf"))`.

## Подсказка 3

```python
k = min(top_k, index_scores.shape[-1])
idx = index_scores.topk(k, dim=-1).indices
keep = torch.zeros_like(index_scores, dtype=torch.bool).scatter_(-1, idx, True)
keep = keep & torch.isfinite(index_scores)       # в строке t всего t + 1 допустимых ключей
```

В `sparse_attention` маска `[B, T, S]` общая для голов: `scores.masked_fill(~keep.unsqueeze(1), float("-inf"))`.
