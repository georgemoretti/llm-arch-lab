## Подсказка 1

InfoNCE — это обычная классификация, где «классы» — документы батча:
```python
logits = q @ d.T / temperature                      # [N, M]
labels = torch.arange(q.shape[0])                   # запрос i → документ i
F.cross_entropy(logits, labels)
```

## Подсказка 2

Симметричный: те же `logits` и `labels`, но ещё `F.cross_entropy(logits.T, labels)` — документ `j` выбирает свой
запрос. Среднее двух.

## Подсказка 3

```python
topk = scores.topk(min(k, scores.shape[1]), dim=-1).indices        # [N, k]
hit = (topk == positives[:, None]).any(-1)
float(hit.float().mean())
```
