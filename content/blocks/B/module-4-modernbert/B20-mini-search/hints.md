## Подсказка 1

```python
n = rng.randint(min_len, max_len)
start = rng.randrange(max(1, len(text_words) - n + 1))     # правая граница randrange не включается
query = text_words[start:start + n]
```
Контекст: `text_words[:start] + text_words[start + n:]` с вероятностью `remove_prob`, иначе `list(text_words)`.

## Подсказка 2

```python
df = torch.zeros(len(vocab))
for d in docs:
    for t in set(d):                       # set: повтор в документе не считается
        if t in vocab:
            df[vocab[t]] += 1
idf = torch.log((1 + len(docs)) / (1 + df)) + 1
```
`tfidf_vectors`: счётчики `out[i, vocab[t]] += 1`, затем `out * idf` и `out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)`.

## Подсказка 3

```python
loss = info_nce(model(q_ids, q_mask), model(d_ids, d_mask), temperature)
opt.zero_grad(); loss.backward(); opt.step()
return loss.item()
```
`hit_at_k`: `top = scores.topk(min(k, scores.shape[1]), dim=-1).indices.tolist()`, затем
`sum(bool(gold[i] & set(row)) for i, row in enumerate(top)) / len(gold)`.
