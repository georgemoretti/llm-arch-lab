## Подсказка 1

`encode_pair`: `[stoi[d] for d in str(n)]` и `[stoi[w] for w in number_to_words(n)] + [EOS]`.

`collate`: найди `S = max(len(s) for s, _ in pairs)` и так же `T`, затем
`torch.tensor([s + [pad_id] * (S - len(s)) for s, _ in pairs], dtype=torch.long)`.

## Подсказка 2

```python
model.train()
logits = model(src, shift_right(tgt, BOS))
loss = label_smoothed_ce(logits, tgt, eps, ignore_index=PAD)
opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
opt.step()
return loss.item()
```

## Подсказка 3

```python
was_training = model.training
model.eval()
src, tgt = collate([encode_pair(n, stoi) for n in numbers])
out = greedy_decode(model, src, MAX_WORDS + 1)
model.train(was_training)
hits = 0
for pred, want in zip(out.tolist(), tgt.tolist()):
    want = [t for t in want if t != PAD]
    if EOS in pred:
        pred = pred[: pred.index(EOS) + 1]
    hits += pred == want
return hits / len(numbers)
```
