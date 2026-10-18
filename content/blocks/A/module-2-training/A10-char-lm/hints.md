## Подсказка 1

`itos = sorted(set(text))`, `stoi = {c: i for i, c in enumerate(itos)}`. `encode`:
`torch.tensor([stoi[c] for c in text], dtype=torch.long)`. `decode`:
`"".join(itos[int(i)] for i in ids)` работает и для тензора, и для списка.

## Подсказка 2

`ids.unfold(0, k, 1)` даёт все окна длины `k` с шагом 1: их `N − k + 1`. У последнего окна нет
следующего символа, поэтому `X = ids.unfold(0, k, 1)[:-1]`, `Y = ids[k:]`.

## Подсказка 3

Forward: `e = self.emb(X)` имеет форму `[B, k, E]`, `e.flatten(1)` — `[B, k·E]`. Дальше
`torch.tanh(self.hidden(...))` и `self.out(...)`. Перплексия: `math.exp(F.cross_entropy(model(X), Y).item())`
между `model.eval()` и возвратом прежнего режима.
