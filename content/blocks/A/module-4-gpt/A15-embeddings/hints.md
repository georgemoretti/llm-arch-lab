## Подсказка 1

`pos = torch.arange(T, device=idx.device)` — номера позиций. `self.wpe(pos)` имеет форму `[T, C]`, и при
сложении с `[B, T, C]` PyTorch сам растянет его по батчу (broadcasting): копировать ничего не нужно.

## Подсказка 2

`lm_head.weight = wte.weight` — присваивание самого `nn.Parameter`. После него оба модуля держат один
тензор, оптимизатор видит один параметр, а `model.parameters()` отдаёт его один раз. `state_dict()` при
этом содержит два ключа, указывающих на один тензор, поэтому считать по нему нельзя.

## Подсказка 3

```python
logits = self.lm_head(self.emb(idx))          # [B, T, V]
loss = None
if targets is not None:
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
return logits, loss
```

`flatten(0, 1)` склеивает оси батча и времени: `[B, T, V]` → `[B·T, V]`.
