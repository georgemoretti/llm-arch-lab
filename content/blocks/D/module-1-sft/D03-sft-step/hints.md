## Подсказка 1

```python
loss = F.cross_entropy(logits.flatten(0, -2), targets.flatten(), ignore_index=IGNORE, reduction="sum")
n = int((targets != IGNORE).sum())
```

## Подсказка 2

Число токенов ответа во всём шаге известно до прогона модели: оно зависит только от меток.
```python
n_total = sum(int((t != IGNORE).sum()) for _, t in micro_batches)
for ids, targets in micro_batches:
    loss_sum, _ = masked_ce_sum(model(ids), targets)
    (loss_sum / n_total).backward()
```

## Подсказка 3

```python
lr = lr_at(step, max_lr, warmup_steps, total_steps, min_lr)
for group in optimizer.param_groups:
    group["lr"] = lr
optimizer.zero_grad(set_to_none=True)
loss, n = accumulate_gradients(model, micro_batches)
grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
optimizer.step()
```
