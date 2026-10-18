## Подсказка 1

`get_batch`: `idx = torch.randint(0, X.shape[0], (batch_size,), generator=generator)`, затем
`X[idx], y[idx]`. `evaluate`: `was = model.training`, `model.eval()`, лосс через `.item()`,
`model.train(was)`. Декоратор `@torch.no_grad()` уже стоит.

## Подсказка 2

Шаг обучения в `train_loop`:

```python
for g in optimizer.param_groups: g["lr"] = lr_fn(step)
xb, yb = get_batch(...)
loss = loss_fn(model(xb), yb)
optimizer.zero_grad(set_to_none=True)
loss.backward()
```

## Подсказка 3

После `backward()`: `nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)`, если порог задан,
затем `optimizer.step()`, затем `history["train"].append(loss.item())`. Валидация:
`if (step + 1) % eval_every == 0 or step == steps - 1`.
