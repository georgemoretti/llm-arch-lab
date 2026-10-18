## Подсказка 1

Разбери три случая по порядку `if`: `step < warmup_steps`, `step >= total_steps`, иначе. Если
`warmup_steps = 0`, первая ветка просто не сработает, и деления на ноль не будет.

## Подсказка 2

Доля пройденного decay: `progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)`.
Косинус $\tfrac12(1 + \cos \pi p)$ идёт от 1 при $p = 0$ до 0 при $p = 1$. Умножь его на
`(max_lr - min_lr)` и прибавь `min_lr`.

## Подсказка 3

```python
grads = [p.grad for p in params if p.grad is not None]
total = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)).item()
coef = max_norm / (total + eps)
if coef < 1: for g in grads: g.mul_(coef)
```
