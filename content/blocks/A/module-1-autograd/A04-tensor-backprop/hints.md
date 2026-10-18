## Подсказка 1

Сверяй формы. `dx` должен иметь форму `x` `[N, D_in]`: из `dy` `[N, D_out]` его даёт умножение на
`W` `[D_out, D_in]`. `dW` должен иметь форму `W`: это `dy.T @ x`. `b` прибавляется к каждой из N
строк, поэтому `db` — сумма `dy` по оси батча. В `cache` Linear нужны `x` (для `dW`) и `W` (для `dx`).

## Подсказка 2

Log-softmax без переполнения:

```python
shifted = logits - logits.max(dim=1, keepdim=True).values
log_probs = shifted - shifted.exp().sum(dim=1, keepdim=True).log()
```

Лосс: `-log_probs[torch.arange(N), targets].mean()`.

## Подсказка 3

Градиент: `d = log_probs.exp()` (это softmax), затем `d[torch.arange(N), targets] -= 1`, и раздели на
`N`. В `mlp_loss_and_grads` backward идёт в обратном порядке: CE → Linear 2 → ReLU → Linear 1.
`dx` первого слоя не нужен.
