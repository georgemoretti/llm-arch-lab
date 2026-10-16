## Подсказка 1

`expert_load`: `torch.bincount(indices.reshape(-1), minlength=n_experts).float()`, затем поделить на
сумму. `minlength` обязателен: иначе эксперты с большими номерами, которых никто не выбрал, пропадут.

## Подсказка 2

`load_balancing_loss`: `probs = router_logits.float().softmax(-1)`, `indices = probs.topk(top_k, -1).indices`,
`f = expert_load(indices, E)` (счётчик, без градиента — так и задумано), `P = probs.mean(dim=0)`,
ответ `E * (f * P).sum()`.

## Подсказка 3

`route_with_bias`: `indices = (scores + bias).topk(top_k, -1).indices`, `weights = scores.gather(-1, indices)`,
затем (если надо) деление на сумму. `update_bias`: `bias - gamma * torch.sign(load - load.mean())` —
`sign` даёт −1/0/+1, поэтому шаг фиксированный.
