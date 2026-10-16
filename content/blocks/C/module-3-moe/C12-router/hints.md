## Подсказка 1

`route` — четыре строки: `logits.float()` → оценки (`softmax(dim=-1)` или `sigmoid()`) →
`scores.topk(top_k, dim=-1)` (возвращает и значения, и индексы) → перенормировка и множитель.
Ось везде последняя: это ось экспертов.

## Подсказка 2

Перенормировка: `weights / weights.sum(dim=-1, keepdim=True)` — `keepdim`, чтобы `[N, 1]`
транслировался на `[N, k]`. Множитель `routed_scaling_factor` применяется в конце в любом случае
(и с перенормировкой, и без).

## Подсказка 3

`forward`: `h = x.reshape(-1, x.shape[-1])`, `logits = F.linear(h.float(), self.weight.float())`,
затем `weights, indices = route(logits, self.top_k, self.scoring_func, self.norm_topk_prob,
self.routed_scaling_factor)` и `return logits, weights, indices`.
