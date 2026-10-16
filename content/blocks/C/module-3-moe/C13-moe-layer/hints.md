## Подсказка 1

Начни с `forward_naive`: `dense = torch.zeros(N, E).scatter_(1, indices, weights)` — у каждого токена
в строке $k$ ненулевых весов. Дальше `sum(dense[:, e:e+1] * expert(h) for e, expert in enumerate(self.experts))`.
Срез `e:e+1` оставляет форму `[N, 1]`, чтобы умножение транслировалось на `[N, C]`. Не забудь
вернуть форму `x.shape` и логиты.

## Подсказка 2

В `forward` сделай три плоских тензора длины `N·k`: `flat_expert = indices.reshape(-1)`,
`flat_token = torch.arange(N).repeat_interleave(k)`, `flat_weight = weights.reshape(-1)`. Слот
`i` означает «токен `flat_token[i]` идёт к эксперту `flat_expert[i]` с весом `flat_weight[i]`».
`order = flat_expert.argsort()` и `counts = torch.bincount(flat_expert, minlength=E).tolist()`
дают для каждого эксперта отрезок `order[start:start + counts[e]]`.

## Подсказка 3

Цикл по экспертам: если `counts[e] == 0` — `continue`; иначе `slots = order[start:start + n_e]`,
`tok = flat_token[slots]`, `y = self.experts[e](h[tok]) * flat_weight[slots, None]`,
`out.index_add_(0, tok, y)`, `start += n_e`. Именно `index_add_`: `out[tok] = y` или `index_copy_`
перезапишут вклад предыдущего эксперта того же токена.
