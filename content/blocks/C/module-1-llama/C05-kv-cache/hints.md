## Подсказка 1

Начни с `KVCache.update`: если `self.k is None` — просто сохрани `k`, `v`; иначе `torch.cat` со
старыми по оси времени (у тензора `[B, H_kv, T, D]` это `dim=2`). Верни `self.k, self.v`.
В `forward` первым делом узнай `start = cache.seq_len` — **до** того, как что-то дописал в кеш.

## Подсказка 2

Порядок в `forward`: позиции `torch.arange(start, start + T)` → `cos, sin` → проекции и разбиение
на головы → `apply_rope` к q и k → `cache.update(k, v)` (теперь k, v — весь кеш, `S = k.shape[2]`)
→ `repeat_kv` → attention. В кеш идут k, v **до** `repeat_kv`: иначе он раздуется в `n_rep` раз.

## Подсказка 3

Маска прямоугольная: `torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)` — «треугольник»,
прижатый к правому нижнему углу. При `T = S` это обычная causal-маска, при `T = 1` — все единицы.
Дальше как в C04: `scores.masked_fill(~mask, -inf)` → softmax → `@ v` →
`transpose(1, 2).reshape(B, T, H * D)` → `o_proj`.
