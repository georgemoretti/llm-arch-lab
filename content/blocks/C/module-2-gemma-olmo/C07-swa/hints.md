## Подсказка 1

Маску удобно строить из двух векторов позиций и broadcasting: `q_pos = torch.arange(S - T, S)[:, None]`
(столбец `[T, 1]`) и `k_pos = torch.arange(S)[None, :]` (строка `[1, S]`). Тогда `k_pos <= q_pos` —
это уже causal-маска `[T, S]`, а `q_pos - k_pos` — матрица расстояний «на сколько ключ в прошлом».

## Подсказка 2

Окно из `W` позиций, включая сам токен, — это расстояния `0, 1, …, W − 1`, то есть
`q_pos - k_pos < W` (строгое неравенство). Проверка: при `W = 1` остаётся только диагональ.
Для `layer_types`: у Gemma слой `i` — full, когда `(i + 1) % pattern == 0`, — последний в группе,
поэтому слой 0 всегда sliding.

## Подсказка 3

`forward` — это `LlamaAttention.forward` из C05 с `cache=None`: позиции `arange(T)` → `cos, sin` →
проекции, `view`, `transpose(1, 2)` → `apply_rope` к q и k → `repeat_kv` → `scores = q @ k.transpose(-2, -1) / sqrt(D)`
→ `scores.masked_fill(~sliding_window_mask(T, window=self.window), -inf)` → softmax → `@ v` →
`transpose(1, 2).reshape(B, T, H * D)` → `o_proj`.
