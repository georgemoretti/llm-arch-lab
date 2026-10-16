## Подсказка 1

`self.q_norm` и `self.k_norm` — обычные `RMSNorm` из C01: нормируют по **последней** оси. Поэтому
ось нормы выбирается тем, **когда** её применить: к `[B, T, H·D]` (до `view`) — норма по всей
проекции; к `[B, H, T, D]` (после `view` и `transpose`) — по каждой голове.

## Подсказка 2

Порядок в `scores`: `q = q_proj(x)`, `k = k_proj(x)` → (если `"full"` — нормы) → `view(B, T, H, D).transpose(1, 2)`
(для k — `H_kv`) → (если `"head"` — нормы) → `cos, sin = rope_cos_sin(arange(T), D, θ)` →
`apply_rope` к q и k → v → `repeat_kv` для k и v → `q @ k.transpose(-2, -1) / sqrt(D)`.

## Подсказка 3

`forward`: `scores, v = self.scores(x)`, маска `torch.ones(T, T, dtype=torch.bool).tril()`,
`scores.masked_fill(~mask, float("-inf"))`, `softmax(dim=-1)`, `@ v` → `[B, H, T, D]` →
`transpose(1, 2).reshape(B, T, -1)` → `o_proj`. Если падает только тест порядка — проверь, что
`apply_rope` вызывается после нормы в обоих режимах.
