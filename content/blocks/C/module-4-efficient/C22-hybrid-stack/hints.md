## Подсказка 1

Расписание — одна строка со списковым включением: `(i + 1) % interval == 0` → full. С `i % interval` full-слой
был бы первым в группе (0, 4, 8, …), а у Qwen3-Next и Gemma 3 он последний.

## Подсказка 2

Блоку не нужно знать тип mixer-а — только его выход: `h = self.mixer(self.input_layernorm(x))`; если это
кортеж (GatedDeltaNet возвращает `(y, state)`), взять `h[0]`. Дальше два residual-а, как в Pre-Norm блоке C08.

## Подсказка 3

Память: `full = layer_types.count("full_attention")`, `linear = layer_types.count("linear_attention")`;
`(full * 2 * n_kv_heads * head_dim * seq_len + linear * linear_heads * linear_k_dim * linear_v_dim) * batch * bytes_per_elem`.
