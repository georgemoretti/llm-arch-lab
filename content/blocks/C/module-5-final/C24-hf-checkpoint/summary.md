## TL;DR

Чекпойнт HF = `config.json` + `model.safetensors`. Перенос весов в свою модель — два перевода: гиперпараметры
(`hidden_size` → `d_model`, `num_key_value_heads` → GQA, `head_dim`, `rope_theta`, `rms_norm_eps`,
`tie_word_embeddings`) и имена/раскладка тензоров. У Qwen3 `head_dim = 128` задан отдельно и не равен
`d_model / H = 64`, плюс QK-Norm по голове. Связанный `lm_head` = `embed_tokens` (в файле Qwen3-0.6B он есть и
совпадает). GPT-2 хранит веса в `Conv1D` как `[in, out]` → транспонировать; `c_attn` режется на q, k, v; буфер
`attn.bias` — маска, не вес. Всё неподдерживаемое (YaRN, окно, bias) — явная ошибка, а не тихий мусор.

## Формула

$$\text{Conv1D}: y = xW + b,\ W \in \mathbb{R}^{C_{in}\times C_{out}} \quad\Rightarrow\quad
\text{nn.Linear}: y = xW'^{\top} + b,\ W' = W^{\top}$$

## Формы тензоров

```
Qwen3-0.6B: q_proj [16·128, 1024] = [2048, 1024]   k/v_proj [8·128, 1024]   o_proj [1024, 2048]
            q_norm / k_norm [128]   gate/up [3072, 1024]   down [1024, 3072]   embed = lm_head [151936, 1024]
GPT-2:      c_attn [C, 3C] → .t() → [3C, C] → q, k, v [C, C];   c_proj [C, C] → .t();   c_fc [C, 4C] → .t()
```

## Что поменялось относительно C06

Та же модель, но гиперпараметры и веса берутся из чужого формата: появляются `head_dim ≠ d/H`, QK-Norm,
связанный выход, bf16 → fp32 и (у GPT-2) транспонированные Conv1D-веса.
