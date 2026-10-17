## TL;DR

Моя модель: один `Model` из `llmlab/core/model.py`, собранный из YAML через реестр компонентов, — в нём работают
все детали блока C. Qwen3-0.6B = 28 Pre-Norm слоёв: RMSNorm (ε = 1e-6), GQA 16 / 8 голов с `head_dim = 128`
(≠ d / H), QK-Norm по голове до RoPE (θ = 10⁶), SwiGLU 1024 → 3072, связанный выход; 596 M параметров
(155.6 M — эмбеддинги). Загрузка: `config.json` → `ModelConfig`, `*.safetensors` (шарды по `index.json`) →
переименование → bf16 → fp32 → `load_state_dict`; связь `lm_head` ↔ `embed_tokens` не должна порваться.
Логиты совпадают с `transformers` до ~6·10⁻⁵ (fp32), жадная генерация — токен в токен. Генерация: prefill,
дальше по одному токену с KV-cache, выбор — `sample_next` из C25a, eos останавливает строку.

## Формула

$$\theta = \operatorname{fp32}(W_{\text{bf16}}),\quad W_{\text{lm\_head}} \equiv E,\quad
\lVert z^{\text{моя}} - z^{\text{transformers}}\rVert_\infty \le 10^{-3};\qquad
x_{t+1} = \texttt{sample\_next}\big(f(x_t \mid \text{cache}_{<t})\big)$$

## Формы тензоров

```
idx [B,T] → embed [B,T,1024] → 28 × (GQA: q [B,16,T,128], k/v [B,8,S,128]; SwiGLU [B,T,3072]) → norm → lm_head [B,T,151936]
генерация: prefill [B,T] → по одному токену [B,1] с KV-cache (28 × 2 × [B,8,S,128])
```

## Карта: какая задача за какую деталь отвечает

| деталь Qwen3-0.6B | задача |
|---|---|
| RMSNorm, Pre-Norm | [[C01-rmsnorm]], [[C08-norm-placement]] |
| SwiGLU | [[C02-swiglu]] |
| RoPE, θ = 10⁶ | [[C03-rope]] |
| GQA 16 / 8 | [[C04-gqa]] |
| KV-cache при генерации | [[C05-kv-cache]] |
| блок и модель | [[C06-llama-block]] |
| QK-Norm по голове | [[C09-qk-norm]] |
| config.json и имена весов | [[C24-hf-checkpoint]] |
| сэмплирование (temperature, top-k, top-p) | [[C25a-sampling]] |

## Что поменялось относительно C24

От словарей к работающей модели на диске: шарды safetensors, сборка без случайной инициализации, сохранение
связанных весов и генерация с KV-cache и остановкой на eos (сэмплирование — из C25a) — проверено
против `transformers` на настоящих весах.
