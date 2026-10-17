# Чекпойнт Hugging Face → своя модель: config.json и имена весов

## Зачем это нужно

За блок C ты собрал по детали все ключевые идеи современных декодеров. В финальном проекте они
работают вместе: `llmlab/core/model.py` собирает модель из YAML-конфига через реестр компонентов
(нормы, позиции, attention, FFN/MoE, расписание слоёв), а готовые конфиги лежат в `llmlab/configs/`.
Осталось влить в эту модель **настоящие веса**: Qwen3-0.6B, обученную на триллионах токенов.

Чекпойнт на Hugging Face — это два файла:

- `config.json` — гиперпараметры в терминах `transformers` (`hidden_size`, `num_key_value_heads`, …);
- `model.safetensors` — словарь «имя → тензор» в той раскладке, которую ждёт `transformers`.

Наша модель ничего не знает о `transformers`, поэтому нужен «переводчик» с двух сторон: конфиг HF →
наш `ModelConfig`, имена и формы тензоров HF → наш `state_dict`. Это самая «инженерная» часть любого
переноса весов — и та, где появляются самые коварные баги: веса загрузились без единой ошибки, а модель
несёт чушь.

## Что поменялось относительно C06

| | C06 (Llama, свой код) | C24 (реальный чекпойнт) |
|---|---|---|
| откуда гиперпараметры | `LlamaConfig` с игрушечными числами | `config.json` производителя |
| откуда веса | случайная инициализация | `model.safetensors` (bf16) |
| `head_dim` | всегда `d_model / n_heads` | у Qwen3 **задан отдельно**: 128 ≠ 1024 / 16 |
| QK-Norm | нет | `q_norm`, `k_norm` по голове (C09) |
| имена | наши | HF: префикс `model.`, у GPT-2 — совсем другие |

## Что значит каждое поле

Qwen3 и Llama (`model_type: qwen3 | llama`):

| `config.json` | `ModelConfig` | откуда знаем |
|---|---|---|
| `hidden_size`, `num_hidden_layers`, `vocab_size` | `d_model`, `n_layers`, `vocab_size` | |
| `num_attention_heads`, `num_key_value_heads` | `attention.n_heads`, `attention.n_kv_heads` | GQA, C04 |
| `head_dim` (если есть, иначе `hidden_size / heads`) | `attention.head_dim` | |
| `intermediate_size` | `ffn.d_hidden` (`type: swiglu`) | C02 |
| `rms_norm_eps` | `norm.eps` (`rmsnorm`, `pre`) | C01, C08 |
| `rope_theta` или `rope_parameters.rope_theta` | `position.theta` (`rope`) | C03 |
| `tie_word_embeddings` | `tie_embeddings` | |
| `model_type == "qwen3"` | `attention.qk_norm: head` | C09, `modeling_qwen3.py` |

GPT-2 (`model_type: gpt2`): `n_embd`, `n_layer`, `n_head`, `n_positions` (→ обучаемые позиции),
`layer_norm_epsilon` (→ `layernorm`), bias у attention и FFN, `gelu` с расширением ×4, выход связан с `wte`.

Формат `config.json` меняется между версиями `transformers`: файлы, сохранённые версией 5.x, хранят RoPE как
`"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}`, а поля `rope_theta` в них нет.
Если взять «θ по умолчанию» (10 000), модель загрузится без единой ошибки и будет считать позиции не так, как
при обучении, — поддержи оба формата.

Если в конфиге есть то, чего наша модель не умеет (`rope_scaling` — YaRN/«llama3», включённое окно,
bias в Llama/Qwen, другая активация), — **ValueError**. Молча проигнорированная опция = модель с правильными
весами и неправильной математикой.

## Формула

Конвертация весов — это переименование плюс две операции над тензорами. Для Qwen3 / Llama — только имена:
$\theta_{\text{наш}} = W_{\text{HF}}$. GPT-2 хранит линейные слои как `Conv1D` с весом $[C_{\text{in}}, C_{\text{out}}]$
($y = xW + b$), а `nn.Linear` — как $[C_{\text{out}}, C_{\text{in}}]$ ($y = xW^\top + b$), поэтому вес
транспонируется; общая проекция `c_attn` режется на три:

$$
W_{\text{Linear}} = W_{\text{Conv1D}}^{\top},\qquad
\begin{bmatrix} W_Q \\ W_K \\ W_V \end{bmatrix} = W_{\text{c\_attn}}^{\top} \in \mathbb{R}^{3C \times C},\qquad
b_Q, b_K, b_V = \operatorname{split}(b_{\text{c\_attn}}, C)
$$

Для `head_dim` из конфига: $d_h = \texttt{head\_dim}$, если он задан, иначе $C / H$ — у Qwen3-0.6B
$d_h = 128 \ne 1024 / 16$, и `q_proj` имеет форму $[H d_h, C] = [2048, 1024]$.

## Формы тензоров

Qwen3-0.6B, один слой (имена в файле → наши):

```shapes
model.embed_tokens.weight [151936, 1024] — → embed_tokens.weight
model.layers.{i}.input_layernorm.weight [1024] — → layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.q_proj.weight [2048, 1024] — 16 голов × 128 = 2048 ≠ 1024!
model.layers.{i}.self_attn.k_proj.weight [1024, 1024] — 8 KV-голов × 128
model.layers.{i}.self_attn.v_proj.weight [1024, 1024]
model.layers.{i}.self_attn.o_proj.weight [1024, 2048]
model.layers.{i}.self_attn.q_norm.weight [128] — QK-Norm по голове
model.layers.{i}.self_attn.k_norm.weight [128]
model.layers.{i}.post_attention_layernorm.weight [1024]
model.layers.{i}.mlp.{gate,up}_proj.weight [3072, 1024]
model.layers.{i}.mlp.down_proj.weight [1024, 3072]
model.norm.weight [1024] — → norm.weight
lm_head.weight [151936, 1024] — в файле есть, но == embed_tokens (tied)
```

Имена нашей модели специально повторяют HF, поэтому для Qwen3 перевод — это снятие префикса плюс
обработка связанного `lm_head`. GPT-2 интереснее:

```shapes
h.{i}.attn.c_attn.weight [C, 3C] — Conv1D хранит (in, out) → .t() → [3C, C] → split по dim 0 → q, k, v по [C, C]
h.{i}.attn.c_attn.bias [3C] — → split → bias q, k, v по [C]
h.{i}.attn.c_proj.weight [C, C] — Conv1D → .t() → self_attn.o_proj.weight ⚠ квадратный!
h.{i}.mlp.c_fc.weight [C, 4C] — Conv1D → .t() → mlp.c_fc.weight [4C, C]
h.{i}.mlp.c_proj.weight [4C, C] — Conv1D → .t() → mlp.c_proj.weight [C, 4C]
h.{i}.attn.bias [1, 1, 1024, 1024] — выбросить: это causal-маска, а не вес
- wte / wpe / ln_f / h.{i}.ln_1 / ln_2 → embed_tokens / embed_positions / norm / нормы блока
```

`Conv1D` в GPT-2 — это линейный слой, у которого матрица хранится транспонированной (`y = x @ W + b`,
`W: [in, out]`). У прямоугольных матриц ошибку выдаст `load_state_dict`, а `c_proj` квадратный: форма
совпадёт, и модель молча посчитает $xW^\top$ вместо $xW$.

:::viz param-budget:::

## Задание

В `solution.py` (рабочая копия в `workspace/C24-hf-checkpoint/`) реализуй:

1. `config_from_hf(hf: dict) -> ModelConfig` — для `qwen3`, `llama`, `gpt2`, с `ValueError` на
   неподдерживаемое. Результат должен совпасть с `llmlab/configs/{qwen3-0.6b,smollm2-135m,gpt2}.yaml`.
2. `convert_state_dict(hf_sd, cfg, model_type, dtype=torch.float32) -> dict` — имена, транспонирование
   Conv1D, разрезание `c_attn`, связанный `lm_head` (если в файле он есть и отличается — `ValueError`),
   приведение к `dtype`.

`check_state_dict(model, sd)` дан готовым: скажет, каких ключей не хватает, какие лишние и где не те формы.
Полезный приём — распечатать `sorted(Model(cfg).state_dict())` рядом с `sorted(hf_sd)`.

## Критерии готовности

- config.json Qwen3-0.6B, SmolLM2-135M и GPT-2 дают те же конфиги, что в `llmlab/configs/`;
- формат transformers 5.x (`rope_parameters`) понимается так же, как старый;
- `rope_scaling` / `rope_type ≠ default`, окно, bias, чужая активация, чужой `model_type` → `ValueError`;
- после конвертации «чекпойнта» Qwen3, Llama (без связи, с мусорными `inv_freq`) и GPT-2 модель даёт
  **те же логиты**, что исходная; все тензоры — `float32`;
- если установлен `transformers` (`uv sync --extra weights`): веса настоящих `Qwen3ForCausalLM` и
  `GPT2LMHeadModel` (случайных, игрушечного размера) дают те же логиты, что у HF.

Запуск: `llmlab test C24-hf-checkpoint`. Дальше — C25: загрузить настоящие 0.6B параметров и поговорить с моделью.
