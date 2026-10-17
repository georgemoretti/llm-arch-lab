# Проект: Qwen3-0.6B на своих деталях — загрузка весов и генерация

## Зачем это нужно

Финал блока C. Всё, что ты писал по кусочку — RMSNorm (C01), SwiGLU (C02), RoPE (C03), GQA (C04),
KV-cache (C05), Pre-Norm блок (C06), QK-Norm (C09), — теперь работает как **настоящая модель**: 596 миллионов
обученных параметров Qwen3-0.6B, которые ты сам загружаешь в свой код и заставляешь писать текст.

Это не игрушка «похожая на Qwen». Тест сравнивает логиты твоей модели с официальной реализацией
`transformers` на одном и том же промпте: если хоть одна деталь за весь блок C сделана неверно (норма не до
RoPE, не тот θ, головы GQA перепутаны), расхождение будет порядка единиц, а не $10^{-5}$.

## Что поменялось относительно C24

C24 перевёл `config.json` и имена тензоров. Здесь — всё остальное, что нужно для работающей модели:

| | C24 | C25 |
|---|---|---|
| вход | dict из `config.json`, словарь тензоров | папка на диске: `config.json` + `*.safetensors` (иногда шарды) |
| выход | `ModelConfig`, `state_dict` | готовая `Model` в `eval()`, со связанным `lm_head` |
| генерация | — | KV-cache, сэмплирование из C25a, eos |
| проверка | логиты = исходной случайной модели | логиты = `transformers` на **настоящих** весах |

## Как устроена твоя модель

```shapes
idx [B, T]
> embed_tokens
x [B, T, 1024]
> 28 слоёв, в каждом:
> h = x + o_proj(Attn(RoPE(q_norm(q)), RoPE(k_norm(k)), v)) — GQA: 16 / 8 голов × 128, KV-cache
> x = h + down(silu(gate(norm(h))) ⊙ up(norm(h))) — SwiGLU 1024 → 3072
x [B, T, 1024]
> norm → lm_head (= embed_tokens)
logits [B, T, 151936]
```

Модель собирает `llmlab/core/model.py` из конфига `llmlab/configs/qwen3-0.6b.yaml` (каждое число сверено с
официальным `config.json`). Тот же код собирает GPT-2, Llama, Gemma 3-, OLMo 2-, DeepSeek- и гибридные модели —
посмотри `llmlab/configs/*.yaml` и `tests/test_core_model.py`, где доказано, что частные случаи общей
`Attention` совпадают с твоими задачами C04, C07, C09, C10, C19.

## Формула

Загрузка — это перенос тензоров без вычислений, поэтому формула здесь про то, **что должно сохраниться**.
Для каждого параметра $\theta$ своей модели и соответствующего тензора чекпойнта $W$ (имена — по C24):

$$
\theta = \operatorname{fp32}(W_{\text{bf16}}),\qquad
W_{\text{lm\_head}} \equiv E\ \ (\text{один тензор, а не копия}),\qquad
\big|\,z^{\text{моя}} - z^{\text{transformers}}\big|_\infty \le 10^{-3}
$$

Генерация с KV-cache: после prefill промпта $x_{1:T}$ каждый шаг подаёт в модель только новый токен, а
прошлые $K, V$ берёт из кеша (C05):

$$
z_{t} = f\big(x_t \mid \text{cache}_{<t}\big),\qquad
x_{t+1} = \texttt{sample\_next}(z_t;\ T, k, p),\qquad
x_{t+1} = \text{eos}, \text{ если строка уже выдала eos}
$$

## Формы тензоров

```shapes
## load_safetensors
- {"model.layers.0.self_attn.q_proj.weight": [2048, 1024] bf16, …} — 311 тензоров
## load_pretrained
- Model: 596 049 920 параметров fp32 (≈ 2.4 ГБ); lm_head.weight is embed_tokens.weight
## generate
idx [B, T]
> prefill: model(idx)
logits [B, T, V]
> шаг: model(next [B, 1], caches)
logits [B, 1, V]
> n ≤ max_new_tokens шагов
idx [B, T + n]
```

:::viz param-budget:::

:::viz sampling:::

## Задание

В `solution.py` (рабочая копия в `workspace/C25-qwen3-project/`) реализуй:

1. `load_safetensors(model_dir)` — словарь всех тензоров. Большие модели режут на шарды
   `model-00001-of-00002.safetensors`, список — в `model.safetensors.index.json` (`weight_map`).
2. `load_pretrained(model_dir, dtype=torch.float32)` — `config.json` → `config_from_hf` → `Model` → веса
   (`convert_state_dict`, `check_state_dict` из C24) → `eval()`. Связь `lm_head` ↔ `embed_tokens` должна
   остаться (если собираешь модель на `meta` и грузишь с `assign=True` — связь рвётся, восстанови).
3. `generate(model, idx, max_new_tokens, temperature=0.0, top_k=None, top_p=None, eos_token_id=None,
   generator=None)` — prefill, дальше по одному токену с кешем; выбор токена — `sample_next` из C25a (импортирован
   в заготовке); после `eos` строка дополняется `eos`; все строки закончили — выход.

## Как запустить на настоящих весах

```bash
uv sync --extra weights                       # huggingface_hub, tokenizers, transformers
llmlab test C25-qwen3-project --weights       # скачает Qwen3-0.6B (~1.5 ГБ) в ~/.cache/huggingface
llmlab generate qwen3-0.6b "The capital of France is" --impl workspace   # твоя модель пишет текст
```

Без `--weights` тесты с реальными весами пропускаются, остальные работают на игрушечном чекпойнте, который тест
пишет во временную папку в том же формате, что и Hugging Face.

**Допуск сравнения с `transformers`.** Обе модели в fp32, разница — только порядок суммирования в matmul,
softmax и RMSNorm (порядка $10^{-7}$ относительно на операцию, накопленная через 28 слоёв). На промпте из теста
фактически max $|\Delta| \approx 6 \cdot 10^{-5}$ при логитах до 21; тест требует $\le 10^{-3}$ и совпадения
argmax во всех позициях. Любая настоящая ошибка даёт расхождение порядка единиц. В bf16 так сравнивать нельзя:
одно только округление весов даёт $|\Delta| \sim 10^{-1}$.

## Бонус

Та же `load_pretrained` грузит **GPT-2** (LayerNorm, обучаемые позиции, Conv1D-веса) и **SmolLM2-135M**
(Llama: без QK-Norm, `head_dim = d / H`) — две другие архитектуры одним кодом. Тест бонуса тоже запускается
с `--weights`.

## Критерии готовности

- шарды по `index.json` читаются все;
- игрушечный Qwen3-чекпойнт (bf16, 2 шарда) даёт те же логиты, что исходная модель; `eval()`, fp32, связь
  `lm_head` сохранена; `dtype=bfloat16` работает; GPT-2-чекпойнт грузится той же функцией;
- жадная генерация == перепрогон префикса; после prefill в модель идёт по одному токену, лишних прогонов нет;
  eos останавливает генерацию;
- `--weights`: 596 049 920 параметров, логиты == `transformers`, «…Germany is» → «Berlin», жадная генерация
  совпадает с `transformers` токен в токен.

Запуск: `llmlab test C25-qwen3-project` (и с `--weights`).
