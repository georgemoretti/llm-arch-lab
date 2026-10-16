# C07. Sliding window attention и паттерн local:global

## Зачем это нужно

В C04–C05 мы считали KV-cache: он растёт линейно с длиной контекста, а attention на каждый новый
токен читает **весь** кеш. На контексте 128K это и память, и время: у модели масштаба Gemma 3 27B
полный кеш на 128K токенов занимает десятки гигабайт.

Наблюдение, на котором держится sliding window attention (SWA): большинству слоёв далёкий контекст
не нужен — им хватает соседних токенов (синтаксис, локальные зависимости). Если запрос видит только
последние $W$ позиций, то:

- attention стоит $O(T \cdot W)$ вместо $O(T^2)$;
- в KV-cache слоя достаточно держать $W$ последних токенов — память перестаёт расти с контекстом
  (это задача C11).

Цена — слой не может «дотянуться» дальше окна. Есть два способа это компенсировать:

1. **Стек слоёв.** Каждый слой SWA расширяет поле зрения на $W - 1$ позиций: выход слоя 1 в
   позиции $t$ уже содержит информацию о $t - W + 1$, а слой 2 смотрит на этот выход. $L$ слоёв
   видят на $L(W-1)$ позиций назад. Так работал Mistral 7B: SWA с окном 4096 во **всех** 32 слоях.
2. **Перемешать local и global.** Часть слоёв оставить с полным attention. Gemma 2 чередовала
   sliding и full 1:1 (окно 4096); Gemma 3 сдвинула пропорцию к **5 sliding : 1 full** и уменьшила
   окно до 1024. Глобальные слои отвечают за дальние зависимости, а их KV-cache — основная часть
   памяти.

:::viz masks:::

## Что поменялось относительно C05

| | C05 (`LlamaAttention`) | C07 (`SlidingWindowAttention`) |
|---|---|---|
| маска | causal: $j \le p$ | causal **и** окно: $p - W < j \le p$ |
| поле зрения слоя | вся история | последние $W$ токенов |
| слои модели | все одинаковые | расписание: `sliding_attention` / `full_attention` |
| параметры | — | `window` (и у Gemma 3 — свой θ RoPE для local/global) |

Веса, RoPE и GQA не меняются вообще. SWA — это **только маска** (и, как следствие, другой KV-cache).

## Формула

Запрос на позиции $p$ видит ключ на позиции $j$, если

$$
M_{p,j} = \begin{cases} 0, & p - W < j \le p \\ -\infty, & \text{иначе} \end{cases}
\qquad\Longleftrightarrow\qquad 0 \le p - j < W
$$

Окно из $W$ позиций **включает сам токен**: при $W = 3$ токен 5 видит 3, 4, 5. Так считают окно и
в Hugging Face (`kv_idx > q_idx - sliding_window`). При $W \ge T$ маска совпадает с causal.

Расписание слоёв Gemma (индексы с 0): слой $i$ — full, если $(i + 1) \bmod \text{pattern} = 0$,
иначе sliding. При pattern = 6 full-слои — 5, 11, 17, …; при pattern = 2 — чередование, начиная
со sliding.

Пример маски $T = 6$, $W = 3$:

```
      k0 k1 k2 k3 k4 k5
q0     1  ·  ·  ·  ·  ·
q1     1  1  ·  ·  ·  ·
q2     1  1  1  ·  ·  ·
q3     ·  1  1  1  ·  ·
q4     ·  ·  1  1  1  ·
q5     ·  ·  ·  1  1  1
```

## Формы тензоров

```
x                      [B, T, C]
q                      [B, H, T, D]      → RoPE (позиции 0..T−1)
k, v                   [B, H_kv, T, D]   → RoPE на k → repeat_kv → [B, H, T, D]
scores = q·kᵀ/√D       [B, H, T, T]
mask = sliding_window_mask(T, window=W)   [T, T]   (bool, True = можно)
out = softmax(scores + mask) · v          [B, H, T, D] → [B, T, H·D] → o_proj → [B, T, C]

sliding_window_mask(T, S, W): запросы — последние T из S позиций → [T, S]
  q_pos = arange(S − T, S)[:, None]    k_pos = arange(S)[None, :]
```

## Реальные модели

| модель | слоёв | окно | sliding : full | источник |
|---|---|---|---|---|
| Mistral 7B v0.1 | 32 | 4096 | все sliding | `mistralai/Mistral-7B-v0.1` config.json: `sliding_window = 4096` |
| Gemma 2 9B | 42 | 4096 | 1 : 1 | `google/gemma-2-9b` config.json: `layer_types` чередуются |
| Gemma 3 1B | 26 | 512 | 5 : 1 | `google/gemma-3-1b-pt` config.json: `sliding_window = 512`, `sliding_window_pattern = 6` |
| Gemma 3 27B | 62 | 1024 | 5 : 1 | `google/gemma-3-27b-pt` config.json (text_config): `sliding_window = 1024`, `sliding_window_pattern = 6` |

Конфиги Gemma на Hugging Face закрыты лицензионным соглашением; числа сверены по открытым зеркалам
`unsloth/gemma-2-9b`, `unsloth/gemma-3-1b-pt`, `unsloth/gemma-3-27b-pt`. У Gemma 3 sliding- и
full-слои ещё и используют разное основание RoPE: `rope_local_base_freq = 10 000` и
`rope_theta = 1 000 000` — далёкие позиции нужны только глобальным слоям. В этой задаче θ один.

## Задание

В `solution.py` (рабочая копия в `workspace/C07-swa/`) реализуй:

1. `sliding_window_mask(T, S=None, window=None)` — bool-маску `[T, S]`; запросы — последние `T`
   из `S` позиций (при `S > T` это пригодится для кеша в C11), `window=None` — обычный causal;
2. `layer_types(n_layers, sliding_window_pattern)` — список `"sliding_attention"` /
   `"full_attention"` по правилу Gemma;
3. `SlidingWindowAttention.forward(x)` — GQA + RoPE, как `LlamaAttention` из C05 без кеша, но с
   маской-окном. Конструктор дан.

`rope_cos_sin`, `apply_rope`, `repeat_kv` импортируются из `llmlab.core.llama`.

## Критерии готовности

- маска: окно из `W` позиций включая себя, causal сохраняется, `W = 1` → диагональ, `W ≥ T` и
  `None` → causal, сдвиг при `S > T`;
- расписание Gemma 3 27B: 10 full-слоёв на индексах 5, 11, …, 59; Gemma 2 чередует, начиная со sliding;
- attention совпадает с оракулом; при `window ≥ T` — ровно `LlamaAttention`;
- выход в позиции `t` не зависит от токенов старше `t − W + 1`; три слоя SWA с остатком видят на
  `3·(W − 1)` позиций назад.

Запуск: `llmlab test C07-swa`.
