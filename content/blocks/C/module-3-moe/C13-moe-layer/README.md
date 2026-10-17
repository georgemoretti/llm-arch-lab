# MoE-слой: отправить каждый токен к его экспертам

## Зачем это нужно

Роутер из C12 сказал, каким $k$ экспертам достался каждый токен и с какими весами. Теперь нужно
**посчитать** MoE-слой. Самый очевидный способ — прогнать каждого эксперта на всех токенах и
занулить лишнее — правильный, но убивает весь смысл MoE: вычислений в $E/k$ раз больше, чем нужно
(у DeepSeek-V3 — в 32 раза).

Правильный способ — **диспетчеризация**: собрать для каждого эксперта только его токены, прогнать
их одним батчем, а результаты вернуть на места, сложив вклады $k$ экспертов каждого токена. Так
работают все реальные реализации — от цикла по экспертам в Hugging Face до специализированных
ядер (grouped GEMM) и распределения экспертов по разным GPU (expert parallelism), где «вернуть на
места» превращается в all-to-all обмен между устройствами.

## Что поменялось относительно C06

| | блок Llama (C06) | блок с MoE |
|---|---|---|
| FFN | `mlp(x)` — один SwiGLU для всех токенов | `Σ_k w · expert_k(x)` — свои $k$ из $E$ для каждого токена |
| параметры FFN | $3 C F$ | $E \cdot 3 C F_e + E C$ (эксперты + роутер) |
| FLOPs на токен | $\approx 2 \cdot 3 C F$ | $\approx 2 \cdot k \cdot 3 C F_e$ |
| выход слоя | тензор | тензор **и** логиты роутера (для aux loss в C14) |

Attention, нормы и остаточные связи не меняются: MoE заменяет только FFN.

## Формула

$$
y_n = \sum_{j=1}^{k} w_{n,j}\, \text{FFN}_{e_{n,j}}(x_n), \qquad (w_{n,\cdot}, e_{n,\cdot}) = \text{route}(W_r x_n)
$$

Через сортировку: разложим назначения в $N \cdot k$ «слотов» $(n, e, w)$, отсортируем слоты по
эксперту $e$. Тогда слоты каждого эксперта идут подряд, и

$$
y = \sum_{e=1}^{E} \operatorname{index\_add}\big(\text{tok}_e,\ w_e \odot \text{FFN}_e(x[\text{tok}_e])\big),
$$

где $\text{tok}_e$ — номера токенов в слотах эксперта $e$. Сложение (а не запись) обязательно:
каждый токен получает вклады от $k$ экспертов.

## Формы тензоров

```shapes
x [B, T, C]
> reshape
h [N, C] — N = B·T
> gate(h) — роутер
logits [N, E]
weights, indices [N, k]

## forward_naive: каждый эксперт на всех токенах
> dense = zeros(N, E).scatter_(1, indices, weights)
dense [N, E]
> out = Σ_e dense[:, e:e+1] · expert_e(h) — E прогонов по N строк
out [N, C]

## forward: каждый эксперт только на своих токенах
flat_expert [N·k] — indices.reshape(-1)
flat_token [N·k] — arange(N).repeat_interleave(k): 0, 0, 1, 1, 2, 2, … при k = 2
flat_weight [N·k] — weights.reshape(-1)
> order = flat_expert.argsort() — слоты, сгруппированные по эксперту
> counts = bincount(flat_expert, minlength=E) → [E], сумма = N·k
- для эксперта e: tok = flat_token[order[start:start+counts[e]]]
> y = expert_e(h[tok]) · w
y [n_e, C]
> out.index_add_(0, tok, y)
out [N, C]
> reshape
out [B, T, C]
axis n_e = сколько слотов досталось эксперту e (counts[e])
```

:::viz moe-routing:::

## Задание

В `solution.py` (рабочая копия в `workspace/C13-moe-layer/`) реализуй у `MoELayer`:

1. `forward_naive(x)` → `(out [B, T, C], router_logits [N, E])` — оракул: все эксперты на всех токенах;
2. `forward(x)` → то же самое, но эксперт видит только свои токены.

Конструктор дан: `self.gate` — роутер из C12 (`llmlab.core.moe.TopKRouter`), `self.experts` —
`nn.ModuleList` из `SwiGLUFFN` (имена `gate_proj/up_proj/down_proj` как у экспертов DeepSeek в HF).

## Критерии готовности

- оба метода совпадают с взвешенной суммой top-k экспертов; `forward == forward_naive` для top-1/3,
  sigmoid-роутера с ×2.5 и без перенормировки;
- эксперт получает ровно свои токены, всего обработано $N \cdot k$ строк;
- невыбранный эксперт не вызывается и не получает градиент, а роутер — получает;
- выход токена не зависит от батча.

Запуск: `llmlab test C13-moe-layer`.
