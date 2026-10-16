# C15. DeepSeekMoE: shared expert и первые dense-слои

## Зачем это нужно

В C13 каждый эксперт — полноценный FFN, и токен выбирает $k$ из них. DeepSeekMoE (Dai et al., 2024)
заметил две проблемы такой схемы:

1. **Общее знание дублируется.** Какие-то преобразования нужны почти каждому токену (частые
   слова, базовый синтаксис). При обычном MoE их выучивает понемногу каждый эксперт — параметры
   тратятся на копии. Решение — **shared experts**: один или несколько экспертов, через которых
   проходит **каждый** токен, без роутера. Routed-экспертам остаётся специализация.
2. **Эксперты слишком крупные.** Если вместо 8 широких экспертов взять 64 узких (fine-grained) и
   выбирать 8 вместо 1, вычисления те же, а комбинаций экспертов на порядки больше — специализация
   тоньше. Отсюда 256 узких экспертов шириной 2048 в DeepSeek-V3 (у dense-слоя той же модели ширина
   18 432).

Ещё одна деталь: **первые слои — dense**. В DeepSeek-V3 первые 3 из 61 слоя имеют обычный FFN.
Ранние слои работают с почти «сырыми» эмбеддингами, роутить там нечего, а нестабильность роутера
в начале сети особенно вредна.

Итог для DeepSeek-V3: из 671 млрд параметров на токен активно около 37 млрд. В этой задаче ты
посчитаешь, откуда берётся основная часть этих чисел.

## Что поменялось относительно C13

| | MoE (C13) | DeepSeekMoE (C15) |
|---|---|---|
| эксперты | $E$ routed, top-$k$ | $E$ routed (top-$k$) **+** $S$ shared (всегда) |
| ширина эксперта | любая | узкая: `moe_intermediate_size` ≪ `intermediate_size` |
| все слои | MoE | первые `first_k_dense_replace` — dense FFN, дальше MoE |
| выход | $\sum_{\text{top-}k} w_e\,\text{FFN}_e(x)$ | $\sum_{\text{top-}k} w_e\,\text{FFN}_e(x) + \sum_{s} \text{FFN}^{\text{sh}}_s(x)$ |

У Gemma 3, OLMo 2 и SmolLM3 MoE нет — все слои dense; Qwen3-MoE обходится без shared experts и без
dense-слоёв в начале.

## Формула

$$
y = \underbrace{\sum_{s=1}^{S} \text{FFN}^{\text{sh}}_s(x)}_{\text{каждый токен}} \;+\;
\underbrace{\sum_{j=1}^{k} w_j(x)\, \text{FFN}_{e_j(x)}(x)}_{\text{роутер}}
$$

В Hugging Face $S$ shared-экспертов хранятся как один SwiGLU шириной $S \cdot F_e$ — это одно и то же
(сумма $S$ независимых SwiGLU = один SwiGLU с конкатенированными скрытыми нейронами).

Параметры FFN-части (без bias, SwiGLU = $3 C F$):

$$
\text{всего} = L_d\cdot 3CF_d + L_m\big((E + S)\cdot 3CF_e + EC\big),\qquad
\text{активно} = L_d\cdot 3CF_d + L_m\big((k + S)\cdot 3CF_e + EC\big)
$$

($L_d$ — dense-слоёв, $L_m$ — MoE-слоёв, $EC$ — роутер.)

## Формы тензоров

```
x                         [B, T, C]
self.routed(x)            → out_r [B, T, C], logits [N, E]       (MoE из C13)
self.shared_experts(x)    → [B, T, C]                            (SwiGLU шириной S·F_e)
out = out_r + shared      [B, T, C]

mlp_layer_types(61, 3)    → ["dense"]×3 + ["moe"]×58
build_ffn_layers(...)     → nn.ModuleList[SwiGLUFFN | DeepSeekMoE]
```

## Задание

В `solution.py` (рабочая копия в `workspace/C15-shared-expert/`) реализуй:

1. `mlp_layer_types(n_layers, first_k_dense_replace, moe_layer_freq=1)`;
2. `DeepSeekMoE.forward(x)` → `(out, router_logits)` — конструктор дан (`self.routed` — MoE из C13,
   `self.shared_experts` — SwiGLU или `None`);
3. `build_ffn_layers(...)` → `nn.ModuleList`;
4. `moe_param_counts(...)` → `(всего, активно)`.

## Критерии готовности

- расписание: DeepSeek-V3 — 3 dense + 58 MoE, V2-Lite — 1 dense, `moe_layer_freq` учитывается;
- выход = routed + shared; при нулевых routed-экспертах — ровно shared(x); без shared — MoE из C13;
- `build_ffn_layers` строит нужные типы и ширины; `moe_param_counts` совпадает с подсчётом параметров
  построенной модели;
- DeepSeek-V3: FFN ≈ 657.8B всего / 24.3B активно; Qwen3-30B-A3B: ≈ 29.0B / 1.82B.

Запуск: `llmlab test C15-shared-expert`.
