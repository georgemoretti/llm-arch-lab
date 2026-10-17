# C08. Куда ставить нормализацию: Post-Norm, Pre-Norm, OLMo 2, sandwich

## Зачем это нужно

Блок трансформера — это два подслоя (attention и FFN), каждый обёрнут в остаточное соединение.
Нормализация нужна, чтобы активации не разъезжались по масштабу, но **где** её поставить —
отдельное решение, и за семь лет оно менялось четыре раза:

1. **Post-Norm** (оригинальный Transformer, 2017): нормируем сумму `x + подслой(x)`. Остаточный
   поток после каждого блока имеет RMS = 1. Проблема — обучение глубоких моделей нестабильно без
   долгого warmup learning rate.
2. **Pre-Norm** (GPT-2, Llama — блок из C06): нормируем только вход подслоя, остаток идёт мимо
   нормы. Xiong et al. (2020) показали, почему это стабильнее: при инициализации градиенты весов
   верхних слоёв у Pre-Norm уменьшаются с глубиной примерно как $1/\sqrt{L}$, а у Post-Norm остаются
   большими при любой глубине. Цена — остаточный поток не нормирован и растёт от слоя к слою.
3. **OLMo 2** (2024): норма на **выходе** подслоя, но внутри остаточной ветки: `x + n(подслой(x))`.
   Остаток по-прежнему идёт мимо нормы (как у Pre-Norm), а вклад каждого подслоя ограничен по
   масштабу. Авторы OLMo 2 описывают это как изменение ради стабильности обучения (вместе с
   QK-Norm — задача C09).
4. **Sandwich** (Gemma 2 и Gemma 3): нормы и на входе, и на выходе подслоя: `x + n(подслой(n(x)))`.

В этой задаче ты соберёшь все четыре и проверишь их свойства численно — включая эффект Xiong et al.

## Что поменялось относительно C06

| | формула подслоя | нормы на подслой | имена в HF |
|---|---|---|---|
| Post-Norm (2017) | $x \leftarrow n(x + f(x))$ | 1 | `post_attention_layernorm`, `post_feedforward_layernorm` (имена условные) |
| Pre-Norm (C06, Llama) | $x \leftarrow x + f(n(x))$ | 1 | `input_layernorm`, `post_attention_layernorm` (это норма **перед** FFN) |
| OLMo 2 | $x \leftarrow x + n(f(x))$ | 1 | `post_attention_layernorm`, `post_feedforward_layernorm` |
| Sandwich (Gemma 2/3) | $x \leftarrow x + n_2(f(n_1(x)))$ | 2 | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` |

Attention и FFN в задаче не меняются: `LlamaAttention` и `SwiGLUFFN` из модуля 1. Меняется только порядок
трёх операций: норма, подслой, сложение с остатком.

FFN у этих моделей и правда почти тот же, что у Llama. OLMo 2 — ровно SwiGLU (`hidden_act: silu`, 4096 → 11 008
у 7B). Gemma 3 — тот же гейтированный FFN из трёх матриц (`gate_proj`, `up_proj`, `down_proj`), но гейт —
GELU с tanh-аппроксимацией вместо SiLU (`hidden_activation: gelu_pytorch_tanh`, 5376 → 21 504 у 27B): это
**GeGLU** из той же статьи Шазира, что и SwiGLU (C02). В конфигурируемой модели финального проекта это одна
строка: `ffn: {type: geglu}` (`llmlab/configs/gemma3-like.yaml`).

Детали реальных реализаций, которых нет в задаче: у Gemma `RMSNorm` умножает на $(1 + w)$, а не на
$w$ (веса инициализируются нулями); у Gemma 3 и OLMo 2 в attention есть ещё QK-Norm (C09). У OLMo 2
и Gemma, как у Llama, есть финальная норма перед `lm_head`.

## Формула

Для одного подслоя $f$ (attention или FFN), $n$ — RMSNorm:

$$
\begin{aligned}
\text{Post:}\quad & x \leftarrow n(x + f(x)) \\
\text{Pre:}\quad & x \leftarrow x + f(n(x)) \\
\text{OLMo 2:}\quad & x \leftarrow x + n(f(x)) \\
\text{Sandwich:}\quad & x \leftarrow x + n_2(f(n_1(x)))
\end{aligned}
$$

Почему у OLMo 2 и sandwich остаток растёт как $\sqrt{1 + 2L}$: каждый подслой прибавляет вектор с
RMS ≈ 1 (выход нормы), почти ортогональный остатку, — квадрат RMS растёт на 1 за подслой, а
подслоёв в блоке два. У Post-Norm RMS всегда 1. У Pre-Norm выходы подслоёв не нормированы и при
инициализации малы, поэтому остаток растёт медленнее.

## Формы тензоров

```
x                       [B, T, C]
n(x)                    [B, T, C]    RMSNorm по последней оси, вес [C]
self.self_attn(·)       [B, T, C] → [B, T, C]
self.mlp(·)             [B, T, C] → [B, T, C]
выход блока             [B, T, C]
```

Формы во всех четырёх вариантах одинаковые — разница только в том, какой тензор нормируется и к
чему прибавляется.

## Задание

В `solution.py` (рабочая копия в `workspace/C08-norm-placement/`) реализуй `forward` у четырёх
классов: `PostNormBlock`, `PreNormBlock`, `OlmoPostNormBlock`, `SandwichNormBlock`.
Конструкторы даны: подслои `self.self_attn`, `self.mlp` и нормы с именами из таблицы уже созданы.
Порядок подслоёв везде один: сначала attention, потом FFN.

## Критерии готовности

- каждый блок совпадает со своей формулой (в тестах у всех норм разные веса — видно, какая где);
- с нулевыми `o_proj` и `down_proj` Pre / OLMo 2 / sandwich — тождество, Post-Norm — нормирует вход;
- RMS остатка после 16 блоков: Post-Norm = 1, OLMo 2 и sandwich ≈ $\sqrt{33} ≈ 5.7$, Pre-Norm — между;
- градиент весов последнего блока при Xavier-инициализации: у Post-Norm не зависит от глубины
  (L = 1 и L = 16), у Pre-Norm падает больше чем вдвое.

Запуск: `llmlab test C08-norm-placement`.
