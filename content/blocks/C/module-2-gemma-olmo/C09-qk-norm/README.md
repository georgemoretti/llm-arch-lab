# C09. QK-Norm: нормировать запросы и ключи до RoPE

## Зачем это нужно

Логит attention — это $q \cdot k / \sqrt{D}$. Ничто в архитектуре не ограничивает его величину:
если во время обучения нормы $q$ и $k$ растут (а при большом learning rate и большой модели они
растут), логиты становятся огромными, softmax вырождается в one-hot, градиенты через него почти
нулевые или скачут — и loss внезапно расходится. Это наблюдали при масштабировании ViT до 22 млрд
параметров (Dehghani et al., 2023): без нормы на q и k обучение расходилось из-за роста логитов.

Лекарство простое: нормировать $q$ и $k$ перед скалярным произведением (Henry et al., 2020,
предложили это для машинного перевода). После RMSNorm длина вектора фиксирована — $\sqrt{D}$ при
единичных весах нормы, — поэтому $|q \cdot k| / \sqrt{D} \le \sqrt{D}$ при любом масштабе $W_Q$ и
$W_K$. «Температуру» softmax теперь задают только обучаемые веса нормы.

В 2024–2025 это стало стандартом: QK-Norm есть в OLMo 2, Gemma 3, Qwen3. Gemma 2 боролась с
большими логитами иначе — soft-capping, $50 \cdot \tanh(\text{logit}/50)$ (`attn_logit_softcapping = 50.0`
в config.json `google/gemma-2-9b`); в Gemma 3 soft-capping убрали (`attn_logit_softcapping = null`)
и поставили QK-Norm.

## Что поменялось относительно C04

| | C04 (GQA) | C09 (GQA + QK-Norm) |
|---|---|---|
| q, k | $W_Q x$, $W_K x$ (потом RoPE) | $\operatorname{RMSNorm}(W_Q x)$, $\operatorname{RMSNorm}(W_K x)$, **потом** RoPE |
| логиты | не ограничены | $\le \sqrt{D} \cdot \max|w_q| \cdot \max|w_k|$ |
| новые параметры | — | `q_norm.weight`, `k_norm.weight` |
| v | без изменений | без изменений — v не нормируется |

Два варианта QK-Norm в реальных моделях:

- **по голове** (Gemma 3, Qwen3): одна RMSNorm размера $D$ (`head_dim`), применяется к каждой голове
  после разбиения `[B, H, T, D]`; веса общие для всех голов;
- **по всей проекции** (OLMo 2): RMSNorm размера $H \cdot D$ на выходе `q_proj` (и $H_{kv} \cdot D$ на
  `k_proj`) до разбиения на головы. RMS считается по всем головам сразу, поэтому длина отдельной
  головы не фиксирована.

В обоих вариантах норма стоит **до RoPE**. Порядок важен: RoPE смешивает каналы $i$ и $i + D/2$
поворотом, а RMSNorm умножает каналы на **разные** обученные веса — эти операции не коммутируют. С
единичными весами (как при инициализации) разницы нет, поэтому ошибку в порядке легко не заметить,
пока не загрузишь обученные веса.

## Формула

Режим «по голове», для каждой головы $h$:

$$
\hat q_h = w_q \odot \frac{q_h}{\operatorname{RMS}(q_h)},\qquad
\hat k_h = w_k \odot \frac{k_h}{\operatorname{RMS}(k_h)},\qquad
\text{score}_{ij} = \frac{(R_i \hat q_{h,i}) \cdot (R_j \hat k_{h,j})}{\sqrt{D}}
$$

При $w = 1$: $\lVert \hat q \rVert = \lVert \hat k \rVert = \sqrt{D}$, поворот длину не меняет, и
$|\text{score}| \le D / \sqrt{D} = \sqrt{D}$.

## Формы тензоров

```
x                         [B, T, C]
q_proj(x)                 [B, T, H·D]      ── "full": q_norm (вес [H·D]) здесь
  view + transpose        [B, H, T, D]     ── "head": q_norm (вес [D]) здесь, по последней оси
k_proj(x)                 [B, T, H_kv·D]   ── то же с k_norm ([H_kv·D] или [D])
  view + transpose        [B, H_kv, T, D]
apply_rope(q), apply_rope(k)               ── после нормы
repeat_kv(k), repeat_kv(v) [B, H, T, D]
scores = q·kᵀ/√D          [B, H, T, T]     ← возвращает scores()
causal-маска, softmax, @ v, o_proj → [B, T, C]
```

## Задание

В `solution.py` (рабочая копия в `workspace/C09-qk-norm/`) реализуй у `QKNormAttention`:

1. `scores(x)` → `(scores [B, H, T, T], v [B, H, T, D])` — всё до маски, с QK-Norm в нужном месте
   для режима `self.qk_norm` (`"head"` или `"full"`);
2. `forward(x)` → `[B, T, C]` — causal-маска, softmax, `@ v`, `o_proj`.

Конструктор дан: `q_norm` и `k_norm` уже нужного размера для режима. `RMSNorm`, `apply_rope`,
`rope_cos_sin`, `repeat_kv` — из `llmlab.core.llama`.

## Критерии готовности

- оба режима совпадают с оракулом при случайных весах норм;
- порядок «норма → RoPE»: обратный порядок даёт другой выход, тест это проверяет;
- `scores` формы `[B, H, T, T]`; при единичных весах нормы `|score| ≤ √D` даже при `W_Q × 1000`;
- выход не меняется при умножении `W_Q` на 30 и `W_K` на 4;
- будущее не протекает.

Запуск: `llmlab test C09-qk-norm`.
