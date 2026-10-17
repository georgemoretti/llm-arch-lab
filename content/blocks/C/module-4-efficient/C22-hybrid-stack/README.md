# Гибридный стек 3 : 1: Gated DeltaNet + gated attention, память кеша

## Зачем это нужно

У двух семейств слоёв противоположные сильные стороны:

- **full attention** (C19) точно достаёт любой токен из прошлого, но его KV-cache растёт с каждым токеном;
- **линейные слои** (Gated DeltaNet, C21) держат состояние фиксированного размера, но вся история в нём
  сжата — точное извлечение фактов из далёкого прошлого даётся хуже.

Гибрид берёт оба: **большинство** слоёв линейные (дёшево), а **каждый $n$-й** — полный attention, который
«чинит» извлечение. Так устроены Qwen3-Next (3 Gated DeltaNet : 1 gated attention) и ряд других моделей
2025 года. Это та же идея, что local:global у Gemma 3 (C07), только «локальный» слой там — скользящее окно,
которое прошлое за окном не видит вообще, а здесь — линейный слой, который помнит всё, но приблизительно.

## Что поменялось относительно C11

| | C11 (Gemma 3: SWA + кольцевой KV-cache) | C22 (Qwen3-Next: гибрид) |
|---|---|---|
| «дешёвые» слои | sliding window, кеш $\min(S, W)$ токенов | Gated DeltaNet, состояние $H \cdot d_k \cdot d_v$ — не зависит от $S$ |
| что видят дешёвые слои | последние $W$ токенов | всё прошлое, сжатое в матрицу |
| соотношение | 5 sliding : 1 full | 3 linear : 1 full |
| расписание | слой $i$ full, если $(i+1) \bmod 6 = 0$ | слой $i$ full, если $(i+1) \bmod 4 = 0$ |
| память на длинном контексте | → $\tfrac16$ от «все full» | → $\tfrac14$ от «все full» |

## Формула

Расписание: $\text{type}(i) = \text{full}$, если $(i + 1) \bmod n = 0$, иначе linear ($i$ с нуля).

Блок — Pre-Norm (C08), одинаковый для обоих типов, меняется только mixer:

$$h = x + \operatorname{Mixer}_i(\operatorname{RMSNorm}(x)),\qquad y = h + \operatorname{FFN}(\operatorname{RMSNorm}(h))$$

Память инференса (batch $B$, $b$ байт на число):

$$
M(S) = B \cdot b \cdot \Big(\underbrace{N_{\text{full}} \cdot 2 H_{kv} d \cdot S}_{\text{KV-cache, растёт}} \;+\;
\underbrace{N_{\text{lin}} \cdot H_{\text{lin}} d_k d_v}_{\text{состояния, константа}}\Big)
$$

Гибрид выгоднее «все full», когда $N_{\text{lin}} \cdot 2H_{kv} d \cdot S > N_{\text{lin}} \cdot H_{\text{lin}} d_k d_v$, то есть при
$S > H_{\text{lin}} d_k d_v / (2 H_{kv} d)$. У Qwen3-Next это $32 \cdot 128 \cdot 128 / (2 \cdot 2 \cdot 256) = 512$ токенов.

## Формы тензоров

```shapes
x [B, T, C]

## слой i
> input_layernorm
fork
> full: GatedAttention(·)
[B, T, C]
branch
> linear: GatedDeltaNet(·) — возвращает ([B, T, C], state [B, H, dk, dv]), берём первый элемент
[B, T, C]
join x + mixer(…)
x [B, T, C]
> x + mlp(post_attention_layernorm(x))
x [B, T, C]

## после всех слоёв
> norm
x [B, T, C]

## память на слой
- full-слой: 2 · H_kv · d · S чисел
- linear-слой: H_lin · dk · dv чисел
```

## Задание

В `solution.py` (рабочая копия в `workspace/C22-hybrid-stack/`) реализуй:

1. `hybrid_layer_types(n_layers, full_attention_interval)` → список строк `"full_attention"` /
   `"linear_attention"` (как `layer_types` в HF);
2. `HybridBlock.forward(x)`;
3. `HybridStack.forward(x)`;
4. `hybrid_cache_bytes(layer_types, seq_len, n_kv_heads, head_dim, linear_heads, linear_k_dim, linear_v_dim, batch=1, bytes_per_elem=2)`.

Конструкторы даны; `GatedAttention` и `GatedDeltaNet` — эталоны C19 и C21 из `llmlab.core.efficient`.
Генерации с кешем в стеке нет: память считается формулой. Упрощения: в Qwen3-Next FFN — MoE (512 экспертов,
10 активных + shared, модуль 3), у Gated DeltaNet есть conv1d — её состояние ($(k - 1) \cdot$ каналов) не учитываем.

## Реальные числа

Qwen3-Next-80B-A3B (`config.json`): 48 слоёв, `full_attention_interval = 4` → 12 full + 36 linear. Full: 2 KV-головы
× 256. Linear: 32 головы значений × 128 × 128. Посмотри на полосу слоёв и кривые памяти:

:::viz hybrid:::

## Критерии готовности

- расписание Qwen3-Next: full в слоях 3, 7, …, 47; `interval = 1` → все full;
- mixer-ы стека по слоям: linear, linear, linear, full, …;
- блок совпадает с Pre-Norm формулой для обоих mixer-ов (у Gated DeltaNet в residual идёт только `y`);
- стек = блоки по порядку + финальная норма, будущее не протекает;
- память: формула; у Qwen3-Next прирост на 1000 токенов — только от 12 full-слоёв, на 512 токенах гибрид = «все full»,
  на 100 — дороже, на 262 144 — ≈ ¼.

Запуск: `llmlab test C22-hybrid-stack`.
