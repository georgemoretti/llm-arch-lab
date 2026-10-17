# MLA с decoupled RoPE и размер его кеша

## Зачем это нужно

В C16 у MLA не было позиционного кодирования. Добавить RoPE «как обычно» — повернуть ключи
$k_h = W_{UK}^{(h)} c$ на угол позиции — нельзя без потери главного трюка: поглощения весов.
Поглощение опирается на то, что $q^\top W_{UK} c$ — это $(W_{UK}^\top q)^\top c$ с **одной и той же**
матрицей для всех позиций. С RoPE между ними встаёт поворот, зависящий от позиции ключа:
$q^\top R_m W_{UK} c_m$ — матрицу больше нельзя один раз свернуть в запрос, и пришлось бы
восстанавливать ключи всего контекста на каждом шаге.

Решение DeepSeek-V2 — **decoupled RoPE**: голова делится на две части.

- **Без позиций** ($d_n$ измерений): $k^{\text{nope}}_h = W_{UK}^{(h)} c$ — из латента, поглощается как в C16.
- **С RoPE** ($d_r$ измерений): отдельный маленький ключ $k^{\text{rope}} = R_m W_{KR} x$, **один на все
  головы** (как в MQA — отсюда имя `kv_a_proj_with_mqa`), и своя RoPE-часть запроса у каждой головы.

Скалярное произведение — сумма двух частей: позиционно-независимой (через латент) и позиционной
(через общий маленький ключ). В кеш идут латент $c$ ($r$ чисел) и повёрнутый $k^{\text{rope}}$ ($d_r$
чисел). У DeepSeek-V3 это $512 + 64 = 576$ чисел на токен и слой.

Выбери в калькуляторе пресет «DeepSeek-V3 (MLA)» и сравни с MHA и MQA:

:::viz kv-cache:::

## Что поменялось относительно C16

| | C16 | C16b |
|---|---|---|
| голова ключа | $d$ (из латента) | $d_n$ из латента **+** $d_r$ общий RoPE-ключ |
| запрос | `q_proj` → $H \cdot d$ | `q_proj` → $H(d_n + d_r)$, RoPE только на последних $d_r$ |
| сжатие | `kv_a_proj`: $C \to r$ | `kv_a_proj_with_mqa`: $C \to r + d_r$ (латент и RoPE-ключ одной матрицей) |
| кеш | $c$: $r$ на токен | $c$ и $k^{\text{rope}}$: $r + d_r$ на токен |
| масштаб | $1/\sqrt{d}$ | $1/\sqrt{d_n + d_r}$ |

У DeepSeek-V3 ещё есть YaRN-масштабирование RoPE (`rope_scaling.factor = 40`, контекст 4K → 160K) и
поправка масштаба `mscale` — в задаче их нет.

## Формула

$$
q_h = [\,q^{\text{nope}}_h ;\ R_i\, q^{\text{rope}}_h\,],\qquad
k_{h,j} = [\,W_{UK}^{(h)} c_j ;\ R_j\, k^{\text{rope}}_j\,],\qquad
[c_j ; k^{\text{rope}}_j] = [\operatorname{norm}(\cdot) ; \cdot\,](W_{DKV}^{\text{mqa}} x_j)
$$

$$
\text{score}_h(i, j) = \frac{q^{\text{nope}\top}_{h,i} W_{UK}^{(h)} c_j + (R_i q^{\text{rope}}_{h,i})^\top (R_j k^{\text{rope}}_j)}{\sqrt{d_n + d_r}}
$$

Первое слагаемое поглощается (C16), второе — обычный RoPE (C03) с одним ключом на все головы.

Память кеша (в элементах, на слой и токен): MHA $= 2 H d$, GQA $= 2 H_{kv} d$, MLA $= r + d_r$ —
без множителя 2: и K, и V восстанавливаются из одного латента.

## Формы тензоров

```shapes
x [B, T, C]
fork
> q_proj → view → transpose
q [B, H, T, dn + dr]
> split
q_nope [B, H, T, dn]
q_rope [B, H, T, dr] — RoPE, позиции start … start + T − 1
branch
> kv_a_proj_with_mqa
[B, T, r + dr]
> split
c [B, T, r] — после norm
k_rope [B, T, dr] — RoPE как одна «голова»
> cache.update(c, k_rope)
c [B, S, r]
k_rope [B, S, dr]
> kv_b_proj(c)
k_nope [B, H, S, dn]
v [B, H, S, dv]
> cat(k_nope, k_rope[:, None].expand(B, H, S, dr))
k [B, H, S, dn + dr]
join q·kᵀ / √(dn + dr) + causal_mask(T, S)
scores [B, H, T, S]
> softmax, @ v, o_proj
out [B, T, C]
axis r:other = размер латента KV
axis dn = часть головы без RoPE
axis dr = часть головы с RoPE
axis dv = размер головы v
```

## Задание

В `solution.py` (рабочая копия в `workspace/C16b-mla-rope/`) реализуй:

1. `MLARopeAttention.forward(x, cache=None)` — конструктор дан (имена как в HF);
2. `mla_kv_cache_bytes(n_layers, kv_lora_rank, qk_rope_head_dim, seq_len, batch=1, bytes_per_elem=2)`.

`LatentCache` и `causal_mask` — из `llmlab.core.mla` (эталон C16), `rope_cos_sin`, `apply_rope` — из
`llmlab.core.llama`.

## Критерии готовности

- без кеша совпадает с оракулом; будущее не протекает;
- в кеше латент `[B, S, r]` и повёрнутый общий RoPE-ключ `[B, S, dr]`;
- по одному токену и кусками — как полный прогон;
- DeepSeek-V3: 61 · 576 · 2 байта = 70 272 байта на токен (≈ 69 КиБ) — в ~71 раз меньше MHA с теми же головами.

Запуск: `llmlab test C16b-mla-rope`.
