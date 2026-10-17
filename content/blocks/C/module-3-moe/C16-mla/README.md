# MLA: латентный KV-cache и поглощение весов

## Зачем это нужно

KV-cache растёт как $2 \cdot L \cdot H_{kv} \cdot d \cdot S$ (C04). GQA и MQA уменьшают его, **выбрасывая**
головы ключей и значений: несколько голов запросов делят одну KV-голову. Это работает, но качество
немного проседает, особенно у MQA.

Multi-head Latent Attention (DeepSeek-V2, 2024) идёт другим путём: **сжимает**. Каждый токен
проецируется в маленький латентный вектор $c$ размера $r$ (у DeepSeek-V3 $r = 512$), и в кеш
кладётся только он. Ключи и значения **всех** голов восстанавливаются из $c$ линейной проекцией:
у каждой головы по-прежнему свои $k$ и $v$, как в MHA, но все они лежат в одном $r$-мерном
подпространстве. Авторы DeepSeek-V2 сообщают, что MLA не уступает MHA по качеству при кеше,
сравнимом с GQA с небольшим числом групп.

Вторая идея — **поглощение весов** (weight absorption). Восстанавливать K и V для каждого шага
генерации не обязательно: $q^\top k = q^\top W_{UK} c = (W_{UK}^\top q)^\top c$. Матрицу $W_{UK}$
можно «перенести» в запрос, а $W_{UV}$ — в выходную проекцию, и тогда attention считается прямо по
латентам из кеша. Именно так MLA эффективен при инференсе.

## Что поменялось относительно C05

| | C05 (`LlamaAttention` + `KVCache`) | C16 (`MLAttention` + `LatentCache`) |
|---|---|---|
| K, V | $W_K x$, $W_V x$ — по $H_{kv}$ голов | $c = \operatorname{norm}(W_{DKV} x)$, затем $k_h = W_{UK}^{(h)} c$, $v_h = W_{UV}^{(h)} c$ для всех $H$ голов |
| в кеше на токен и слой | $2 \cdot H_{kv} \cdot d$ чисел | $r$ чисел (латент) |
| параметры | `k_proj`, `v_proj` | `kv_a_proj` ($C \to r$), `kv_a_layernorm`, `kv_b_proj` ($r \to H(d + d_v)$) |
| позиции | RoPE | **нет** в этой задаче (как NoPE из C10); RoPE несовместим с поглощением — это C16b |

У DeepSeek-V3 ещё и запрос идёт через свой латент (`q_lora_rank = 1536`) — это экономит параметры
и активации при обучении, но не кеш; в задаче запрос считается одной матрицей, как у DeepSeek-V2-Lite
(`q_lora_rank = null`).

## Формула

Для токена $x$ и головы $h$:

$$
c = \operatorname{RMSNorm}(W_{DKV}\, x) \in \mathbb{R}^{r},\qquad
k_h = W_{UK}^{(h)} c \in \mathbb{R}^{d},\qquad
v_h = W_{UV}^{(h)} c \in \mathbb{R}^{d_v},\qquad
q_h = W_Q^{(h)} x
$$

$$
\text{score}_{h}(i, j) = \frac{q_{h,i}^\top k_{h,j}}{\sqrt{d}} = \frac{(W_{UK}^{(h)\top} q_{h,i})^\top c_j}{\sqrt{d}},
\qquad
o_{h,i} = \sum_j a_{h,ij}\, v_{h,j} = W_{UV}^{(h)} \sum_j a_{h,ij}\, c_j
$$

Справа — поглощённая форма: запрос переводится в латентное пространство ($\tilde q = W_{UK}^\top q$,
размер $r$), веса внимания применяются к латентам, и только в конце результат проецируется в
$d_v$. `kv_b_proj.weight` имеет форму $[H(d + d_v), r]$: после `view(H, d + d_v, r)` первые $d$
строк каждой головы — это $W_{UK}^{(h)}$, следующие $d_v$ — $W_{UV}^{(h)}$.

## Формы тензоров

```shapes
x [B, T, C]
fork
> q_proj → view → transpose
q [B, H, T, d]
branch
> kv_a_proj → norm
c [B, T, r] — латент: именно он идёт в кеш
> cache.update по dim=1
c [B, S, r]
join

## forward: распаковать K и V из латента
> kv_b_proj(c)
[B, S, H·(d + dv)]
> view → transpose
[B, H, S, d + dv]
> split по последней оси на d и dv
k [B, H, S, d]
v [B, H, S, dv]
> q·kᵀ/√d + маска со сдвигом S − T
scores [B, H, T, S]
> softmax, @ v
out [B, H, T, dv]
> → [B, T, H·dv] → o_proj
y [B, T, C]

## forward_absorbed: внимание прямо в латенте
- W = kv_b_proj.weight.view(H, d + dv, r)
W_uk [H, d, r] — первые d строк каждой головы
W_uv [H, dv, r] — остальные dv строк
> q_lat = einsum("bhtd,hdr->bhtr", q, W_uk)
q_lat [B, H, T, r]
> q_lat · cᵀ / √d — c как [B, 1, S, r]
scores [B, H, T, S]
> softmax(scores) · c
o_lat [B, H, T, r]
> einsum("bhtr,hvr->bhtv", o_lat, W_uv)
out [B, H, T, dv]
axis r:other = размер латента KV (kv_lora_rank)
axis d = размер головы q/k
axis dv = размер головы v
```

:::viz kv-cache:::

## Задание

В `solution.py` (рабочая копия в `workspace/C16-mla/`) реализуй:

1. `LatentCache.update(c, k_rope=None)` → `(все латенты [B, S, r], все k_rope или None)`;
2. `MLAttention.forward(x, cache=None)`;
3. `MLAttention.forward_absorbed(x, cache=None)` — без вызова `kv_b_proj(...)`.

Конструктор `MLAttention` дан (имена как в HF DeepSeek, без RoPE-части).

## Критерии готовности

- `forward` совпадает с оракулом; будущее не протекает;
- в кеше — нормированный латент `[B, S, r]`; по одному токену и кусками — как полный прогон;
- с `k_rope` кеш копит и его (`[B, S, d_r]`, по оси 1) — он понадобится в C16b;
- `forward_absorbed` совпадает с `forward` (с кешем и без) и не вызывает `kv_b_proj`.

Запуск: `llmlab test C16-mla`.
