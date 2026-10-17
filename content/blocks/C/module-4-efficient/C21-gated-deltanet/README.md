# DeltaNet → Gated DeltaNet: правило дельты и гейт забывания

## Зачем это нужно

В C20 память линейного внимания — матрица $S = \sum_s \varphi(k_s) v_s^\top$. Она только **накапливает**:
если по похожему ключу пришло новое значение, старое не стирается, а складывается с новым. Чем длиннее
текст, тем больше в $S$ «шума» от старых пар, и извлечение по ключу деградирует.

**DeltaNet** (Schlag et al., 2021; параллельное обучение — Yang et al., 2024) смотрит на $S$ как на
ассоциативную память «ключ → значение» и обновляет её **правилом дельты** (правило Видроу — Хоффа): сначала
читает, что память сейчас отдаёт по ключу, $v^{\text{old}} = S^\top k_t$, и записывает только **разность**:

$$S_t = S_{t-1} + \beta_t\, k_t\,(v_t - S_{t-1}^\top k_t)^\top$$

Это ровно один шаг градиентного спуска по $\tfrac12\lVert S^\top k_t - v_t\rVert^2$ с шагом $\beta_t$: память
**перезаписывает** значение по ключу, а не наслаивает.

**Gated DeltaNet** (Yang, Kautz, Hatamizadeh, ICLR 2025) добавляет гейт забывания из Mamba-2: перед записью
память затухает в $\alpha_t = e^{g_t} \in (0, 1]$ раз. Гейт быстро стирает память целиком (смена темы,
новый документ), правило дельты аккуратно правит отдельные пары. Эти слои стоят в 36 из 48 слоёв Qwen3-Next.

## Что поменялось относительно C20

| | C20 (линейное внимание) | C21 (Gated DeltaNet) |
|---|---|---|
| запись в память | $S \mathrel{+}= \varphi(k) v^\top$ | $S \leftarrow \alpha S;\ S \mathrel{+}= \beta\, k\,(v - S^\top k)^\top$ |
| признаки q, k | $\varphi = \operatorname{elu} + 1$ | L2-нормировка ($\lVert k \rVert = 1$), запрос ещё $/\sqrt{d_k}$ |
| нормировка выхода | делить на $\varphi(q)^\top z$ | нет (её роль играет $\lVert k \rVert = 1$ и $\beta \le 1$) |
| забывание | нет | $\alpha_t = \exp(g_t)$, $g_t = -e^{A}\operatorname{softplus}(a_t + \Delta_{\text{bias}}) \le 0$ |
| сила записи | всегда 1 | $\beta_t = \sigma(b_t) \in (0, 1)$, своя у токена и головы |
| состояние | $S \in \mathbb{R}^{d_k \times d_v}$, $z \in \mathbb{R}^{d_k}$ | только $S \in \mathbb{R}^{d_k \times d_v}$ |

## Формула

На шаге $t$ для каждой головы (q, k уже нормированы, $\tilde q_t = q_t / \sqrt{d_k}$):

$$
S \leftarrow e^{g_t} S,\qquad
v^{\text{old}}_t = S^\top k_t,\qquad
S \leftarrow S + k_t \big(\beta_t (v_t - v^{\text{old}}_t)\big)^\top,\qquad
o_t = S^\top \tilde q_t
$$

В матричном виде: $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t k_t v_t^\top$. При $\lVert k_t\rVert = 1$ и
$\beta_t \in (0, 1)$ матрица $I - \beta_t k_t k_t^\top$ сжимает направление $k_t$ и не трогает остальные — память
устойчива. Два частных случая: $g = 0$ — DeltaNet; запись $k v^\top$ без дельты, но с гейтом — Mamba-2 (C23).

## Формы тензоров

```shapes
x [B, T, C]
fork
> q_proj, k_proj → l2norm по головам
q, k [B, H, T, dk]
branch
> v_proj
v [B, H, T, dv]
branch
> b_proj, a_proj
β, g [B, H, T] — β = σ(b_proj(x)), g = −e^{A_log}·softplus(a_proj(x) + dt_bias)
join

## шаг t
> S ← exp(g_t)·S — [B, H, dk, dv] · [B, H, 1, 1]
> v_old = Sᵀk_t — (S * k_t[..., None]).sum(-2) → [B, H, dv]
> S ← S + k_t ⊗ β_t(v_t − v_old) — [B, H, dk, 1] * [B, H, 1, dv]
S [B, H, dk, dv]
> o_t = Sᵀq_t / √dk
o_t [B, H, dv]

## выход
out [B, H, T, dv]
> → [B, T, H·dv] → o_proj
y [B, T, C]
axis dk = размер ключа
axis dv = размер значения
```

:::viz state-memory:::

## Задание

В `solution.py` (рабочая копия в `workspace/C21-gated-deltanet/`) реализуй:

1. `l2norm(x)`;
2. `gated_delta_rule(q, k, v, beta, g=None, state=None)` → `(out, state)`; `g=None` — DeltaNet без гейта;
3. `GatedDeltaNet.gates(x)` → `(beta, g)` и `GatedDeltaNet.forward(x, state=None)` → `(y, state)`.

Конструктор дан. Упрощения относительно Qwen3-Next: нет короткой причинной свёртки (`conv1d`, ядро 4) и SiLU
перед нормой, нет выходного гейта (`RMSNormGated(o) · SiLU(z)`), число голов q/k и v одинаковое. При обучении
используют **чанковую** форму (внутри куска — матричные операции, между кусками — рекуррентность); здесь только
рекуррентная, как при генерации.

## Реальные числа

Qwen3-Next-80B-A3B (`config.json`): `linear_num_key_heads = 16`, `linear_num_value_heads = 32` (q и k
повторяются на 2 головы значений), `linear_key_head_dim = linear_value_head_dim = 128`,
`linear_conv_kernel_dim = 4`. Состояние одного слоя — $32 \cdot 128 \cdot 128 = 524\,288$ чисел при **любой**
длине. Для сравнения: KV-cache full-слоя той же модели (2 KV-головы по 256) — $2 \cdot 2 \cdot 256 = 1024$ числа
на токен, то есть состояние Gated DeltaNet «весит» как KV-cache на 512 токенов.

## Критерии готовности

- `l2norm` даёт единичные векторы;
- рекуррентная форма совпадает с эталоном (без гейта и с гейтом);
- запись $(k, v_1)$, затем $(k, v_2)$ при $\beta = 1$ → по $k$ читается $v_2$ (не $v_1 + v_2$); ортонормированные
  ключи извлекаются точно; $\beta = 0.25$ сдвигает значение на четверть;
- $g = 0$ — то же, что DeltaNet; $g = -60$ на шаге $t$ стирает всё до $t$;
- кусками с переносом состояния — как целиком, будущее не протекает;
- слой: $\beta \in (0, 1)$, $g \le 0$ по формуле, выход совпадает с оракулом, состояние $[B, H, d_k, d_v]$.

Запуск: `llmlab test C21-gated-deltanet`.
