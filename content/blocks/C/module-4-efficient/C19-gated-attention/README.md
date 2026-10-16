# C19. Gated attention: выходной гейт и partial RoPE (Qwen3-Next)

## Зачем это нужно

У обычного attention две известные странности.

**Attention sink.** Softmax обязан распределить вес 1 между ключами — даже если голове в этом месте
«нечего сказать». Модели учатся сбрасывать лишний вес на первый токен (BOS): он превращается в «сток»,
получает огромную долю внимания и аномально большие активации. Это мешает квантованию, длинному
контексту и делает обучение менее стабильным.

**Линейность пути value → выход.** Внутри головы $v = W_V x$ и $o = W_O(\sum a\,v)$ — две матрицы подряд
без нелинейности между ними, то есть по сути одно низкоранговое линейное отображение.

Qwen (Qiu et al., NeurIPS 2025) перебрали десятки мест и форм гейтов и нашли простой лучший вариант:
**поэлементный сигмоидный гейт на выходе attention, свой у каждой головы и у каждого запроса**:

$$o = \operatorname{Attn}(x) \odot \sigma(W_g x)$$

Гейт добавляет нелинейность между $W_V$ и $W_O$ и, что важнее, позволяет голове **промолчать**: $\sigma \to 0$
гасит её вклад без всякого стока. По данным авторов, гейтированные модели не образуют attention sink,
обучаются стабильнее (терпят больший learning rate) и лучше экстраполируют на длинный контекст. Этот
вариант стоит в full-attention слоях Qwen3-Next.

**Partial RoPE.** В тех же слоях RoPE применяется лишь к **четверти** каналов головы
(`partial_rotary_factor = 0.25`): 64 из 256 каналов знают позицию, остальные 192 сравнивают чистое
содержание, как NoPE-слой (C10), но внутри одной головы. Идея та же, что в C10 и в decoupled RoPE MLA (C16b):
позиционная часть нужна, но не обязана занимать всю голову.

## Что поменялось относительно C09

| | C09 (`QKNormAttention`, режим `head`) | C19 (`GatedAttention`) |
|---|---|---|
| `q_proj` | $C \to H \cdot D$ | $C \to H \cdot 2D$: у каждой головы `[запрос | гейт]` |
| выход | $W_O \cdot \operatorname{Attn}$ | $W_O \cdot (\operatorname{Attn} \odot \sigma(\text{гейт}))$ |
| RoPE | на всех $D$ каналах | на первых $R = D \cdot \text{factor}$ каналах, частоты считаются для размера $R$ |
| `head_dim` | $C / H$ | задаётся явно (у Qwen3-Next 256 при $C = 2048$, $H = 16$, т. е. $H \cdot D = 2C$) |
| QK-Norm | по голове | по голове (у Qwen3-Next норма вида $(1 + w)$, как у Gemma; в задаче — обычный RMSNorm) |

## Формула

$$
[\,q_h \mid g_h\,] = W^{(h)}_{Q} x \in \mathbb{R}^{2D},\qquad
\tilde q_h = \operatorname{pRoPE}(\operatorname{norm}(q_h)),\quad
\tilde k_h = \operatorname{pRoPE}(\operatorname{norm}(k_h))
$$

$$
o_h = \operatorname{softmax}\!\left(\frac{\tilde q_h \tilde K_h^\top}{\sqrt D} + M\right) V_h,\qquad
y = W_O\,[\,o_1 \odot \sigma(g_1);\ \dots;\ o_H \odot \sigma(g_H)\,]
$$

Partial RoPE для вектора $u \in \mathbb{R}^D$ на позиции $m$:
$\operatorname{pRoPE}(u) = [\,\operatorname{RoPE}_m(u_{0:R});\ u_{R:D}\,]$, где $\operatorname{RoPE}_m$ — поворот из C03
с $R/2$ частотами $\theta^{-2i/R}$. Скалярное произведение распадается на «позиционную» часть, зависящую
от $m - n$, и «содержательную», от позиций не зависящую.

## Формы тензоров

```
x                                  [B, T, C]
q_proj(x)                          [B, T, H·2D] → view [B, T, H, 2D] → chunk(2, -1)
  q                                [B, T, H, D] → q_norm → transpose → [B, H, T, D]
  gate                             [B, T, H, D] → reshape [B, T, H·D]
k = k_norm(k_proj(x))              [B, T, H_kv, D] → [B, H_kv, T, D];   v аналогично
cos, sin = rope_cos_sin(T, R, θ)   [T, R]
apply_partial_rope(q / k)          [..., :R] → apply_rope;  [..., R:] как есть;  cat → [B, ·, T, D]
repeat_kv, scores, causal, softmax [B, H, T, T] → @ v → [B, H, T, D] → [B, T, H·D]
⊙ σ(gate)                          [B, T, H·D]
o_proj                             [B, T, C]
```

## Задание

В `solution.py` (рабочая копия в `workspace/C19-gated-attention/`) реализуй:

1. `apply_partial_rope(x, cos, sin)` — $R$ берётся из `cos.shape[-1]`;
2. `GatedAttention.forward(x)` → `[B, T, C]`.

Конструктор дан (имена как в HF `Qwen3NextAttention`). Готовые `RMSNorm`, `apply_rope`, `rope_cos_sin`,
`repeat_kv` импортируются из `llmlab.core.llama`. Кеша в задаче нет.

## Реальные числа

Qwen3-Next-80B-A3B (`config.json`): 48 слоёв, `full_attention_interval = 4` — слой $i$ полный, если
$(i + 1) \bmod 4 = 0$, т. е. каждый четвёртый; остальные 36 — линейное внимание Gated DeltaNet (C21).
В полных слоях: 16 голов запросов, 2 KV-головы, `head_dim = 256`, `partial_rotary_factor = 0.25`
(вращаются 64 канала), $\theta = 10^7$.

## Критерии готовности

- `apply_partial_rope` не трогает каналы $R \dots D-1$, первые $R$ — ровно RoPE размера $R$; при $R = D$ —
  обычный RoPE; норма сохраняется, score зависит только от $m - n$;
- `GatedAttention` совпадает с оракулом (раскладка гейта по головам, сигмоида, partial RoPE, GQA, causal);
- при нулевых весах гейта выход ровно вдвое меньше негейтированного ($\sigma(0) = \tfrac12$);
- гейт зависит от запроса: закрытый гейт головы полностью убирает её вклад для этого токена;
- будущее не протекает.

Запуск: `llmlab test C19-gated-attention`.
