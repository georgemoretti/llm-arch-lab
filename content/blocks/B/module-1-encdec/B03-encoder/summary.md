## TL;DR

Слой энкодера = двунаправленный multi-head self-attention с padding-маской + FFN с ReLU, у каждого подслоя
остаток и LayerNorm. В статье LayerNorm стоит после сложения (Post-LN): $\operatorname{LN}(x + f(x))$, поэтому
выход каждого слоя нормирован. В Pre-LN — $x + f(\operatorname{LN}x)$ и финальная норма после стопки. Маска
строится по `src` один раз и передаётся во все слои. Результат — память $[B, S, C]$ для декодера.

## Формула

$$h = \operatorname{LN}_1(x + \operatorname{MHA}(x)),\quad y = \operatorname{LN}_2(h + \max(0, hW_1 + b_1)W_2 + b_2)$$

## Формы тензоров

```
src [B,S] → mask [B,1,1,S]; embed → [B,S,C] → (MHA + x → LN → FFN [B,S,d_ff→C] + h → LN) × N → memory [B,S,C]
```

## Что поменялось относительно A17

Attention двунаправленный (без causal-маски) и многоголовый, с маской паддинга; FFN с ReLU; LayerNorm после
остатка, как в исходной статье (Pre-LN — по флагу `norm_first`).
