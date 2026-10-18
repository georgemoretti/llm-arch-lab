## TL;DR

Блок трансформера = attention (смешивает позиции) + MLP (обрабатывает каждую позицию: $C \to 4C \to$
GELU $\to C$), оба с остаточной связью $x \leftarrow x + f(\cdot)$. В pre-LN (GPT-2, современные LLM)
LayerNorm стоит перед подслоем, а residual-поток не нормализуется. При нулевых подслоях блок — тождество,
и градиент по остаткам доходит до входа без затухания. В post-LN (Transformer 2017) нормализуется сумма
$\operatorname{LN}(x + f(x))$. Выход тогда всегда нормализован, но глубокие стеки обучаются хуже (C08).

## Формула

$$\text{pre: } x' = x + \operatorname{Attn}(\operatorname{LN}_1 x),\ y = x' + \operatorname{MLP}(\operatorname{LN}_2 x');\qquad \text{post: } x' = \operatorname{LN}_1(x + \operatorname{Attn} x),\ y = \operatorname{LN}_2(x' + \operatorname{MLP} x')$$

## Формы тензоров

```
x [B,T,C] → (ln_1 → attn) + x → x′ [B,T,C] → (ln_2 → c_fc [B,T,4C] → GELU → c_proj) + x′ → y [B,T,C]
```

## Что поменялось относительно A16

Голова attention стала частью повторяемого блока: добавились MLP, две остаточные связи и две
LayerNorm. Форма входа и выхода одинакова, поэтому блоки ставятся в стопку.
