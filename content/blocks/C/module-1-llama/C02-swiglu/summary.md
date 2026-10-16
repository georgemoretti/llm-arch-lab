## TL;DR

SwiGLU FFN: две параллельные проекции входа в F — gate (через SiLU) и up (линейная), их
поэлементное произведение, затем down обратно в C. Гейт решает, какие каналы открыты, up —
что по ним передать. Три матрицы без bias, поэтому F ≈ 8/3·d вместо 4·d (тот же бюджет
параметров); Llama округляет F вверх до кратного `multiple_of`, у Llama 3 8B F = 14 336.

## Формула

$$\operatorname{FFN}(x) = W_{\text{down}}\bigl(\operatorname{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\bigr),\quad \operatorname{SiLU}(z) = z\,\sigma(z)$$

## Формы тензоров

```
x [B,T,C] → gate_proj → [B,T,F] → silu ─┐
x [B,T,C] → up_proj   → [B,T,F] ────────⊙→ [B,T,F] → down_proj → [B,T,C]
```

## Что поменялось относительно GPT-2

GELU-MLP (`c_fc` C→4C, GELU, `c_proj` 4C→C, с bias) заменён на SwiGLU (gate/up C→F, SiLU на
gate, произведение, down F→C, без bias); скрытый размер 4·d → ≈ 8/3·d.
