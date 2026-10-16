## TL;DR

Mamba-2 — селективная SSM со скалярным затуханием на голову: $h_t = e^{\Delta_t A}h_{t-1} + \Delta_t B_t x_t^\top$,
$y_t = h_t^\top C_t + Dx_t$. Её можно считать сканом (состояние $[N, P]$ на голову, генерация) или как attention:
$y = (L \odot CB^\top)(\Delta x) + Dx$, где $L = \exp(\operatorname{segsum}(\Delta A))$ — causal-маска с затуханием (обучение).
Это линейное внимание без нормировки с гейтом забывания; Gated DeltaNet = этот гейт + правило дельты.

## Формула

$$y_t = \sum_{s\le t}\exp\Big(\sum_{r=s+1}^{t}\Delta_r A\Big)(C_t\cdot B_s)\,\Delta_s x_s + Dx_t$$

## Формы тензоров

```
x [b,T,H,P], dt [b,T,H], B,C [b,T,N];  скан: h [b,H,N,P];  квадратично: L [b,H,T,T] ⊙ CBᵀ [b,T,T] → einsum с Δx → [b,T,H,P]
```

## Что поменялось относительно C20

Вместо $\varphi(q), \varphi(k)$ — $C_t, B_s$ (общие для голов), значения $\Delta_s x_s$, нет знаменателя, causal-маска из единиц
заменена затухающей $L_{ts} = \prod e^{\Delta_r A}$; добавлена skip-связь $D x$.
