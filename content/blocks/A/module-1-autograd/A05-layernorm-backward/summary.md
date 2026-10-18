## TL;DR

В backward LayerNorm $\mu$ и $\sigma$ — функции от $x$, а не константы. Если учесть пути через них,
получится $dx = r\,(g - \operatorname{mean}(g) - \hat x\,\operatorname{mean}(g\hat x))$, где $g = dy\,\gamma$.
Вычитания убирают из градиента направления $\mathbf 1$ и $\hat x$, к которым LayerNorm нечувствительна.
Поэтому $\sum dx = 0$, а $dx(kx) = dx(x)/k$. Для параметров: $d\gamma = \sum dy\,\hat x$, $d\beta = \sum dy$.

## Формула

$$dx = \frac{1}{\sqrt{\sigma^2+\varepsilon}}\Big(g - \tfrac{1}{C}\textstyle\sum_j g_j - \hat x\,\tfrac{1}{C}\sum_j g_j \hat x_j\Big),\qquad g = dy \odot \gamma$$

## Формы тензоров

```
x [B,T,C] → μ, rstd [B,T,1] → x̂ [B,T,C] → y;  backward: dγ, dβ [C] (Σ по B,T), dx [B,T,C]
```

## Что поменялось относительно A04

В A04 якобианы были простыми: диагональ у ReLU, матрица $W$ у Linear. Здесь якобиан плотный внутри
строки, потому что каждый выход зависит от всех входов строки через общие статистики. Ключевой навык —
учесть все пути зависимости, иначе получится правдоподобная, но неверная формула.
