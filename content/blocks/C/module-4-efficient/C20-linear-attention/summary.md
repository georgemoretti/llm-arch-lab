## TL;DR

Линейное внимание заменяет $\exp(q^\top k)$ на $\varphi(q)^\top\varphi(k)$ с $\varphi = \operatorname{elu}+1 > 0$,
и ассоциативность $(\varphi(Q)\varphi(K)^\top)V = \varphi(Q)(\varphi(K)^\top V)$ убирает матрицу $[T, T]$. Causal-форма —
это RNN: состояние $S_t = S_{t-1} + \varphi(k_t)v_t^\top$ размера $[d, d_v]$ и $z_t = z_{t-1} + \varphi(k_t)$, выход
$\varphi(q_t)^\top S_t / \varphi(q_t)^\top z_t$. Памяти $O(1)$ по длине вместо KV-cache, но вся история сжата в
фиксированную матрицу — точные выборки из далёкого прошлого хуже. Отсюда DeltaNet и гибриды.

## Формула

$$S_t = S_{t-1} + \varphi(k_t)v_t^\top,\quad z_t = z_{t-1} + \varphi(k_t),\quad
o_t = \frac{\varphi(q_t)^\top S_t}{\varphi(q_t)^\top z_t} = \frac{\sum_{s\le t}\varphi(q_t)^\top\varphi(k_s)\,v_s}{\sum_{s\le t}\varphi(q_t)^\top\varphi(k_s)}$$

## Формы тензоров

```
φ(q), φ(k) [B,H,T,d], v [B,H,T,dv];  параллельно: A [B,H,T,T].tril() → A·V / ΣA;  рекуррентно: S [B,H,d,dv], z [B,H,d] → o_t [B,H,dv]
```

## Что поменялось относительно C00

Нет softmax и экспонент: вес — $\varphi(q)^\top\varphi(k)$, нормировка — деление на сумму весов, маска — нули.
Вместо KV-cache — состояние фиксированного размера; при обучении можно считать параллельно, при генерации — как RNN.
