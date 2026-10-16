## TL;DR

Sliding window attention — это обычный causal attention с маской, где запрос на позиции $p$ видит
только ключи $p - W < j \le p$ (окно из $W$ позиций, включая себя). Веса и RoPE не меняются.
Выигрыш — $O(TW)$ вычислений и KV-cache размера $W$ на слой; цена — слой не видит дальше окна.
Компенсация: стек из $L$ слоёв видит на $L(W-1)$ назад (Mistral 7B: все слои SWA, $W = 4096$) или
часть слоёв оставляют глобальными (Gemma 2 — 1:1, $W = 4096$; Gemma 3 — 5 sliding : 1 full, $W = 1024$).

## Формула

$$M_{p,j} = 0 \text{ если } 0 \le p - j < W, \text{ иначе } -\infty;\qquad
\text{слой } i \text{ full} \iff (i + 1) \bmod \text{pattern} = 0$$

## Формы тензоров

```
q_pos = arange(S−T, S)[:, None], k_pos = arange(S)[None, :]
mask = (k_pos <= q_pos) & (q_pos − k_pos < W)          [T, S]
scores [B,H,T,T] + mask → softmax → out [B,T,C]
```

## Что поменялось относительно C05

Только маска: к условию causal $j \le p$ добавилось $p - j < W$. Плюс расписание слоёв: модель
перестаёт быть стопкой одинаковых блоков — у каждого слоя тип `sliding_attention` или
`full_attention` (у Gemma 3 ещё и своё основание RoPE).
