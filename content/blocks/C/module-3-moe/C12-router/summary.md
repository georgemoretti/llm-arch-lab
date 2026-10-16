## TL;DR

Роутер MoE — линейный слой $W_r \in \mathbb{R}^{E \times C}$ без bias: логиты по экспертам → оценки
(softmax по экспертам у Mixtral/Qwen, sigmoid у DeepSeek-V3) → top-k → веса выбранных,
обычно перенормированные к сумме 1 (и у DeepSeek-V3 умноженные на 2.5). Роутинг считают в fp32,
каждый токен — независимо. Индексы top-k не дифференцируемы: роутер учится только через веса,
поэтому при top-1 вес не перенормируют (Switch).

## Формула

$$s = \operatorname{softmax}(W_r x)\ \text{или}\ \sigma(W_r x),\quad \mathcal{T} = \operatorname{top-}k(s),\quad
w_e = \alpha\,\frac{s_e}{\sum_{j\in\mathcal{T}} s_j}$$

## Формы тензоров

```
x [B,T,C] → [N,C] → logits [N,E] (fp32) → scores [N,E] → weights, indices [N,k]
```

## Что поменялось относительно C02

FFN перестаёт быть общим: появляется роутер, который для каждого токена выбирает $k$ из $E$ FFN-экспертов
и их веса. Сам эксперт — тот же SwiGLU из C02.
