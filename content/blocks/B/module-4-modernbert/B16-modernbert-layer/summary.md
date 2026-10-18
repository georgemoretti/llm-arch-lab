## TL;DR

ModernBERT — BERT с приёмами современных декодеров: RoPE вместо обучаемых позиций (θ = 160 000 в глобальных слоях,
10 000 в локальных), GeGLU-FFN, ни одного bias, Pre-LN с LayerNorm без bias и `final_norm`, одна матрица `Wqkv`.
Глобальный слой идёт через два локальных (0, 3, 6, …); локальное окно двунаправленное, `|i − j| ≤ 64` при
`local_attention = 128`. У слоя 0 нет `attn_norm`: вход уже нормирован после эмбеддингов. Закрытые ключи — `finfo.min`,
чтобы PAD-запрос без видимых ключей не дал nan. Base: 22 слоя, 768, 12 голов, `intermediate_size` 1152, словарь
50 368, контекст 8192.

## Формула

$$x \leftarrow x + W_o \operatorname{softmax}\!\big(R_\theta q (R_\theta k)^\top/\sqrt{D} + M_\ell\big) v;\quad x \leftarrow x + W_o'(\operatorname{GELU}(a)\odot g),\ [a|g] = W_i \operatorname{LN}(x)$$

## Формы тензоров

```
ids [B,T] → emb + LN → [B,T,C] → 22 × слой (Wqkv [B,T,3C] → q,k,v [B,H,T,D], RoPE, маска PAD / PAD+окно) → final_norm
```

## Что поменялось относительно B10

Позиции — RoPE, а не таблица; Pre-LN вместо Post-LN; GeGLU вместо GELU-FFN; нет bias; локальное окно в двух слоях из
трёх; нет token type; контекст 8192 вместо 512.
