## TL;DR

Слой декодера — три подслоя: causal self-attention по цели, cross-attention к памяти энкодера, FFN; у каждого
остаток и LayerNorm. В cross-attention запросы из декодера, ключи и значения из памяти, матрица весов $T \times S$
и маска — только padding источника (не causal: источник известен целиком). Маска self-attention декодера —
causal И padding цели по ключам. Память и `src_mask` одни на все слои.

## Формула

$$h_1 = \operatorname{LN}(y + \operatorname{MHA}(y, y; M_{\text{tgt}})),\ h_2 = \operatorname{LN}(h_1 + \operatorname{MHA}(h_1, \text{mem}; M_{\text{src}})),\ \text{out} = \operatorname{LN}(h_2 + \operatorname{FFN}(h_2))$$

## Формы тензоров

```
tgt [B,T] → mask [B,1,T,T]; y [B,T,C]; memory [B,S,C] → k,v [B,H,S,D]; cross weights [B,H,T,S]; out [B,T,C]
```

## Что поменялось относительно B03

Добавлен третий подслой — cross-attention к памяти энкодера, а self-attention стал causal. Именно cross-attention
связывает две половины модели; в decoder-only GPT его нет.
