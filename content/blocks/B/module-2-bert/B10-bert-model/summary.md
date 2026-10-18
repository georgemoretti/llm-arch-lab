## TL;DR

BERT = вход B08 + стопка Post-LN слоёв энкодера (multi-head self-attention с padding-маской, GELU-FFN ×4) + pooler
`tanh(dense(h[CLS]))` + MLM-голова (dense → GELU → LN → матрица эмбеддингов + bias). bert-base: 12 слоёв, 768,
12 голов, 109 М параметров, eps LayerNorm 1e-12, точный GELU. Имена параметров как в Hugging Face позволяют
загрузить настоящие веса в свою модель: «the capital of france is [MASK]» → paris.

## Формула

$$h = \operatorname{LN}(x + \operatorname{MHA}(x)),\ y = \operatorname{LN}(h + W_2\operatorname{GELU}(W_1 h)),\ \text{logits} = \operatorname{LN}(\operatorname{GELU}(W_t y))E^\top + b$$

## Формы тензоров

```
ids [B,T] → [B,T,H] → 12 × слой → [B,T,H] → pooled [B,H]; logits [B,T,V]
```

## Что поменялось относительно B03

Без декодера; GELU вместо ReLU; вход со сегментами и обучаемыми позициями; имена Hugging Face; pooler по [CLS];
MLM-голова со связанной матрицей эмбеддингов.
