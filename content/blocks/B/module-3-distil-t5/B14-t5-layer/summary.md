## TL;DR

T5 — энкодер-декодер, где всё упрощено: RMSNorm без центрирования и bias (T5LayerNorm, eps 1e-6), Pre-LN плюс
финальная норма стека, ни одного bias в линейных слоях, никаких позиций на входе — только относительный bias в
scores (таблица у блока 0, остальные блоки берут его bias), scores без $1/\sqrt{D}$ (масштаб в инициализации),
FFN `wo(ReLU(wi(x)))`, одна матрица эмбеддингов на энкодер, декодер и голову (с множителем $d^{-1/2}$). t5-small:
6 + 6 слоёв, 512, 8 голов по 64, FFN 2048, 60.5 М параметров.

## Формула

$$x \leftarrow x + \operatorname{Attn}(\operatorname{RMS}(x)),\ \ \operatorname{Attn}: \operatorname{softmax}(qk^\top + b_{\text{rel}} + M)v;\quad x \leftarrow x + W_{wo}\operatorname{ReLU}(W_{wi}\operatorname{RMS}(x))$$

## Формы тензоров

```
ids [B,S] → shared → [B,S,C] → 6 × блок (bias [1,H,S,S] от блока 0) → final_norm → memory [B,S,C]
dec ids [B,T] → … self (causal bias) → cross к memory [B,H,T,S] → FFN → [B,T,C]
```

## Что поменялось относительно B04

Pre-LN вместо Post-LN; RMSNorm вместо LayerNorm; нет bias в Linear; позиции — относительный bias в scores вместо
синусоид на входе; нет деления на √D; общие эмбеддинги энкодера, декодера и головы.
