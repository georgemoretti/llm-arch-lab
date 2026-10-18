## TL;DR

Seq2seq = энкодер (один раз) + декодер. Обучение — teacher forcing: декодеру подают правильный ответ, сдвинутый
вправо (`[BOS, y0 … y_{T−2}]`), и все позиции считаются за один проход. Лосс — cross-entropy со сглаженной целью
$(1-\varepsilon)\,\text{one-hot} + \varepsilon/V$, PAD не считается; градиент $p - q$. Генерация — цикл по
декодеру на растущем префиксе: argmax последней позиции, PAD после EOS, стоп, когда закончили все. Одна матрица
эмбеддингов на энкодер, декодер и выходной слой.

## Формула

$$\mathcal L = (1-\varepsilon)(-\log p_y) + \varepsilon\Big(-\tfrac1V\textstyle\sum_v \log p_v\Big),\qquad \text{tgt\_in} = [\text{BOS}, y_0, \dots, y_{T-2}]$$

## Формы тензоров

```
src [B,S] → memory [B,S,C]; tgt [B,T] → shift_right → [B,T] → logits [B,T,V] → loss []
greedy: [B,1] BOS → … → [B,L+1] → без BOS [B,L]
```

## Что поменялось относительно A18

Вместо продолжения одного текста — отображение источника в цель: отдельный энкодер, cross-attention, shift right с
BOS, label smoothing и генерация с одним проходом энкодера.
