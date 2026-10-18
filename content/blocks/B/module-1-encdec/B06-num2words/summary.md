## TL;DR

Энкодер-декодер из B01–B05 за 25 с на CPU учится переводить числа в слова (0 … 999 999) по 2000 примерам и
переводит 99.4 % отложенных чисел без единой ошибки. Батч — пары с PAD справа; шаг — teacher forcing с label
smoothing и clip; качество — exact match (сравнение до EOS включительно). Cross-attention сам выучил
выравнивание «слово ↔ цифры». Ошибки — только на коротких числах: среди случайных чисел до миллиона их почти
нет (сдвиг распределения).

## Формула

$$\text{EM} = \frac1N\sum_n\big[\operatorname{greedy}(\text{src}_n)_{\le \text{EOS}} = \text{tgt}_n\big]$$

## Формы тензоров

```
pairs → collate → src [B,≤6], tgt [B,≤8] → shift_right → logits [B,T,55] → loss
cross-attention последнего слоя: [T_out, S] — выравнивание слов и цифр
```

## Что поменялось относительно A18

Вместо продолжения одного текста — пары «источник → цель» разной длины, PAD и маски, teacher forcing через
shift right, метрика exact match и карта cross-attention.
