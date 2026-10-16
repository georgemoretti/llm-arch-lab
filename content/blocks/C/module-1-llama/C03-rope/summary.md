## TL;DR

RoPE кодирует позицию поворотом: вектор q/k делится на пары координат, пара i на позиции m
поворачивается на угол m·θ^(−2i/D). Поворот сохраняет норму, а ⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩
— score зависит только от расстояния. Применяется к q и k в каждом слое, v не трогается,
обучаемых параметров нет. В раскладке HF пары — (j, j + D/2), и всё сводится к
x·cos + rotate_half(x)·sin. Llama 3: θ = 500 000.

## Формула

$$\theta_i = \theta^{-2i/D},\qquad \operatorname{RoPE}(x, m) = x \odot \cos(m\boldsymbol\theta) + \operatorname{rotate\_half}(x) \odot \sin(m\boldsymbol\theta)$$

$$\operatorname{rotate\_half}([x_1 \,|\, x_2]) = [-x_2 \,|\, x_1],\qquad \langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$$

## Формы тензоров

```
positions [T] × inv_freq [D/2] → angles [T, D/2] → cat(angles, angles) → cos, sin [T, D]
q, k [B,H,T,D] → q·cos + rotate_half(q)·sin → [B,H,T,D]   (v без изменений)
```

## Что поменялось относительно GPT-2

Обучаемые абсолютные эмбеддинги позиций `wpe` на входе модели убраны. Вместо них q и k в каждом
слое attention поворачиваются на угол, зависящий от позиции; в score остаётся только
относительное расстояние m − n, и таблицы позиций с жёстким пределом длины больше нет.
