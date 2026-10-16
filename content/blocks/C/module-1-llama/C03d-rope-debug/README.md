# C03d. Отладка: после RoPE «плывёт» норма и score зависит от абсолютной позиции

## Зачем это нужно

У RoPE две раскладки пар координат — интерливная (оригинальный код Meta, RoFormer) и
«половинами» (Hugging Face). Код часто собирают из кусков, взятых из разных реализаций, и
получают гибрид, который не падает, выдаёт тензоры правильных форм и даже совпадает с эталоном
на позиции 0. Задача — найти такую ошибку по **свойствам** RoPE.

## Что поменялось относительно C03

Архитектура та же — функции `rope_inv_freq`, `rope_cos_sin`, `rotate_half`, `apply_rope` из
C03, раскладка «половинами». Код написан кем-то другим, и в нём **одна** ошибка.

## Формула

$$
\operatorname{RoPE}(x, m) = x \odot \cos(m\boldsymbol\theta) + \operatorname{rotate\_half}(x) \odot \sin(m\boldsymbol\theta),
\qquad \langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle
$$

## Формы тензоров

```
positions [T], inv_freq [D/2] → angles [T, D/2] → cos, sin [T, D]
q [B,H,T,D] → q·cos + rotate_half(q)·sin → [B,H,T,D]
```

Все формы в коде правильные.

## Симптом

- `rope_inv_freq` и `rotate_half` проходят свои тесты, `apply_rope` с правильными cos/sin —
  тоже;
- на позиции 0 всё совпадает с эталоном;
- но в полном пайплайне (`rope_cos_sin` → `apply_rope`) норма q **меняется** с позицией, а
  score $\langle q_m, k_n\rangle$ для пар с одинаковым $m - n$ получается разным — модель
  «видит» абсолютные позиции, хотя не должна.

Код — в `workspace/C03d-rope-debug/solution.py`. Найди и исправь ошибку.

## Критерии готовности

Проходят все тесты C03. Запуск: `llmlab test C03d-rope-debug`.
