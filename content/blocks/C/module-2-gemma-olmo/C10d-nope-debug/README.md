# Отладка: веса SmolLM3 загрузились, а модель генерирует мусор

## Зачем это нужно

В моделях, где слои **разные** (sliding/full у Gemma, RoPE/NoPE у SmolLM3, dense/MoE у DeepSeek),
появляется новый класс ошибок: каждый слой по отдельности написан верно, но тип слоя выбран не по
тому индексу. Загрузка весов при этом проходит без единого предупреждения — у RoPE- и NoPE-слоя
одинаковые параметры. Модель просто считает не то, и найти это можно только сверкой расписания
с `config.json`.

## Что поменялось относительно C10

Архитектура та же — `rope_layers`, `NoPEAttention`, `build_attention_layers` из C10. Код написан
кем-то другим, и в нём **одна** ошибка.

## Формула

$$\text{score}_{ij} = \begin{cases}(R_i q_i)\cdot(R_j k_j)/\sqrt{D}, & \text{слой с RoPE}\\ q_i \cdot k_j/\sqrt{D}, & \text{NoPE-слой}\end{cases}$$

## Формы тензоров

```shapes
x [B, T, C]
> проекции
q [B, H, T, D]
k, v [B, H_kv, T, D]
> q·kᵀ/√D + causal
scores [B, H, T, T]
> softmax, @ v, o_proj
out [B, T, C]
```

## Симптом

- отдельный слой с `use_rope=True` совпадает с `LlamaAttention`, с `use_rope=False` — с оракулом NoPE;
- causal-маска на месте, свойства NoPE («мешок слов» для последнего токена, различение позиций) выполняются;
- в стеке слоёв **число** NoPE-слоёв правильное — 9 из 36 у SmolLM3-3B;
- но сверка с `no_rope_layers` из config.json SmolLM3 не сходится.

Код — в `workspace/C10d-nope-debug/solution.py`. Найди и исправь ошибку.

## Критерии готовности

Проходят все тесты C10. Запуск: `llmlab test C10d-nope-debug`.
