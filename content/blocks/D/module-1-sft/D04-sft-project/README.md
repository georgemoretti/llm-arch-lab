# Проект: SFT игрушечной модели — формат от SFT, знания от предобучения

> Черновик: раздел с результатами эксперимента допишем после прогона `main` (см. docs/PROGRESS.md).

## Задание

В `solution.py` (рабочая копия в `workspace/D04-sft-project/`) напиши `build_example`, `check_answer`, `chat_answer`,
`evaluate`. Данные, «предобучение» и цикл SFT готовы. Эксперимент запускается командой
`python content/blocks/D/module-1-sft/D04-sft-project/solution.py`.

## Формы тензоров

```shapes
ids [T]
targets [T]
> collate
batch [B, T]
> sft_step (D03)
loss []
axis T = токенов в диалоге
axis B = примеров в батче
```

## Критерии готовности

Проходят все тесты. Запуск: `llmlab test D04-sft-project`.
