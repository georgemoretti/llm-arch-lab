# LLM Arch Lab

Локальный интерактивный курс: шаг за шагом реализуешь эволюцию LLM-архитектур — от GPT-2 к
Llama, Gemma 3 / Olmo 2, MoE и MLA из DeepSeek и гибридам с линейным вниманием. Каждая задача —
маленький кусок механизма с автотестами, теорией, подсказками и автоконспектом.

> Статус: ранняя разработка (Фаза 0 — каркас, CLI и демо-задача). План — `docs/PLAN.md`.

## Быстрый старт

```bash
git clone <repo> && cd llm-arch-lab
uv sync                     # или: pip install -e .
source .venv/bin/activate

llmlab list                           # задачи и статус
llmlab init-workspace C00-causal-mha  # заготовка → workspace/C00-causal-mha/solution.py
llmlab test C00-causal-mha            # прогнать тесты на своём решении
```

Теория и задание лежат в `content/blocks/C/module-0-gpt2/C00-causal-mha/README.md`.
Веб-интерфейс (`llmlab serve`) появится в Фазе 1.

## Для авторов

- `docs/TASK_FORMAT.md` — формат задачи и чек-лист качества;
- `llmlab validate --all` — эталоны проходят тесты, заготовки — нет;
- `pytest tests/` — тесты платформы.

Лицензия — MIT (`LICENSE`), атрибуции — `NOTICE`.
