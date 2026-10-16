# LLM Arch Lab

Локальный интерактивный курс: шаг за шагом реализуешь эволюцию LLM-архитектур — от GPT-2 к
Llama, Gemma 3 / Olmo 2, MoE и MLA из DeepSeek и гибридам с линейным вниманием. Каждая задача —
маленький кусок механизма с автотестами, теорией, подсказками и автоконспектом.

> Статус: ранняя разработка (Фаза 1 — веб-интерфейс и демо-задача). План — `docs/PLAN.md`.

## Быстрый старт

```bash
git clone <repo> && cd llm-arch-lab
uv sync                     # или: pip install -e .
source .venv/bin/activate

llmlab serve                # откроет http://127.0.0.1:8765
```

В браузере: карта курса → задача → слева теория, подсказки и визуализации, справа редактор.
Ctrl/⌘+Enter — прогнать тесты. Код сохраняется в `workspace/<id>/solution.py`, его можно
править и в своём IDE — интерфейс подхватит изменения.

Без браузера:

```bash
llmlab list                           # задачи и статус
llmlab init-workspace C00-causal-mha  # заготовка → workspace/C00-causal-mha/solution.py
llmlab test C00-causal-mha            # прогнать тесты на своём решении
```

## Для авторов

- `docs/TASK_FORMAT.md` — формат задачи и чек-лист качества;
- `llmlab validate --all` — эталоны проходят тесты, заготовки — нет;
- `pytest tests/` — тесты платформы;
- фронтенд: `cd web && npm install && npm run dev` (Vite на :5173, API проксируется на
  `llmlab serve`), сборка — `npm run build` → `llmlab/static/` (бандл коммитится, Node
  пользователю не нужен).

Лицензия — MIT (`LICENSE`), атрибуции — `NOTICE`.
