# LLM Arch Lab

Локальный интерактивный курс: шаг за шагом реализуешь эволюцию LLM-архитектур — от GPT-2 к
Llama, Gemma 3 / Olmo 2, MoE и MLA из DeepSeek и гибридам с линейным вниманием. Каждая задача —
маленький кусок механизма с автотестами, теорией, подсказками и автоконспектом.

> Статус: блок C готов целиком — 50 задач от GPT-2 до гибридов и финальный проект на настоящих весах
> Qwen3-0.6B. План — `docs/PLAN.md`.

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

Когда все тесты прошли, интерфейс попросит 1–3 строки «Главное, что я понял» и соберёт
конспект `notes/<блок>/<id>.md`: выжимка, твой код, diff с эталоном, ошибки по пути, вопросы.
Вкладка «Конспекты» — поиск, теги, граф связей, таблица архитектур и экспорт в zip. Папка
`notes/` — готовый vault для Obsidian.

Вкладка **Interview** — все вопросы курса и твои вопросы с собеседований (`data/interview_raw/`, см.
`docs/QUESTIONS_IMPORT.md`): интервальное повторение по SM-2 с самооценкой, квиз по модулю и список с
фильтрами. Ответы во вкладке «Вопросы» задачи тоже идут в повторение.

Без браузера:

```bash
llmlab list                           # задачи и статус
llmlab init-workspace C00-causal-mha  # заготовка → workspace/C00-causal-mha/solution.py
llmlab test C00-causal-mha            # прогнать тесты на своём решении
llmlab notes build                    # пересобрать конспекты (твои выводы сохраняются)
llmlab notes export                   # zip с конспектами
llmlab questions import               # импорт своей базы вопросов из data/interview_raw/
```

## Финальный проект: своя модель на настоящих весах

Все детали блока C собираются в одну модель из YAML (`llmlab/core/model.py`, конфиги — `llmlab/configs/`,
схема и как добавить компонент — `docs/MODEL.md`). Задачи **C24** (config.json и имена весов) и **C25**
(загрузка safetensors, сэмплирование, генерация, сверка логитов с `transformers`) превращают её в
настоящую Qwen3-0.6B.

```bash
uv sync --extra weights                       # huggingface_hub, tokenizers, transformers (для сверки)
llmlab model                                  # готовые конфиги и число параметров
llmlab model qwen3-0.6b                       # расписание слоёв
llmlab generate qwen3-0.6b "The capital of France is"
llmlab generate qwen3-0.6b "Name three planets of the solar system" --chat -n 60
llmlab generate smollm2-135m "Once upon a time" -t 0.8 --top-p 0.9

llmlab test C25-qwen3-project                 # тесты на случайных весах — без скачивания
llmlab test C25-qwen3-project --weights       # + логиты и генерация против transformers на реальных весах
llmlab generate qwen3-0.6b "Hello" --impl workspace   # генерировать своим решением C25
```

Первый запуск с реальными весами скачивает ~1.5 ГБ в `~/.cache/huggingface`. Без `--weights` (и без
`LLMLAB_WEIGHTS=1`) тесты `@pytest.mark.weights` пропускаются, так что курс работает и офлайн.

## Для авторов

- `docs/TASK_FORMAT.md` — формат задачи и чек-лист качества;
- `docs/MODEL.md` — конфигурируемая модель: схема YAML, реестр компонентов;
- `llmlab validate --all` — эталоны проходят тесты, заготовки — нет;
- `pytest tests/` — тесты платформы;
- фронтенд: `cd web && npm install && npm run dev` (Vite на :5173, API проксируется на
  `llmlab serve`), сборка — `npm run build` → `llmlab/static/` (бандл коммитится, Node
  пользователю не нужен).

Лицензия — MIT (`LICENSE`), атрибуции — `NOTICE`.
