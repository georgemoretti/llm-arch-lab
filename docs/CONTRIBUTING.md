# Как внести вклад

Спасибо, что хочешь помочь курсу. Здесь — порядок работы: как поставить окружение, добавить задачу и
проверить её перед PR. Формат файлов и интерфейсы платформы подробно описаны в `docs/TASK_FORMAT.md`,
конфигурируемая модель — в `docs/MODEL.md`.

## Окружение

```bash
git clone https://github.com/georgemoretti/llm-arch-lab && cd llm-arch-lab
uv sync                       # Python 3.12 из .python-version, CPU-сборка torch, httpx для тестов
source .venv/bin/activate
llmlab doctor                 # всё ли готово
pytest tests/                 # тесты платформы (~15 с)
llmlab validate --all         # инварианты всех задач (~1–2 мин)
```

Без uv: `python -m venv .venv && pip install -e . httpx` (на Linux/Windows сначала
`pip install torch --index-url https://download.pytorch.org/whl/cpu`, иначе приедет сборка с CUDA на ~2 ГБ).

Для финального проекта и `@pytest.mark.weights`: `uv sync --extra weights`.

## Добавить задачу

1. **Найди место в плане.** Задачи блока перечислены в `docs/PLAN.md`; новый модуль добавь в
   `content/blocks/<блок>/block.yaml` (порядок модулей = порядок на карте).
2. **Скопируй образец.** Ближайшая задача того же типа — лучший шаблон: `C00-causal-mha` (implement),
   `C03d-rope-debug` (debug), `C-P1-roformer-gqa` (paper), `C-Q1-llama-quiz` (quiz).
   ```bash
   cp -r content/blocks/C/module-1-llama/C03-rope content/blocks/C/<модуль>/C26-my-task
   ```
   Поменяй `id` (= имя папки), `module`, `depends_on`, `diff_from`, `entrypoint`.
3. **Сначала эталон и тесты, потом текст.** `solution.py` → `test_task.py` → `starter.py` (те же сигнатуры,
   тела — `raise NotImplementedError("…что сделать…")`).
   ```bash
   llmlab test C26-my-task --impl reference    # всё зелёное
   llmlab test C26-my-task --impl starter      # всё красное
   ```
4. **Сломай эталон сам.** 2–3 типичные ошибки (не та ось, нет маски, забытый масштаб) — каждая должна падать
   с сообщением, которое объясняет, что не так:
   ```bash
   cp content/…/solution.py /tmp/broken.py && $EDITOR /tmp/broken.py
   llmlab test C26-my-task --impl-path /tmp/broken.py
   ```
5. **Текст:** `README.md` (мотивация, что поменялось, формула, схема форм тензоров, задание, критерии),
   `hints.md` (3 подсказки по нарастающей), `summary.md` (4 обязательных раздела для конспекта).
6. **Связи:** вопросы — `content/questions/*.yaml` (id с префиксом темы, глоб в `task.yaml → questions`); статья —
   `content/papers/<id>.yaml`; ячейка таблицы архитектур — `content/architectures.yaml` + `task.yaml → arch`;
   визуализация — `content/viz/<id>.yaml` + компонент в `web/src/viz/`.
7. **Эталон в `llmlab/core/`.** Если следующие задачи или конфигурируемая модель будут использовать эту
   деталь — добавь её в соответствующий модуль `llmlab/core/` и тест сверки с `solution.py` в
   `tests/test_core_*.py` (так сделано для всех модулей блока C).
8. **Проверка:** `llmlab validate C26-my-task`, затем `llmlab validate --all` и `pytest tests/`.

Debug-задача (`C26d-…-debug`) — это `buggy.py` с **одним** правдоподобным багом и копия тестов implement-задачи
без диагноста, который называет баг. При правке тестов implement-задачи синхронизируй debug-копию.

## Правила контента

- **Язык:** объяснения по-русски, термины по-английски (attention, KV-cache, router), код и идентификаторы — на
  английском.
- **Свои слова.** Не копируй текст, рисунки и таблицы из статей, блогов и курсов — пересказывай и ставь ссылку
  (`reading` в `task.yaml`). Схемы — свои (SVG, компоненты). Код из чужих репозиториев — только после проверки
  лицензии и с атрибуцией в `NOTICE`.
- **Числа реальных моделей** (головы, окно, θ, эксперты) — из официального `config.json` на Hugging Face или из
  статьи, со ссылкой. Не можешь проверить — пиши `TODO-verify`, не угадывай.
- **Размер:** задача решается за 15–60 минут; больше — дроби.
- **Тесты:** фиксированные сиды, `torch.testing.assert_close` с явными `atol/rtol`, d_model ≤ 64, seq ≤ 32,
  каждый тест < 2 с на CPU; тесты с реальными весами — `@pytest.mark.weights`.
- Сырые вопросы с собеседований (`data/interview_raw/`) в репозиторий не попадают — см. `docs/QUESTIONS_IMPORT.md`.

## Фронтенд

```bash
cd web && npm ci
npm run dev          # Vite на :5173, API проксируется на запущенный `llmlab serve`
npm run build        # сборка в llmlab/static/
```

Собранный бандл **коммитится** вместе с исходниками: пользователю не нужен Node. CI пересобирает его и падает,
если `llmlab/static/` не совпадает с `web/src`. Всё должно работать офлайн — никаких CDN.

## Коммиты и PR

- Сообщения в стиле Conventional Commits: `feat(C/module-2): add SWA task`, `fix(runner): …`, `docs: …`.
- Один PR — одна задача или одно изменение платформы.
- Перед PR: `pytest tests/` и `llmlab validate --all` зелёные; для новой задачи — чек-лист из
  `docs/TASK_FORMAT.md` §5.
