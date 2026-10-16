# Прогресс

> Обновляется в конце КАЖДОЙ сессии Claude Code. Пиши кратко и конкретно.

## Текущая фаза
Фаза 1 — **завершена** (2026-10-16). Следующая: Фаза 2 — конспекты (вкладка Notes).
(`data/interview_raw/` пуст, поэтому Фаза 7 не вставляется после Фазы 2.)

## Следующий шаг (точный)
Запустить Фазу 2 по промпту из `docs/PROMPTS.md`: генератор `llmlab/notes/` (summary.md +
код + diff + ошибки из `attempts` + поле «Главное, что я понял»), миграция SQLite v2 (FTS5),
эндпоинт `/api/notes` уже есть как заглушка-список в `llmlab/server.py`; модалка после
первого `passed` — в `web/src/components/TaskPage.tsx` (функция `run`).

## Сделано
**Фаза 0:**
- `pyproject.toml` (hatchling, uv; CPU-колёса torch на Linux/Windows через `tool.uv.sources`),
  `.python-version` = 3.12, `.gitignore`, `LICENSE` (MIT), `NOTICE`, краткий `README.md`.
- `llmlab/content_loader.py` — pydantic-модели `Task/Question/Paper/Block`, строгая схема,
  проверки (id = папка, модули из block.yaml, обязательные файлы по типу, depends_on без циклов,
  diff_from, papers), `Course.issues` вместо исключений.
- `llmlab/pytest_plugin.py` — `--impl`, `--impl-path`, фикстуры `impl`/`ref`/`task_info`,
  сиды, JSON-отчёт с человекочитаемыми сообщениями и строкой кода пользователя, где упало.
- `llmlab/runner/` — subprocess-pytest с таймаутом 60 с, RunResult, SQLite `attempts`
  с миграциями через `PRAGMA user_version`.
- `llmlab/validate.py` + CLI `llmlab list | test | validate | init-workspace | serve(заглушка)`.
- `llmlab/core/gpt2.py` — базовая GPT-2 с комментариями.
- Задача `C00-causal-mha` (все 7 файлов, 12 тестов, функция-диагност типичных ошибок),
  `content/blocks/C/block.yaml`, статьи `attention-is-all-you-need`, `gpt2`, 4 вопроса.
- `docs/TASK_FORMAT.md` — формат, интерфейсы (модели, фикстуры, RunResult, SQLite), чек-лист.
- `tests/` — 32 теста платформы на игрушечном курсе; `.github/workflows/ci.yml`.

**Фаза 1:**
- `llmlab/server.py` — FastAPI-приложение `create_app(root)`: `/api/course`, `/api/tasks[/{id}]`,
  `/code` (GET/PUT с `base_sha256` → 409 при внешней правке), `/code/meta`, `/reset`, `/run`,
  `/attempts`, `/reference` (403 до решения или 3 неудач), `/progress`, `/notes`, `/viz`; SPA-фоллбек.
  Защита: Host-allowlist, заголовок `X-LLMLab: 1` на изменяющих запросах, пути только через id
  задачи + проверка `is_relative_to(workspace/)`.
- `llmlab serve [--port] [--no-open]` — uvicorn на 127.0.0.1, открывает браузер.
- Загрузчик: `content/viz/*.yaml` (модель `Viz`), проверка `task.viz` и директив `:::viz id:::`.
- `web/` (Vite 5 + React 18 + TS, Monaco 0.52 локально без CDN, react-markdown + GFM + KaTeX):
  карта курса (модули, статусы, Δ diff_from, граф зависимостей SVG), страница задачи
  (вкладки Теория/Подсказки по одной/Чтение/Визуализация/Попытки; редактор; панель тестов с
  переходом к строке ошибки и подсветкой строк; Run/Reset/Diff в Monaco diff-editor),
  автосохранение 700 мс, Ctrl/⌘+Enter, опрос файла раз в 1.5 с + баннер конфликта,
  тёмная/светлая тема, перетаскиваемые сплиттеры, адаптивность до ~860 px.
- Визуализация `causal-mask` (матрица весов, слайдеры T/D, маска, √D) — в C00 README и task.yaml.
- Бандл в `llmlab/static/` (5.3 МБ, Monaco грузится лениво только на странице задачи).
- Тесты: `tests/test_server.py` (8), `tests/test_viz_registry.py` (2); всего 42.
  CI: job `frontend` проверяет, что закоммиченный бандл совпадает с исходниками.
- E2E (Playwright, вне репо): карта → C00 → Run на заготовке (0/12) → автосохранение →
  правка «из IDE» подхвачена → Ctrl+Enter → 12/12 → Diff открылся; конфликт «IDE vs
  несохранённое в UI» → «Взять версию с диска». Скриншоты светлой/тёмной темы, 1440 и 1180 px.

## Не сделано / долги
- CI ни разу не запускался (нет git remote) — проверить после первого push.
- E2E-сценарий UI не в репозитории (Playwright ставился во временную папку). Если UI начнёт
  ломаться — завести `web/e2e/` с Playwright и отдельный CI-job.
- Отслеживание внешних правок — опросом (1.5 с), не SSE/watchdog: проще и без зависимостей.
- Подсветка кода в README (```python) — простая моноширинная, без раскраски.
- Задачи не блокируются: зависимости только рекомендуются («после: …», «следующая» на карте).
- Таблицы для конспектов/SRS в SQLite — в фазах 2 и 7 (миграция v2+).
- `llmlab validate` гоняет задачи в 4 потока; при 40+ задачах может понадобиться кеш по хэшу файлов.

## Решения (почему сделано так, а не иначе)
- **pytest-плагин в пакете** (`llmlab/pytest_plugin.py`), а корневой `conftest.py` только
  подключает его: раннер передаёт `-p llmlab.pytest_plugin` явно и работает в любом корне
  (в т.ч. во временном курсе в тестах платформы), ручной `pytest content/...` тоже работает.
- **`--import-mode=importlib`**: все тесты задач называются `test_task.py`, иначе конфликт имён.
- **Флаг `--impl=buggy`** в дополнение к workspace/reference/starter — нужен validate для
  debug-задач. Для debug `--impl=starter` = `buggy.py`, и `init-workspace` копирует `buggy.py`.
- **Debug-задачи без `starter.py`**: пользователь начинает с `buggy.py`; «исправленная версия»
  для инварианта = `solution.py`.
- **Попытки пишутся в БД только для `impl=workspace`** — validate не засоряет историю.
- **Статус задачи не хранится, а вычисляется** из `attempts` (solved/attempted/new) — один
  источник правды; при появлении конспектов можно добавить таблицу-кеш.
- **C00: `diff_from: gpt2-baseline`**, а baseline-ы описываются в `block.yaml` — так `diff_from`
  валидируется (либо id задачи, либо baseline).
- **C00 — модуль + функция** (`causal_attention`, `CausalSelfAttention`); конструктор дан
  готовым с именами `c_attn`/`c_proj` как в `core/gpt2.py`, чтобы тесты грузили веса эталона.
  Готовые `F.scaled_dot_product_attention`/`nn.MultiheadAttention` запрещены через monkeypatch;
  `F.sdpa` используется как независимый оракул.
- **Новый модуль `module-0-gpt2`** в блоке C для C00 (в PLAN модули начинаются с module-1).
- Добавлен `numpy` в зависимости: без него torch печатает предупреждение при импорте.
- Папки `notes/`, `workspace/`, `.llmlab/` в `.gitignore` (как в CLAUDE.md).
- **Фаза 1, UI:** hash-роутинг (`#/task/<id>`) без react-router — нет зависимостей и не нужен
  SPA-фоллбек (он всё равно есть). Своя обёртка Monaco вместо `@monaco-editor/react` (тот по
  умолчанию тянет Monaco с CDN). Monaco подключён как `editor.api` + `editor.all` + только
  Python-подсветка и базовый воркер — без TS/JSON/CSS-воркеров.
- **Защита от CSRF заголовком `X-LLMLab: 1`**, а не токеном: сервер локальный, CORS выключен,
  кастомный заголовок кросс-доменно без preflight не отправить. Host-allowlist — от DNS rebinding.
- **Конфликт с внешним IDE:** PUT несёт `base_sha256`; если файл на диске другой → 409 и баннер
  «Взять с диска / Оставить мою». Чистый редактор подхватывает правки молча (с уведомлением).
- **Diff с эталоном** открывается после первого `passed` или 3 неудачных попыток (любой статус,
  кроме passed, считается неудачей) — проверяется на сервере, не только в UI.
- **Курс перечитывается на каждый запрос** (несколько YAML) — правки контента видны без
  перезапуска сервера.
- **Подсказки** — секции `##` в hints.md; сколько открыто — в localStorage браузера (не в БД:
  это удобство интерфейса, не прогресс).
- **Визуализация `causal-mask` для C00** — чтобы каркас viz был проверен на реальной задаче;
  V-masks из Фазы 4 может её расширить.
- **Публичный репозиторий** (GitHub, origin). Сырые вопросы `data/interview_raw/*` в
  `.gitignore` (кроме README); в Фазе 7 вопросы переформулируются своими словами — правило в CLAUDE.md.
- `httpx` — в `[dependency-groups].dev` (нужен только `fastapi.testclient`), в CI ставится явно.
- В `tests/` фильтр предупреждения Starlette о `httpx` (просит `httpx2`) — косметика.

## Известные проблемы
- Холодный старт subprocess с torch ≈ 1–1.5 с на прогон — приемлемо для UI, но заметно.

## Журнал сессий
| Дата | Фаза | Что сделано | Коммит |
|------|------|-------------|--------|
| 2026-10-16 | 0 | Каркас, загрузчик, раннер, validate, CLI, GPT-2, C00, TASK_FORMAT, CI, 32 теста | см. `git log` |
| 2026-10-16 | 1 | FastAPI API, `llmlab serve`, React/Monaco/KaTeX UI, viz-реестр + causal-mask, E2E на C00, 42 теста | см. `git log` |
