# Прогресс

> Обновляется в конце КАЖДОЙ сессии Claude Code. Пиши кратко и конкретно.

## Текущая фаза
Фаза 0 — **завершена** (2026-10-16). Следующая: Фаза 1 — локальный веб-интерфейс (MVP).

## Следующий шаг (точный)
Запустить Фазу 1 по промпту из `docs/PROMPTS.md`: FastAPI-эндпоинты поверх
`llmlab.content_loader.load_course` и `llmlab.runner.run_task_tests` (не дублировать логику),
фронтенд в `web/`, сборка в `llmlab/static/`, заменить заглушку `llmlab serve` в `llmlab/cli.py`.

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

## Не сделано / долги
- CI ни разу не запускался (нет git remote) — проверить после первого push.
- Реестр визуализаций: `viz` в task.yaml пока не валидируется (появится в Фазе 1).
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

## Известные проблемы
- Холодный старт subprocess с torch ≈ 1–1.5 с на прогон — приемлемо для UI, но заметно.

## Журнал сессий
| Дата | Фаза | Что сделано | Коммит |
|------|------|-------------|--------|
| 2026-10-16 | 0 | Каркас, загрузчик, раннер, validate, CLI, GPT-2, C00, TASK_FORMAT, CI, 32 теста | см. `git log` |
