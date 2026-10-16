# Формат задач LLM Arch Lab

Спецификация для авторов задач (людей и Claude Code). Источник правды по архитектуре —
`CLAUDE.md`; этот документ описывает файлы задачи, интерфейсы платформы и чек-лист качества.

Эталонный пример задачи: `content/blocks/C/module-0-gpt2/C00-causal-mha/`.

---

## 1. Где лежат задачи

```
content/blocks/<BLOCK>/block.yaml                  # блок: модули и базовые архитектуры
content/blocks/<BLOCK>/<module-id>/<task-id>/      # одна задача
content/questions/*.yaml                           # вопросы (YAML-список)
content/papers/<paper-id>.yaml                     # одна статья на файл
```

**Именование id:**

| вид | шаблон | пример |
|-----|--------|--------|
| implement / project | `<Блок><NN>-<slug>` | `C03-rope` |
| debug | `<Блок><NN>d-<slug>-debug` | `C03d-rope-debug` |
| paper / quiz модуля | `<Блок>-P<N>-<slug>`, `<Блок>-Q<N>-<slug>` | `C-P1-roformer-gqa` |

id задачи **совпадает с именем папки**, `block` и `module` — с папками выше. Это проверяет загрузчик.

### `block.yaml`

```yaml
id: C
title: "Эволюция декодеров"
description: "…"
modules:                       # порядок = порядок в UI
  - id: module-1-llama
    title: "GPT-2 → Llama"
    description: "RMSNorm, SwiGLU, RoPE, GQA, KV-cache."
baselines:                     # допустимые значения diff_from помимо id задач
  gpt2-baseline:
    title: "GPT-2"
    path: llmlab/core/gpt2.py
```

---

## 2. Файлы задачи

| файл | implement / project | debug | paper / quiz |
|------|:---:|:---:|:---:|
| `task.yaml` | ✓ | ✓ | ✓ |
| `README.md` | ✓ | ✓ | ✓ |
| `starter.py` | ✓ | — | — |
| `buggy.py` | — | ✓ | — |
| `solution.py` | ✓ | ✓ | — |
| `test_task.py` | ✓ | ✓ | — |
| `hints.md` | ✓ | ✓ | опц. |
| `summary.md` | ✓ | ✓ | опц. |

### 2.1 `task.yaml`

```yaml
id: C03-rope                     # = имя папки
title: Rotary Position Embeddings
block: C                         # = папка блока
module: module-1-llama           # = папка модуля, описан в block.yaml
type: implement                  # implement | debug | paper | quiz | project
difficulty: medium               # easy | medium | hard
depends_on: [C01-rmsnorm]        # id задач; граф без циклов
diff_from: gpt2-baseline         # id задачи или baseline из block.yaml; null — нет
tags: [positional-encoding, attention]
entrypoint: apply_rope           # строка или список имён, которые импортируют тесты
time_estimate_min: 30            # 5..120; задача должна решаться за 15–60 мин
papers: [roformer]               # id из content/papers/
reading:                         # внешние ссылки — только ссылки, без копий текста
  - title: "…"
    url: https://…
viz: [rope-rotation]             # id визуализаций из content/viz/ (вкладка «Визуализация»)
questions: [rope-*, positional-*]  # fnmatch-глобы по id вопросов
arch:                            # ячейки таблицы архитектур, которые открывает решение (§3.4)
  llama-3: {position: "RoPE, θ = 500 000"}   # модель → столбец → значение (с источником!)
```

Неизвестные поля — ошибка (схема строгая, опечатки ловятся сразу).

### 2.2 `README.md` — обязательные разделы

1. **Зачем это нужно** — какую проблему решает механизм.
2. **Что поменялось относительно `<diff_from>`** — лучше таблицей «было / стало».
3. **Формула** — KaTeX (`$…$`, `$$…$$`).
4. **Формы тензоров** — ASCII-схема в блоке кода, каждая стрелка подписана операцией.
5. **Задание** — что реализовать, какие сигнатуры менять нельзя, что запрещено использовать.
6. **Критерии готовности** — человеческим языком перечисляет то, что проверяют тесты.

Для debug-задачи вместо «Задания» — **«Симптом»**: что наблюдает пользователь (например,
«генерация с KV-cache расходится с полным прогоном после 1-го шага»), но **не** сам баг.

Гиперпараметры реальных моделей — только с источником (`config.json` на HF или статья).
Если не удалось проверить — пометка `TODO-verify`.

Визуализацию можно вставить прямо в текст отдельной строкой `:::viz <id>:::` (см. §3.3).

### 2.3 `starter.py`

- Те же имена и сигнатуры, что в `solution.py`, docstring с формами входа/выхода.
- Тело каждой функции/метода, которые проверяют тесты: `TODO`-комментарии с наводящими
  вопросами + `raise NotImplementedError("<имя>")`.
- Конструкторы `nn.Module` можно давать готовыми, если тесты грузят веса эталона
  (имена параметров должны совпадать) — в этом случае комментарий «не меняй».
- **Инвариант:** на заготовке не проходит ни один тест.

### 2.4 `solution.py`

Эталон: короткий, читаемый, с комментариями форм. Проходит все тесты.

### 2.5 `buggy.py` (debug)

Копия эталона с **одним** правдоподобным багом (из тех, что реально делают люди).
Комментариев, выдающих баг, нет. Пользователь начинает с этого файла
(`init-workspace` копирует его в `workspace/<id>/solution.py`).
**Инвариант:** падает хотя бы один тест; исправленная версия (= `solution.py`) проходит всё.

### 2.6 `test_task.py`

```python
def test_something(impl, ref):
    """Короткое человекочитаемое название — его показывает UI"""
    ...
```

Фикстуры (из `llmlab/pytest_plugin.py`):

| фикстура | что это |
|----------|---------|
| `impl` | модуль с проверяемой реализацией (scope=module). Выбор — флаг `--impl` |
| `ref` | модуль `solution.py` этой задачи — для численного сравнения |
| `task_info` | dict из `task.yaml` |

Перед каждым тестом автоматически: `torch.manual_seed(0)` и
`torch.use_deterministic_algorithms(True, warn_only=True)` (если torch импортирован).

Правила:

- игрушечные размеры (`d_model ≤ 64`, `T ≤ 32`), каждый тест < 2 с на CPU;
- сравнение через `torch.testing.assert_close(..., atol=…, rtol=…, msg=…)` или
  `torch.allclose` + `pytest.fail(…, pytrace=False)`;
- проверять: формы, численное совпадение (с `ref` или с независимым оракулом), **свойства
  механизма** (causal-маска не пропускает будущее; RoPE сохраняет норму; KV-cache == полный
  прогон; веса роутера суммируются в 1 …);
- **каждый тест должен вызывать реализацию** — иначе он пройдёт на заготовке и нарушит
  инвариант;
- сообщения учат: не «assert failed», а «выход в позиции 3 зависит от токена 4 → утечка
  через маску». Хороший приём — функция-диагност, которая сравнивает выход с типичными
  неправильными вариантами (без масштаба, с перевёрнутой маской…) и называет совпавший
  (см. `_diagnose_attention` в C00);
- если готовые функции запрещены — запрещать через `monkeypatch` в autouse-фикстуре,
  а оракул сохранять до патча;
- тесты, которым нужны реальные веса, — `@pytest.mark.weights` (по умолчанию не запускаются).

### 2.7 `hints.md`

Три уровня, заголовки `## Подсказка 1..3`: (1) направление мысли, (2) ключевая операция,
(3) почти код для самой коварной части. Полного решения в подсказках нет.

### 2.8 `summary.md`

Авторская выжимка для автоконспекта (§4.7). Разделы строго (в конспект попадают именно они,
в этом порядке; заголовок четвёртого раздела копируется как есть):

```markdown
## TL;DR
## Формула
## Формы тензоров
## Что поменялось относительно <diff_from>
```

---

## 3. Вопросы и статьи

### `content/questions/*.yaml` — YAML-список

```yaml
- id: rope-relative-01          # уникальный; префикс = тема (для глобов в task.yaml)
  topic: positional-encoding
  tasks: [C03-rope]             # явная привязка (дополняет глоб из task.yaml)
  difficulty: medium            # easy | medium | hard
  kind: open                    # open | mcq | code-reading
  question: "…"
  answer: "…"                   # эталонный ответ своими словами
  options: []                   # для mcq
  source: course                # course | user-db | …
```

### `content/papers/<id>.yaml`

```yaml
id: roformer
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: "Su, Lu, Pan, Murtadha, Wen, Liu"
year: 2021
venue: arXiv            # опционально
url: https://arxiv.org/abs/2104.09864
tags: [positional-encoding]
```

### 3.3 Визуализации: `content/viz/<id>.yaml` + компонент

```yaml
id: causal-mask
title: "Causal-маска и веса внимания"
description: "Что показывает и что покрутить."
```

Интерактивная часть — React-компонент в `web/src/viz/<Name>.tsx`, зарегистрированный в
`web/src/viz/registry.ts` (`"causal-mask": CausalMask`). Компонент без пропсов, сам держит
состояние, использует CSS-переменные темы (`--accent`, `--border`, …) и работает офлайн.
Загрузчик проверяет, что каждый id из `task.yaml → viz` и из директив `:::viz id:::` в README
описан в `content/viz/`; `tests/test_viz_registry.py` — что у каждого описания есть компонент.
После правок фронтенда: `cd web && npm run build` и коммит `llmlab/static/`.

### 3.4 Таблица архитектур: `content/architectures.yaml` + `task.yaml → arch`

```yaml
columns:                       # компоненты (столбцы)
  - {id: norm, title: "Нормализация"}
models:                        # модели (строки)
  - id: llama-3
    title: Llama 3 (8B)
    year: 2024
    module: module-1-llama     # модуль курса, где строка начнёт заполняться
    source: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
```

Значения ячеек пишет **задача**, которая объясняет компонент (`arch` в task.yaml), и ячейка
открывается во вкладке «Конспекты», когда задача решена. Закрытая ячейка показывает только,
какая задача её откроет. Загрузчик проверяет: модель и столбец описаны, одну ячейку задаёт одна
задача. Факты — только из `source` (config.json на HF или статья); источник указывать
комментарием рядом с `arch`.

---

## 4. Интерфейсы платформы

### 4.1 Pydantic-модели (`llmlab/content_loader.py`)

- `Task` — поля `task.yaml` + `dir: Path` (заполняет загрузчик) и свойства
  `entrypoints`, `has_code`, `starter_file` (`buggy.py` для debug), `solution_file`, `test_file`.
- `Question`, `Paper`, `Block` (`modules: list[Module]`, `baselines: dict[str, Baseline]`).
- `load_course(root=None) -> Course`. Не бросает исключений на ошибках контента — копит
  `Course.issues: list[Issue(level, where, message)]`. Методы: `task(id)`, `topo_order()`,
  `questions_for(task)`, свойство `errors`.

Проверки загрузчика: схема YAML; id = папка; block/module = папки; модуль описан в
`block.yaml`; обязательные файлы по типу; `entrypoint` у задач с кодом; `depends_on`
ссылается на существующие задачи и не образует циклов; `diff_from` — задача или baseline;
`papers` существуют. Предупреждения: глоб `questions` ничего не нашёл, нестандартный id.

### 4.2 pytest-плагин (`llmlab/pytest_plugin.py`)

Флаги: `--impl=workspace|reference|starter|buggy` (по умолчанию `workspace`),
`--impl-path=<file>` (перекрывает `--impl`), `--llmlab-report=<json>`.

| `--impl` | файл |
|----------|------|
| `workspace` | `workspace/<task_id>/solution.py` (корень — `LLMLAB_ROOT` или папка с `content/blocks`) |
| `reference` | `<task_dir>/solution.py` |
| `starter` | `<task_dir>/starter.py` (для debug — `buggy.py`) |
| `buggy` | `<task_dir>/buggy.py` |

Подключается корневым `conftest.py`, поэтому работает и ручной запуск:
`pytest content/blocks/C/module-0-gpt2/C00-causal-mha --impl=reference`.

### 4.3 Раннер (`llmlab/runner/run.py`)

`run_task_tests(task, impl="workspace", impl_path=None, timeout_s=60, record=None) -> RunResult`

Запускает `python -m pytest <test_task.py> -p llmlab.pytest_plugin --impl=… --llmlab-report=…`
в subprocess с таймаутом. Попытка пишется в SQLite только для `impl="workspace"`
(прогоны `validate` историю не засоряют).

**RunResult (JSON):**

```json
{
  "task_id": "C00-causal-mha",
  "impl": "workspace",              // workspace | reference | starter | buggy | path
  "created_at": "2026-10-16T12:00:00+00:00",
  "status": "failed",               // passed | failed | error | timeout
  "passed": 9, "failed": 3, "total": 12,
  "duration_s": 3.4,
  "exit_code": 1,                   // null при таймауте
  "code_path": "/…/workspace/C00-causal-mha/solution.py",
  "code_sha256": "…",
  "tests": [
    {
      "nodeid": "content/…/test_task.py::test_attention_matches_oracle",
      "name": "test_attention_matches_oracle",
      "title": "causal_attention: численно совпадает с эталонной формулой …",  // 1-я строка docstring
      "outcome": "failed",          // passed | failed | error | skipped | not-run
      "message": "Выход не совпадает с эталоном (max |Δ| = 2.02). Похоже, забыто деление на √D …",
      "location": {"file": "solution.py", "line": 18, "function": "causal_attention",
                   "source": "scores = q @ k.transpose(-2, -1)"},   // кадр в коде пользователя или null
      "duration_s": 0.004
    }
  ],
  "collect_errors": [],             // ошибки сборки тестов (битый test_task.py)
  "message": null,                  // общее сообщение для timeout/error
  "output_tail": "…",               // хвост stdout/stderr pytest (до 4000 символов)
  "attempt_id": 7                   // только если попытка записана в БД
}
```

`status`: `passed` — все тесты прошли (и их > 0); `failed` — есть упавшие (в т.ч. ошибка
импорта кода пользователя: тогда у тестов `outcome: error`); `error` — тесты не собрались или
не запустились; `timeout` — превышен таймаут.

`message` теста: для `AssertionError` — авторское сообщение без хвоста интроспекции pytest;
для `NotImplementedError` — «Функция ещё не реализована»; иначе `Тип: текст`.

### 4.4 SQLite (`.llmlab/progress.db`, путь переопределяется `LLMLAB_HOME`)

Схема версионируется через `PRAGMA user_version` (миграции в `llmlab/runner/db.py`).

```sql
-- v1
CREATE TABLE attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,   -- ISO-8601 UTC
    impl         TEXT    NOT NULL,
    status       TEXT    NOT NULL,   -- passed | failed | error | timeout
    passed       INTEGER NOT NULL,
    failed       INTEGER NOT NULL,   -- failed + error
    total        INTEGER NOT NULL,
    duration_s   REAL    NOT NULL,
    code_sha256  TEXT,
    result_json  TEXT    NOT NULL    -- полный RunResult
);
```

```sql
-- v2 (конспекты)
CREATE TABLE notes (
    task_id      TEXT PRIMARY KEY,
    insight      TEXT NOT NULL,      -- «Главное, что я понял»
    code         TEXT,               -- снимок кода последнего зелёного прогона
    code_sha256  TEXT,
    path         TEXT NOT NULL,      -- относительно notes/
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE notes_index (path TEXT PRIMARY KEY, task_id TEXT, mtime_ns INTEGER NOT NULL);
-- notes_fts — FTS5 (path UNINDEXED, title, tags, body), создаётся лениво в llmlab/notes/index.py;
-- без FTS5 поиск идёт сканом файлов.
```

Статус задачи вычисляется из попыток (`db.task_statuses()`): `solved` — была попытка
`passed`, `attempted` — были попытки, иначе `new`.

### 4.5 CLI

```bash
llmlab list                         # задачи в порядке зависимостей и статус
llmlab init-workspace <id> [--force]
llmlab test <id> [--impl reference|starter|buggy] [--impl-path file.py] [--timeout 60]
llmlab validate <id> | --all [-j 4]
llmlab serve [--port 8765] [--no-open]   # веб-интерфейс на 127.0.0.1
llmlab notes build [<id>] [--insight "строка\nстрока"]   # создать/пересобрать конспект(ы)
llmlab notes export [--out llmlab-notes.zip]             # vault для Obsidian
```

После зелёного `llmlab test` конспект обновляется; если его ещё нет, в терминале спрашивается
«Главное, что я понял» (в неинтерактивном режиме печатается подсказка с `notes build`).

### 4.5.1 HTTP API (`llmlab/server.py`, только 127.0.0.1)

| метод | путь | что делает |
|-------|------|------------|
| GET | `/api/course` | блоки → модули → задачи со статусами, `issues` загрузчика |
| GET | `/api/tasks`, `/api/tasks/{id}` | список / детали (README, hints по `##`, reading, papers, viz, `diff_unlocked`) |
| GET/PUT | `/api/tasks/{id}/code` | рабочая копия; GET создаёт её из заготовки; PUT с `base_sha256` → 409, если файл изменён снаружи (`force: true` — перезаписать) |
| GET | `/api/tasks/{id}/code/meta` | sha256/mtime файла — фронтенд опрашивает раз в 1.5 с, чтобы подхватить правки из IDE |
| POST | `/api/tasks/{id}/reset` | сброс к заготовке (`buggy.py` для debug) |
| POST | `/api/tasks/{id}/run` | прогон тестов → RunResult (§4.3), попытка пишется в БД |
| GET | `/api/tasks/{id}/attempts` | история попыток, новые сверху; полный RunResult — у последней |
| GET | `/api/tasks/{id}/reference` | эталон; 403, пока задача не решена и неудачных попыток < 3 |
| GET | `/api/progress`, `/api/viz` | сводка прогресса, визуализации |
| GET | `/api/notes` | конспекты (по файлам `notes/`) + теги с частотой + путь к папке |
| GET | `/api/notes/search?q=` | FTS5-поиск (каждое слово — префикс); совпадения в `snippet` обрамлены `\u0002…\u0003` |
| GET | `/api/notes/{id}` | frontmatter (`meta`), тело без frontmatter, `insight`, абсолютный путь |
| POST | `/api/notes/{id}` | `{insight?}` → создать/пересобрать; 409 `needs_insight` / `not_solved`, 422 — неверный вывод |
| POST | `/api/notes` | пересобрать все конспекты |
| GET | `/api/notes/export.zip` | zip всей папки `notes/` |
| GET | `/api/architectures` | таблица архитектур; закрытые ячейки без значения |

`POST /run` при `status: passed` добавляет к RunResult поле `note`: `{"status": "needs_insight"}`
(конспекта ещё нет → UI показывает модалку) или `{"status": "updated", "path": …}`.

Изменяющие запросы требуют заголовок `X-LLMLab: 1` (защита от CSRF), `Host` — только
localhost/127.0.0.1 (защита от DNS rebinding). Документация OpenAPI — `/api/docs`.

### 4.6 Инварианты `llmlab validate`

Для каждой задачи: ошибки загрузчика по задаче → FAIL. Для задач с кодом:

- `reference ✓` — эталон проходит все тесты (и тестов > 0);
- `starter ✗` (implement/project) — на заготовке не проходит **ни один** тест;
- `buggy ✗` (debug) — багованный код падает хотя бы на одном тесте.

### 4.7 Конспекты (`llmlab/notes/`)

Файл `notes/<block>/<task_id>.md` создаётся после первого зелёного прогона, когда пользователь
ввёл «Главное, что я понял» (1–3 строки, ≤ 600 символов — без этого конспекта нет). Разделы:

| раздел | откуда |
|--------|--------|
| frontmatter: `id, title, aliases, block, module, type, difficulty, tags, solved_at, attempts, parent, depends_on, updated, source` | task.yaml + попытки; `parent`/`depends_on` — `"[[wiki-ссылки]]"` |
| строка связей: Родитель / Опирается на / Дальше | `diff_from`, `depends_on`, обратные зависимости |
| TL;DR, Формула, Формы тензоров, Что поменялось… | `summary.md` |
| Мой код | рабочая копия, если её sha совпадает с зелёной попыткой, иначе снимок из БД |
| Отличия от эталона | unified diff в свёрнутом callout `> [!example]-` |
| Мои ошибки по пути | попытки до последней зелёной: ошибки группируются по тексту (числа не различаются), шум `NotImplementedError` отброшен |
| Главное, что я понял | ввод пользователя, списком |
| Вопросы для повторения | `questions_for(task)`, каждый — свёрнутый callout `> [!question]-` с ответом |
| Читать | `papers` + `reading` |

Пересборка (`notes build`, повторный зелёный прогон, кнопка в UI) перезаписывает всё, кроме
«Главное, что я понял»: правка этого раздела прямо в файле (Obsidian) побеждает и сохраняется в
БД. `notes/index.md` — оглавление vault'а и открытая часть таблицы архитектур, пересобирается
автоматически. Конспекты создаются только для задач с тестами (implement/debug/project).

---

## 5. Чек-лист качества задачи

**Содержание**
- [ ] Решается за 15–60 минут; если больше — разбить на две задачи.
- [ ] README: мотивация, «что поменялось», формула, схема форм, задание, критерии.
- [ ] Текст свой, без копирования статей/блогов; на источники — ссылки.
- [ ] Гиперпараметры реальных моделей — с источником или `TODO-verify`.
- [ ] Термины на английском (attention, KV-cache), объяснения на русском.
- [ ] Подсказки — по нарастающей, без полного решения.
- [ ] `summary.md` — 4 обязательных раздела.

**Тесты**
- [ ] Проверяются формы, численное совпадение и хотя бы одно свойство механизма.
- [ ] Каждое сообщение об ошибке объясняет, что сломано и почему.
- [ ] Сознательно сломай эталон 2–3 типичными способами (`llmlab test <id> --impl-path …`)
      и убедись, что тесты падают с понятным текстом.
- [ ] Фиксированные сиды/генераторы, явные `atol/rtol`, игрушечные размеры, < 2 с на тест.
- [ ] Каждый тест вызывает реализацию (иначе он пройдёт на заготовке).

**Инварианты**
- [ ] `llmlab validate <id>` зелёный.
- [ ] Для debug: баг один, правдоподобный, README описывает симптом, а не причину.
