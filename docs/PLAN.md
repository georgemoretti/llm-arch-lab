# План: LLM Arch Lab

Работа разбита на фазы. **Одна фаза — одна сессия Claude Code** (большие фазы делятся на
подфазы a/b/c). Между фазами `/clear`. Состояние передаётся через `docs/PROGRESS.md`.

Порядок выбран так, чтобы как можно раньше получить работающий сквозной сценарий
(задача → тесты → конспект), а потом наращивать контент.

---

## Фаза 0 — Каркас и формат задач

**Цель:** репозиторий, пакет, CLI, раннер тестов, формат задач, одна демо-задача от начала до
конца, но пока без UI.

- `pyproject.toml` (uv/pip), зависимости: torch (CPU), pytest, fastapi, uvicorn, typer,
  pyyaml, pydantic, safetensors, huggingface_hub (опционально), rich.
- `llmlab/content_loader.py`: pydantic-модели `Task`, `Question`, `Paper`, загрузка
  `content/**/task.yaml`, валидация графа `depends_on` (без циклов, все id существуют).
- `conftest.py` с опцией `--impl` и фикстурой `impl` (см. CLAUDE.md).
- `llmlab/runner/`: запуск pytest в subprocess с таймаутом, JSON-отчёт (pytest-json-report
  или собственный плагин), сохранение попытки в SQLite (время, прошедшие/упавшие тесты,
  тексты ошибок, хэш кода).
- `llmlab validate`: проверка инвариантов (reference ✓ / starter ✗ / debug-bug ✗).
- `llmlab init-workspace <id>`: копирует starter в `workspace/`.
- Демо-задача `C00-causal-mha` (вспомнить causal multi-head attention из GPT-2) со всеми
  файлами.
- `docs/TASK_FORMAT.md`: подробная спецификация для авторов задач.
- GitHub Actions: `pytest tests/` + `llmlab validate --all`.

**Готово, когда:** `llmlab test C00-causal-mha` работает, `validate --all` зелёный, CI зелёный.

---

## Фаза 1 — Локальный веб-интерфейс (MVP)

**Цель:** пользователь открывает браузер и проходит задачу целиком, не трогая терминал.

- FastAPI: `/api/tasks`, `/api/tasks/{id}`, `/api/tasks/{id}/code` (GET/PUT workspace-файла),
  `/api/tasks/{id}/run`, `/api/tasks/{id}/attempts`, `/api/progress`, `/api/notes`.
- Защита: слушать только 127.0.0.1; проверять, что пути не выходят за `workspace/`.
- Frontend:
  - **Карта курса:** блоки → модули → задачи, статусы, граф зависимостей, метка «что
    поменялось» (diff_from).
  - **Страница задачи:** слева README (markdown + KaTeX), вкладки «Подсказки» (по одной),
    «Чтение», «Визуализация»; справа Monaco с кодом пользователя, кнопки Run / Reset to
    starter / Show diff with reference (только после решения или 3 неудачных попыток);
    панель результатов тестов с человекочитаемыми ошибками.
  - Автосохранение кода; горячая клавиша Ctrl+Enter = Run.
  - Тёмная/светлая тема.
- Пользователь также может редактировать `workspace/` в своём IDE: UI отслеживает изменения
  файла и подхватывает их.
- Каркас для визуализаций: реестр компонентов `viz/<id>` и вставка в README через директиву
  `:::viz rope-rotation:::`.

**Готово, когда:** C00 проходится полностью через UI, бандл собран в `llmlab/static/`.

---

## Фаза 2 — Конспекты (вкладка Notes)

**Цель:** после того как все тесты задачи прошли, автоматически создаётся конспект.

Конспект `notes/C/C03-rope.md`:
```
---
id: C03-rope
solved_at: 2026-10-20
attempts: 4
tags: [positional-encoding]
parent: "[[C00-causal-mha]]"
---
# RoPE
## TL;DR            ← из summary.md (авторское)
## Формула          ← из summary.md
## Формы тензоров   ← из summary.md
## Что поменялось относительно GPT-2   ← из summary.md
## Мой код          ← финальная версия из workspace
## Отличия от эталона ← unified diff (свёрнут)
## Мои ошибки по пути ← автоматически из истории попыток: какие тесты падали и почему
## Главное, что я понял ← ОБЯЗАТЕЛЬНОЕ поле, пользователь вводит 1–3 строки в модалке
## Вопросы для повторения ← ссылки на карточки
## Читать              ← ссылки
```

- UI: вкладка **Notes** — список/дерево, полнотекстовый поиск (SQLite FTS5), фильтр по тегам,
  граф связей (кто от кого произошёл), просмотр одного конспекта; кнопка «Экспорт всего в
  zip/Obsidian vault».
- **Таблица архитектур:** ячейки «норма / позиционные / attention / FFN / MoE» по моделям
  открываются по мере прохождения курса.
- Конспект можно пересоздать (`llmlab notes build`), пользовательское поле при этом
  сохраняется.

**Готово, когда:** после решения C00 появляется конспект, он ищется и экспортируется.

---

## Фаза 3 — Блок C / Модуль 1: GPT-2 → Llama

Базовая модель `llmlab/core/gpt2.py` даётся готовой (это отправная точка, сравнение идёт с ней).

| id | Задача | Тип |
|----|--------|-----|
| C01 | RMSNorm (vs LayerNorm) | implement |
| C02 | SwiGLU FFN (vs GELU MLP) | implement |
| C03 | RoPE: частоты, rotate_half, применение к q/k; тест на относительность | implement |
| C04 | GQA: группировка KV-голов, repeat_interleave; MQA как частный случай | implement |
| C05 | KV-cache: пошаговая генерация == полный прогон | implement |
| C06 | Собрать Llama-блок и модель (дифф к GPT-2) | implement |
| C01d–C06d | Отладка: пропущен 1/√d; перепутаны оси/половины в RoPE; утечка через маску; сдвиг позиций при KV-cache; eps внутри/вне sqrt в RMSNorm | debug |
| C-P1 | Статьи: RoFormer, GQA | paper |
| C-Q1 | Вопросы с собеседований по модулю | quiz |

Визуализации: **V-rope** (вращение пар координат, зависимость score от m−n),
**V-kvcache** (калькулятор размера KV-cache: MHA / MQA / GQA по длине контекста, пресеты по
реальным конфигам).

Подфазы: **3a** = C01–C03 + их debug-задачи, **3b** = C04–C06 + debug + визуализации + paper/quiz.

---

## Фаза 4 — Модуль 2: Gemma 3 / Olmo / SmolLM3

| id | Задача |
|----|--------|
| C07 | Sliding window attention: маска с окном, паттерн local:global по слоям (5:1, 3:1) |
| C08 | Размещение нормализаций: Post-Norm (оригинал), Pre-Norm (GPT-2), Post-Norm в стиле Olmo 2, Sandwich-norm (Gemma 3); тест на стабильность градиента при инициализации |
| C09 | QK-Norm (до RoPE) |
| C10 | NoPE: слои без позиционного кодирования, применение через каждый N-й слой |
| C11 | KV-cache для SWA: кольцевой буфер |
| debug | off-by-one в окне SWA; QK-Norm после RoPE; global-слои по неверному индексу |
| paper/quiz | Longformer (SWA), NoPE-статья; вопросы |

Визуализация: **V-masks** — интерактивные матрицы масок: causal, SWA (слайдер окна),
local/global по слоям, позже sparse. Расширение V-kvcache на SWA.

---

## Фаза 5 — Модуль 3: MoE и DeepSeek

| id | Задача |
|----|--------|
| C12 | Роутер: логиты → softmax → top-k → перенормировка |
| C13 | MoE-слой: диспетчеризация токенов по экспертам (наивная и через сортировку/scatter) |
| C14 | Балансировка нагрузки: auxiliary loss (Switch-style); упоминание bias-based подхода из DeepSeek V3 |
| C15 | Shared expert + первые dense-слои |
| C16 | MLA: сжатие KV в латент, восстановление; затем decoupled RoPE |
| C17 | MTP: дополнительные головы для t+2..t+k, суммарный лосс |
| debug | softmax по неверной оси; нет перенормировки top-k; токены теряются при переполнении capacity; в MLA кешируется не латент |
| paper/quiz | Switch Transformer, DeepSeekMoE, DeepSeek-V2 (MLA), MTP |

Визуализации: **V-moe-routing** (heatmap «токены × эксперты», гистограмма загрузки
экспертов, эффект aux loss), V-kvcache + MLA.

Подфазы: **5a** = C12–C15, **5b** = C16–C17 + визуализации.

---

## Фаза 6 — Модуль 4: эффективное внимание и гибриды

| id | Задача |
|----|--------|
| C18 | Упрощённое sparse attention в стиле DSA: индексатор + top-k селектор → маска |
| C19 | Gated attention: выходной гейт, partial RoPE |
| C20 | Линейное внимание (kernel trick), рекуррентная форма == параллельная |
| C21 | DeltaNet → Gated DeltaNet (рекуррентная форма, игрушечный размер) |
| C22 | Гибридный стек 3:1 (linear : full), сравнение памяти |
| C23* | (hard, опционально) упрощённый Mamba-2 / SSM-скан |
| debug | утечка будущего в рекуррентной форме; неверная нормировка в линейном внимании |
| paper/quiz | Transformers are RNNs, Gated DeltaNet, Gated Attention, Mamba-2 |

Визуализации: sparse-маска в V-masks; **V-hybrid** (схема стека и кривая памяти по длине
контекста для full / SWA / hybrid).

Подфазы: **6a** = C18–C20, **6b** = C21–C23 + визуализации.

---

## Фаза 7 — База вопросов и повторение

- `llmlab questions import <file>`: импорт сырой базы пользователя (любой формат: md/csv/json/
  txt). Сначала скрипт нормализации, затем проход Claude для разметки (тема, сложность,
  привязка к задачам, эталонный ответ).
- Нормализованный формат `content/questions/*.yaml`:
  ```yaml
  - id: rope-relative-01
    topic: positional-encoding
    tasks: [C03-rope]
    difficulty: medium
    kind: open            # open | mcq | code-reading
    question: "..."
    answer: "..."         # эталон (свой текст)
    options: []           # для mcq
    source: user-db       # откуда вопрос
  ```
- UI: вкладка **Interview**: квиз после модуля, режим флеш-карточек, интервальное
  повторение (SM-2), самооценка для open-вопросов (Again/Hard/Good/Easy).
- Вопросы попадают в конспекты задач.

---

## Фаза 8 — Финальный проект: конфигурируемая модель + реальные веса

- `llmlab/core/model.py`: одна модель, собираемая из YAML через реестр компонентов
  (norm, pos-encoding, attention, ffn/moe, порядок слоёв, паттерн local/global/linear).
- Готовые конфиги: `gpt2.yaml`, `llama-like.yaml`, `gemma3-like.yaml`, `qwen3-0.6b.yaml`,
  `deepseek-tiny.yaml`, `hybrid-tiny.yaml`.
- Задача-проект: загрузить веса **Qwen3-0.6B** (safetensors с HF), сопоставить имена
  параметров, сравнить логиты с `transformers` (`@pytest.mark.weights`), сгенерировать текст.
  Все параметры конфига проверить по официальному `config.json`.
- Бонус: загрузить веса ещё одной маленькой открытой модели другой архитектуры.
- Итоговый конспект «Моя модель».

---

## Фаза 9 — Полировка и выпуск

- README с GIF, быстрый старт, скриншоты; CONTRIBUTING (как добавить задачу); NOTICE
  (атрибуции); LICENSE.
- Проверка установки с нуля на чистом окружении (Linux/macOS/Windows).
- `llmlab doctor`: проверка окружения.
- Тег v0.1.

---

## Позже (блоки A, B, D)

A — фундамент (autograd, MLP, BPE с нуля); B — энкодеры (Transformer enc-dec, BERT, RoBERTa,
DistilBERT, T5, ModernBERT, Sentence-BERT); D — обучение и инференс (SFT, LoRA, DPO, GRPO,
квантизация, speculative decoding). Формат задач тот же.
