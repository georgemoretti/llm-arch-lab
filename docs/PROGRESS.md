# Прогресс

> Обновляется в конце КАЖДОЙ сессии Claude Code. Пиши кратко и конкретно.

## Текущая фаза
**Фаза B3 завершена** (2026-10-18): блок B, модуль 3 «DistilBERT и T5» (8 задач). Ждёт просмотра пользователем.

Предыдущее: Фаза B2 (модуль «BERT»).

## Следующий шаг (точный)
Фаза **B4** — модуль 4 «ModernBERT и эмбеддинги текста» (`content/blocks/B/module-4-modernbert/`, id модуля уже в
block.yaml): B16 слой ModernBERT (RoPE, GeGLU, без bias, Pre-LN, чередование global / local — окно и шаг из
`answerdotai/ModernBERT-base` config.json; == `transformers.ModernBertModel` на случайных весах, если класс есть в
установленной версии), B17 unpadding (`cu_seqlens`, блочно-диагональная маска == батч с паддингом), B18 sentence-эмбеддинги
(mean / CLS pooling, L2, bi- и cross-encoder), B19 InfoNCE с in-batch негативами, B20 проект «мини-поиск по корпусу курса»,
B17d (`cu_seqlens` со сдвигом), B18d (mean pooling с PAD), B-P4, B-Q4 (+ итоговый по блоку), V-unpadding,
V-embedding-space. В конце B4 — таблица «энкодеры → декодеры», CHANGELOG, выпуск v0.3.0. Вопросы user-db
`iv-retrieval-*`, `iv-embeddings-*` — привязать глобами. В кеше HF уже есть `sentence-transformers/all-MiniLM-L6-v2`,
`BAAI/bge-small-en-v1.5` — можно взять для weights-тестов B18.

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

**Фаза 2:**
- `llmlab/notes/`: `generator.py` (конспект из summary.md + код + diff + «Мои ошибки по пути»
  из попыток + «Главное, что я понял»; `notes/index.md` — оглавление + таблица архитектур),
  `index.py` (FTS5-индекс с синхронизацией по mtime, фолбэк-скан без FTS5, экспорт zip),
  `arch.py` (таблица «модель × компонент»).
- SQLite v2: `notes`, `notes_index`; FTS5-таблица `notes_fts` создаётся лениво.
- Контент: `content/architectures.yaml` (6 моделей × 5 компонентов, ссылки на config.json),
  поле `arch` в task.yaml; C00 открывает строку GPT-2.
- API: `/api/notes` (список+теги), `/search`, `/{id}` (GET/POST), `POST /api/notes`
  (пересобрать все), `/export.zip`, `/api/architectures`; `POST /run` возвращает `note`.
- CLI: `llmlab notes build [id] [--insight]`, `llmlab notes export`; `llmlab test` после
  первого зелёного прогона спрашивает вывод в терминале (если TTY).
- UI: вкладка «Конспекты» (`#/notes`): дерево блок→модуль, поиск с подсветкой, фильтр по тегам,
  просмотр (callout-блоки Obsidian, wiki-ссылки), «Изменить вывод», граф связей, таблица
  архитектур, экспорт zip, «Пересобрать»; модалка «Главное, что я понял» после первого
  `passed`; кнопка «📝 Конспект / Создать конспект» в шапке задачи.
- Тесты: `tests/test_notes.py` (13) + 2 в `test_server.py`; всего 57.
- E2E (Playwright из scratchpad, вне репо): C00 → неудача → правка «из IDE» → Ctrl+Enter →
  модалка (кнопка неактивна без текста) → конспект → поиск «дисперс» → граф → таблица →
  фильтр по тегу → «Изменить вывод» → тёмная тема → экспорт; 1440 и 820 px.
- Исправлен `.gitignore`: `notes/` игнорировал и `llmlab/notes/` → пути привязаны к корню.

**Фаза 3a:**
- Модуль `module-1-llama`: 6 задач, у каждой все файлы (task.yaml, README, starter/buggy,
  solution, test_task, hints, summary).
  - `C01-rmsnorm` (13 тестов): `rms_norm` + `RMSNorm`; свойства — RMS ≈ 1, инвариантность к
    масштабу, нет центрирования, нулевой вход, eps под корнем на |x| ~ 1e-3, fp16 с |x| ~ 300
    (переполнение x² → нужен fp32); диагност различает LayerNorm, std, eps-после-корня, L2-норму,
    неверную ось. `F.rms_norm`/`nn.RMSNorm` запрещены.
  - `C02-swiglu` (13): `silu`, `llama_ffn_hidden_dim` (11 008 для Llama 2 7B, 14 336 для Llama 3 8B),
    `SwiGLUFFN` (gate/up/down, имена как в HF); свойства — линейность ветки up, закрытый gate → 0,
    нет bias; диагност различает перепутанные ветки, GeGLU, GLU, bilinear. `F.silu` запрещена.
  - `C03-rope` (11): `rope_inv_freq`, `rope_cos_sin(positions, …)`, `rotate_half`, `apply_rope`;
    оракул через комплексные числа; свойства — позиция 0 = identity, D = 2 → поворот на m рад,
    сохранение нормы, score зависит только от m − n, θ = 500 000 и смещённые позиции.
  - Debug: `C01d` (eps после корня), `C02d` (SiLU на ветке up), `C03d` (углы через
    `repeat_interleave` при раскладке половинами). Тесты = тесты implement-задачи, но
    диагност не называет баг.
- Статьи: `rmsnorm`, `llama`, `llama3`, `glu-variants`, `roformer`.
- Вопросы: `content/questions/llama-basics.yaml` (8: rmsnorm-*, swiglu-*, rope-*).
- Таблица архитектур: строка Llama 3 — norm, position, ffn (из config.json).
- Проверено: каждую задачу ломал 3–4 типичными способами (`--impl-path`) — падают с понятным
  текстом; сквозной сценарий в копии репо: init-workspace → test → notes build → конспект с
  «Моими ошибками» и вопросами, ячейки Llama 3 открываются в `notes/index.md`.

**Фаза 3b:**
- `llmlab/core/llama.py` — эталоны C01–C05 (RMSNorm, SwiGLU, RoPE, repeat_kv, kv_cache_bytes,
  KVCache, LlamaAttention) для поздних задач и Фазы 8; `tests/test_core_llama.py` сверяет их с
  `solution.py` задач.
- `C04-gqa` (12 тестов): `repeat_kv`, `kv_cache_bytes` (Llama 2 7B → 2 ГиБ, Llama 3 8B → 1 ГиБ),
  `GroupedQueryAttention` (q/k/v/o_proj без bias); оракул — sdpa; свойства: MHA/MQA как частные
  случаи, GQA = MHA с общими весами K/V в группе, causal, независимость батча; диагност: раскладка
  «по кругу», нет √D, √d_model, нет маски. `F.sdpa` запрещён.
- `C05-kv-cache` (9): `KVCache.update`, `LlamaAttention.forward(x, cache)` (GQA + RoPE + кеш);
  свойства: по одному токену / prefill+decode / куски 4+3+5 == полный прогон, в кеше H_kv голов
  и повёрнутые k, кеши не делят состояние; диагност: позиции с 0, маска от левого верхнего угла
  (в т. ч. `is_causal=True`), нет маски, кеш не используется.
- `C06-llama-block` (8): `LlamaBlock`, `LlamaModel` (имена HF), `generate` дан; тесты грузят веса
  эталона с неединичными весами норм; свойства: identity при нулевых o_proj/down_proj, tie
  embeddings, causal, кеши по слоям, generate с кешем == без; диагност: Post-Norm, остаток от
  norm(x), нет остатков, одна норма на оба подслоя, параллельный блок, нет финальной нормы,
  общий кеш на все слои.
- Debug: `C04d` (`.repeat` вместо repeat_interleave — MHA и MQA работают, GQA нет),
  `C05d` (позиции RoPE = arange(T) при кеше), `C06d` (остаток берётся от norm(x)).
- `C-P1-roformer-gqa` (paper: RoFormer, MQA, GQA; 7 вопросов), `C-Q1-llama-quiz` (22 вопроса
  модуля по глобам). Статьи `gqa`, `mqa`, `llama2`. Вопросы модуля: +14 в `llama-basics.yaml`,
  `papers-module-1.yaml`.
- Вопросы: поле `correct` для mcq (валидация в загрузчике); варианты с ✅ в конспекте;
  `/api/tasks/{id}` отдаёт `questions`; вкладка **«Вопросы»** на странице задачи.
- Визуализации: `rope-rotation` (диски всех пар, плоскость выбранной пары, сдвиг обеих позиций,
  кривая score от расстояния) → C03; `kv-cache` (6 пресетов из config.json, контекст/batch/dtype,
  MHA/GQA/MQA на одном графике) → C04, C05. Общий `web/src/viz/LineChart.tsx` (hover-тултип,
  легенда, подписи на концах). Цвета серий `--series-1..3` (проверены валидатором палитры, обе темы).
- Граф зависимостей сжимается под ширину; задачи модуля упорядочены: implement → его debug, paper/quiz в конце.
- Таблица архитектур: Llama 3 — attention (C04), moe = «нет, dense» (C06). Строка Llama 3 заполнена.
- Проверено: мутации эталонов C04 (5), C05 (7), C06 (7) падают с понятными сообщениями;
  скриншоты UI (карта, обе viz, квиз, тёмная тема, 860 px) — без ошибок в консоли; сквозной
  сценарий в копии репо: C04 решена → конспект с mcq-вариантами и ячейкой таблицы.
- Тесты платформы: 64.

**Фаза 4:**
- `llmlab/core/gemma_olmo.py` — эталоны C07–C11 (`sliding_window_mask`, `layer_types`,
  `SlidingWindowAttention`, 4 блока с разным местом нормы, `QKNormAttention`, `rope_layers`,
  `NoPEAttention`, `build_attention_layers`, `RingKVCache`, `SlidingWindowCachedAttention`,
  `swa_kv_cache_bytes`); `tests/test_core_gemma_olmo.py` сверяет их с `solution.py` задач.
- `C07-swa` (11 тестов): маска-окно `[T, S]` со сдвигом, расписание Gemma (`layer_types`), SWA-attention;
  свойства: W=1 → диагональ, W ≥ T → `LlamaAttention`, локальность, 3 слоя видят на 3·(W−1) назад
  (через градиент); диагност: окно ±1, нет causal, окно от начала, `i % pattern`.
- `C08-norm-placement` (7): Post / Pre / OLMo 2 / sandwich; формулы при случайных весах норм, тождество при
  нулевых подслоях, RMS остатка (Post = 1, OLMo/sandwich ≈ √(1+2L)), эффект Xiong et al. (Xavier-init:
  ‖∇W‖ последнего блока у Post не зависит от L, у Pre падает > 2× при L 1 → 16).
- `C09-qk-norm` (6): режимы `head` (Gemma 3) и `full` (OLMo 2), норма до RoPE, |score| ≤ √D,
  инвариантность к масштабу W_Q/W_K.
- `C10-nope` (8): `rope_layers` = `no_rope_layers` SmolLM3, NoPE = «мешок слов» для последнего токена,
  но различает позиции через маску; RoPE-слой = `LlamaAttention`.
- `C11-swa-kv-cache` (10): кольцевой буфер (ячейка p % W, позиции −1 для пустых), `update` возвращает
  буфер ДО записи + новые, маска по абсолютным позициям; по токену / prefill 11 + decode / куски /
  W=1 / W>T == полный SWA; `swa_kv_cache_bytes` (Gemma 3 27B на 128K: 10.4 ГиБ против 62).
  Диагност — симуляция трёх типичных ошибок (позиции RoPE, пустые ячейки, маска по индексам).
- Debug: `C07d` (окно на 1 шире: `k_pos >= q_pos - W`), `C09d` (QK-Norm после RoPE — видно только при
  неединичных весах норм), `C10d` (NoPE-слои `i % k` вместо `(i+1) % k`).
- `C-P2-longformer-nope` (Longformer, NoPE-статья, Xiong et al.; 6 вопросов), `C-Q2-gemma-olmo-quiz`
  (19 вопросов модуля по глобам). Статьи: `longformer`, `mistral-7b`, `gemma-2`, `gemma-3`, `olmo-2`,
  `nope-length-generalization`, `rope-to-nope`, `layernorm-placement`, `qk-norm`, `vit-22b`.
- Вопросы: `content/questions/gemma-olmo.yaml` (19), `papers-module-2.yaml` (6).
- Таблица архитектур: + строка SmolLM3; ячейки Gemma 3 (attention, position — C07; norm — C08),
  OLMo 2 (norm — C08; attention, position — C09), SmolLM3 (position, attention — C10).
- Визуализации: `masks` (V-masks: маска слоя со слайдером окна, расписание слоёв 5 моделей с
  sliding/full/NoPE, поле зрения стека) → C07, C07d; `kv-cache` расширен пресетами Mistral 7B, Gemma 2 9B,
  Gemma 3 4B/27B и флажком «учитывать окно SWA» (плитка «без окна ×N»).
- Проверено: мутации эталонов C07 (4), C08 (5), C09 (2), C11 (5) падают с понятными сообщениями;
  152 формулы новых README/summary проходят KaTeX; скриншоты V-masks (3 режима) и V-kvcache (Gemma 3,
  Mistral) в обеих темах, 860 px — без ошибок в консоли; сквозной сценарий в копии репо: C09 решена →
  конспект с вопросами, ячейки OLMo 2 открыты в `notes/index.md`.
- Тесты платформы: 69.

**Фаза 5a:**
- `llmlab/core/moe.py` — эталоны C12–C15 (`route`, `TopKRouter`, `MoELayer` (наивный и сортировочный
  forward), `load_balancing_loss`, `expert_load`, `route_with_bias`, `update_bias`, `mlp_layer_types`,
  `DeepSeekMoE`, `build_ffn_layers`, `moe_param_counts`); `tests/test_core_moe.py`.
- `C12-router` (8 тестов): softmax/sigmoid, top-k, перенормировка, `routed_scaling_factor`; свойства —
  независимость от батча, softmax(top-k логитов) = перенормированный top-k, fp32 при bf16, градиент к
  роутеру (top-1 с перенормировкой → 0 — почему Switch не перенормирует).
- `C13-moe-layer` (6): `forward_naive` и `forward` через argsort/bincount/index_add; хуками проверяется,
  что эксперт видит ровно свои токены (Σ = N·k), невыбранный эксперт не вызывается и без градиента.
- `C14-load-balancing` (9): aux loss Switch-style (баланс → 1, коллапс → E, = HF Mixtral / k, градиент
  только через P, 30 шагов SGD разгоняют коллапс), bias-балансировка DeepSeek-V3 (выбор с bias, веса без;
  ±γ по знаку; симуляция 300 шагов выравнивает нагрузку).
- `C15-shared-expert` (7): `mlp_layer_types` (V3: 3 dense + 58 MoE), `DeepSeekMoE` = routed + shared,
  `build_ffn_layers`, `moe_param_counts` (V3 FFN: 657.8B / 24.3B; Qwen3-30B-A3B: 29.0B / 1.82B).
- Debug: `C12d` (softmax по оси токенов — маскируется перенормировкой), `C13d` (`out[tok] = y` вместо
  `index_add_` — видно только при top-k ≥ 2), `C14d` (знак шага bias — положительная обратная связь).
- Статьи: `sparsely-gated-moe`, `gshard`, `switch-transformer`, `st-moe`, `mixtral`, `deepseekmoe`,
  `deepseek-v2`, `deepseek-v3`, `aux-loss-free`, `olmoe`. Вопросы: `content/questions/moe.yaml` (15).
- Таблица архитектур: `deepseek-v3.ffn` (C13), `deepseek-v3.moe`, `gemma-3.moe`, `olmo-2.moe`,
  `smollm3.moe` (C15, «нет — dense» закрывает долг Фазы 4).
- Проверено: мутации C12 (4), C13 (3), C14 (4), C15 (3) падают с понятными сообщениями; 108 формул —
  KaTeX без ошибок; сквозной сценарий в копии репо: C15 решена → конспект, ячейки MoE открыты.
- Тесты платформы: 73.

**Фаза 5b:**
- `llmlab/core/mla.py` — эталоны C16, C16b, C17 (`LatentCache`, `causal_mask`, `MLAttention` с
  `forward_absorbed`, `MLARopeAttention`, `mla_kv_cache_bytes`, `MTPModule`, `mtp_logits`, `mtp_loss`);
  `tests/test_core_mla.py`.
- `C16-mla` (8 тестов): латент c = norm(W_dkv x), K/V всех голов из c; в кеше латент [B, S, r]; по токену и
  кусками == полный прогон; `forward_absorbed` (W_uk в запрос, W_uv в выход) == forward и не вызывает kv_b_proj.
- `C16b-mla-rope` (6): decoupled RoPE (общий RoPE-ключ из `kv_a_proj_with_mqa`, масштаб 1/√(dn+dr)),
  кеш = латент + повёрнутый k_rope без оси голов; `mla_kv_cache_bytes` (V3: 70 272 Б/токен, ×71 к MHA).
- `C17-mtp` (7): модуль `eh_proj([enorm(emb); hnorm(h)])` → Pre-Norm блок, цепочка глубин, общие
  embed/lm_head, причинность (токен j влияет на глубину k с позиции j − k), loss L_main + λ/D·Σ L_k,
  градиент в h_main.
- Debug: `C16d` (в кеш кладутся восстановленные K/V — выход верный, память ×H(d+dv)/r).
- `C-P3-moe-deepseek` (Switch, DeepSeekMoE, DeepSeek-V2, DeepSeek-V3; 6 вопросов в `papers-module-3.yaml`),
  `C-Q3-moe-quiz` (22 вопроса модуля по глобам). Вопросы: + 7 в `moe.yaml` (mla-*, mtp-*).
- Таблица: `deepseek-v3.attention`, `deepseek-v3.position` (C16b). Строка DeepSeek-V3 заполнена, кроме `norm`.
- Визуализации: `moe-routing` (V-moe-routing: 32 токена × 8 экспертов, top-k, гистограмма загрузки,
  шаги aux loss / bias, всё считается в браузере) → C12, C13, C14 (встроена в README C14); V-kvcache:
  пресет DeepSeek-V3 (MLA) — сравнение с MHA тех же голов (×71); встроен в README C16b.
- Проверено: мутации C16 (2), C16b (3), C17 (4) падают с понятными сообщениями; 137 формул — KaTeX без ошибок;
  скриншоты V-moe-routing (старт, 40 шагов aux, 60 шагов bias) и V-kvcache в обеих темах — без ошибок в
  консоли; сквозной сценарий: C16b → конспект, ячейки DeepSeek-V3 открыты.
- Тесты платформы: 76.

**Фаза 6a:**
- `llmlab/core/efficient.py` — эталоны C18–C20 (`LightningIndexer`, `topk_mask`, `sparse_attention`, `DSAttention`,
  `apply_partial_rope`, `GatedAttention`, `feature_map`, `linear_attention_noncausal`, `linear_attention`,
  `linear_attention_recurrent`); `tests/test_core_efficient.py`.
- `C18-sparse-attention` (10 тестов): индексатор Σ_h w·ReLU(q·k)/√(H_I·d_I) с −inf для будущего; `topk_mask`
  (ровно min(k, t+1) ключей, только прошлое, лучшие по оценке; при k ≥ S = causal); `sparse_attention` не зависит
  от невыбранных ключей; `DSAttention` при k ≥ T == плотное causal MHA, при k = 3 == attention по выбранным.
- `C19-gated-attention` (7): `apply_partial_rope` (первые R каналов, частоты для R, остальное без изменений, R = D →
  обычный RoPE, относительность); `GatedAttention` (раскладка `[запрос | гейт]` по головам как в HF, QK-Norm,
  GQA, σ-гейт до o_proj); нулевые веса гейта → выход ровно ½; гейт закрывает голову для одного запроса.
- `C20-linear-attention` (8): φ = elu + 1; некаузальная через φ(K)ᵀV; causal параллельная (v = const → const,
  без утечки); рекуррентная == параллельной, S = φ(K)ᵀV, z = Σφ(k), кусками с переносом state; размер state не
  растёт. Softmax запрещён monkeypatch-ем.
- Debug: `C20d` (знаменатель параллельной формы по ВСЕМ ключам: совпадает с рекуррентной только в последней
  позиции; падает 3/8).
- Статьи: `deepseek-v32` (arXiv 2512.02556), `gated-attention` (2505.06708), `transformers-are-rnns` (2006.16236).
  Вопросы: `efficient.yaml` — 12 (sparse-*, gated-attn-*, partial-rope-*, linear-attn-*).
- Таблица: новая строка `deepseek-v3.2` (attention — C18); `qwen3-next.attention`, `qwen3-next.position` (C19).
- V-masks: режим «Sparse (DSA)» — игрушечные оценки (якорные токены + близость + шум), слайдеры T, k, вес близости,
  подсчёт «выбрано дальше окна W = k»; встроен в README C18.
- Проверено: мутации C18 (без ReLU), C19 (гейт по всей проекции; cos/sin от D с обрезкой) ловятся диагностами с
  точным названием ошибки; скриншоты README C19/C20 и V-masks/Sparse — без ошибок в консоли (нашёл и исправил `|`
  внутри кода в таблице C19).
- Тесты платформы: 79.

**Фаза 6b:**
- `C21-gated-deltanet` (9 тестов): `l2norm`, `gated_delta_rule` (рекуррентная форма как `torch_recurrent_gated_delta_rule`
  в HF: S ← e^g S, запись β(v − Sᵀk), чтение Sᵀq/√dk), слой `GatedDeltaNet` (β = σ(b), g = −e^{A_log}·softplus(a + dt_bias)).
  Свойства: перезапись по ключу (не v₁ + v₂), точное извлечение ортонормированных ключей, β интерполирует, g = 0 ≡ DeltaNet,
  g = −60 стирает прошлое, куски с state, causal.
- `C21d` — выход читается из итогового состояния после цикла (утечка будущего при prefill; по одному токену верно), падает 3/9.
- `C22-hybrid-stack` (7): `hybrid_layer_types` ((i + 1) % 4 == 0 → full), Pre-Norm `HybridBlock` с любым mixer, `HybridStack`,
  `hybrid_cache_bytes`; Qwen3-Next: гибрид = «все full» ровно на 512 токенах, на 256K ≈ ¼.
- `C23-mamba2-ssd` (6, hard, опционально): `segsum`, скан `ssd_recurrent`, квадратичная `ssd_quadratic` == скан; при A = 0 —
  линейное внимание без нормировки.
- `core/efficient.py` теперь содержит эталоны C18–C23 (+ тесты в `test_core_efficient.py`, 6 шт.).
- `C-P4-efficient-attention` (5 групп статей, 7 вопросов в `papers-module-4.yaml`), `C-Q4-efficient-quiz` (21 вопрос по глобам,
  таблица «что видит / память / вычисления»). Вопросы: + 9 в `efficient.yaml` (deltanet-*, hybrid-*, ssm-*).
- Статьи: `deltanet` (2406.06484), `gated-deltanet` (2412.06464), `mamba-2` (2405.21060).
- Таблица: новый столбец `layers` («Чередование слоёв»), его открывает C22 для Qwen3-Next, Gemma 3, Llama 3.
- V-hybrid: полоса слоёв (Llama 3 / Gemma 3 / Qwen3-Next) + кривые памяти «все full / SWA 5:1 (слайдер W) / гибрид 3:1»
  в геометрии Qwen3-Next, контекст 512 … 1M; встроена в README C22.
- Проверено: мутация C21 «накопление как в C20» ловится диагностом в 3 тестах; скриншоты V-hybrid (светлая/тёмная),
  README C21/C23/C-Q4 — без ошибок консоли и KaTeX (обрезанная подпись SWA на графике — исправлена).
- Тесты платформы: 82; задач в курсе: 48.

**Фаза 7:**
- Схема `Question`: + `tags` (подтема для фильтров) и `source_ref` (хэши сырых вопросов, в кавычках).
- `llmlab/questions/importer.py`: парсеры md (заголовки → раздел, пункты списка и строки с «?») / txt / csv
  (автоопределение разделителя, колонки question/answer/topic) / json; нормализация, точная дедупликация по ключу
  (регистр, ё/е, пунктуация), `raw.jsonl` в `.llmlab/` (не в git); статус разметки, следующая пачка, проверка
  дословных совпадений (difflib, порог 0.8). CLI: `llmlab questions import | status | batch | check`.
  `docs/QUESTIONS_IMPORT.md` — маппинг. Тесты на синтетике: `tests/test_questions_import.py` (6).
- Разметка `data/interview_raw/INTERVIEW.md` в 4 пачки: 246 строк → 244 уникальных → **105 вопросов** в
  `content/questions/interview-{transformer,encoders-retrieval,generation,agents,rag-evals,finetuning-embeddings,
  nlp-pipelines}.yaml`, все 244 покрыты (`llmlab questions status`), совпадений с сырым текстом нет даже при пороге
  0.75. Темы: other = 71 (RAG, evals, агенты, retrieval, NLP-пайплайны, …), курс = 34 (attention, позиции, KV-cache,
  инференс/квантизация, генерация, токенизация, обучение); 13 mcq/code-reading; привязки к C00, C01, C03, C04, C05,
  C06, C10, C17, C18, C20 и квизам C-Q1 (6 вопросов) / C-Q4 (1).
- `llmlab/srs.py` (SM-2: again/hard/good/easy → q 2–5, повтор через 10 мин при провале, 1 → 6 → × ease, easy ×1.3);
  SQLite v3: `srs_cards`, `srs_reviews`; `attempts.impl = 'manual'` для пройденных paper/quiz.
- API: `GET /api/interview`, `POST /api/interview/review`, `GET /api/interview/{qid}/history`,
  `POST /api/tasks/{id}/complete`. Тесты: `tests/test_srs.py` (6).
- UI: вкладка **Interview** (`#/interview`, `#/interview/q/<id>`): «Повторение» (очередь due + до N новых, фильтр
  источника и «вне курса», повтор «Не помню» в конце сессии, итог), «Квиз по модулю» (в т. ч. «без привязки к
  задачам»), «Все вопросы» (поиск, источник, тема, статус карточки). Общий `ReviewCard` с кнопками оценки и
  интервалами следующего показа; вкладка «Вопросы» задачи теперь тоже пишет в SRS; кнопка «Отметить пройденным» у
  paper/quiz. Конспект: у каждого вопроса ссылка «повторить в Interview» и пометка «с собеседования».
- Проверено в браузере (Playwright во временной папке): сессия повторения (оценка → 2 / 20, статистика обновилась),
  квиз, список, вкладка «Вопросы» C05 в тёмной теме, отметка paper; ошибок в консоли нет; исправлены кнопки оценки
  (интервал вылезал за рамку).
- Тесты платформы: 94; вопросов в курсе: 219.

**Фаза 8:**
- `llmlab/core/model.py` — модель из YAML: pydantic-схема (`extra=forbid`, понятные ошибки), `layer_specs` через
  расписания эталонов C07/C10/C15/C22, реестр `norm` / `mixer` / `ffn` (`register`, `build`), общая `Attention`
  (GQA, QK-Norm head/full, окно + кольцевой кеш, partial RoPE, NoPE, выходной гейт, bias), MLA, Gated DeltaNet,
  SwiGLU/GeGLU/GELU, MoE; нормы `(1 + w)`; 4 схемы размещения норм; кеши по слоям; имена параметров = HF.
- Конфиги `llmlab/configs/`: gpt2, llama-like, gemma3-like (свой θ у sliding-слоёв), olmo2-like, smollm3-like,
  deepseek-tiny, hybrid-tiny, qwen3-0.6b, smollm2-135m — у каждого источник чисел в комментарии.
- `llmlab/core/hf.py` (= эталоны C24/C25): `config_from_hf` (qwen3, llama, gpt2; transformers 4.x и 5.x),
  `convert_state_dict` (Conv1D → Linear, связанный lm_head), `check_state_dict`, `load_safetensors` (шарды),
  `load_pretrained` (meta + assign + восстановление tie), `sample_next` (T, top-k, top-p), `generate` (KV-cache, eos).
- Новый модуль `module-5-final`: **C24-hf-checkpoint**, **C25-qwen3-project** (17 тестов: 12 на случайных весах +
  5 `@pytest.mark.weights` — число параметров, логиты против transformers, жадная генерация токен в токен, бонус
  GPT-2 и SmolLM2-135M). Строка таблицы архитектур `qwen3` («моя модель», открывает C25). 10 вопросов
  `final-project.yaml`, статья `qwen3`.
- CLI: `llmlab model [имя]` (конфиги, расписание слоёв, параметры), `llmlab generate <модель> "…"`
  (`--chat`, `-t`, `--top-k`, `--top-p`, `--impl workspace`), `llmlab test <id> --weights` (таймаут ≥ 30 мин).
  pytest-флаг `--weights` / `LLMLAB_WEIGHTS=1`; extras `weights` в pyproject, CI ставит их (без скачивания весов).
- Документация: README — раздел «Финальный проект»; `docs/MODEL.md` — схема YAML, реестр, как добавить компонент,
  реальные веса; TASK_FORMAT — `--weights`, дополнительные разделы summary.md.
- Конспект: дополнительные разделы `summary.md` попадают в конспект после четырёх обязательных — у C25 это карта
  «деталь Qwen3 → задача» (итоговый конспект «Моя модель»).
- Проверено: `llmlab test C25-qwen3-project --weights --impl reference` — 17/17 за 75 с на реальных весах;
  `LLMLAB_WEIGHTS=1 pytest tests/test_core_hf.py` — 7/7; `llmlab generate qwen3-0.6b … --chat` — 12.6 ток/с на CPU;
  сквозной сценарий в копии репо: заготовка 0/17 → эталон → конспект с картой, выводом, вопросами; ячейка qwen3
  открыта в `notes/index.md`, вывод сохраняется при пересборке.
- Тесты платформы: 122 (+1 weights); задач: 50; вопросов в курсе: 229.

**Фаза 9:**
- `llmlab doctor` (`llmlab/doctor.py`): Python ≥ 3.11, чужой активированный venv, зависимости, torch (matmul),
  загрузка курса без ошибок, бандл и его ассеты, запись в `.llmlab/` + SQLite/FTS5, свободный порт, пробный прогон
  эталона C00 в subprocess, пакеты `weights` и кеш Qwen3; у каждой проблемы — команда-исправление; exit 1 при fail;
  `--quick` без прогона. `llmlab --version`. Тесты: `tests/test_doctor.py` (7).
- Установка с нуля: свежий клон на macOS через `uv sync` и через `python -m venv && pip install -e .` — doctor зелёный.
  CI: тесты платформы + `validate --all` на ubuntu (3.11, 3.12 + weights), macOS, Windows; job `install`
  (`uv sync` → `llmlab doctor` с прогоном C00) на трёх ОС — всё зелёное. Actions обновлены до Node 24.
- Windows: UTF-8 в раннере (`PYTHONUTF8`, `encoding="utf-8"` у subprocess), `encoding=` во всех `read_text`/
  `write_text` тестов и `core/hf.py`.
- README: GIF (решение C04 от красных тестов до вывода в конспект), 5 скриншотов (`docs/img/`, Playwright на демо-копии
  курса), обзор модулей, установка через uv и pip, «Как это выглядит». `docs/CONTRIBUTING.md`, `CHANGELOG.md`,
  NOTICE (лицензии бандла: 131 пакет — MIT / ISC / BSD-2; лицензии весов Qwen3, SmolLM2, GPT-2). Версия 0.1.0.
- Долг Фазы 4 закрыт: C08 открывает ячейки `gemma-3.ffn` (GeGLU, 5376 → 21 504) и `olmo-2.ffn` (SwiGLU,
  4096 → 11 008), абзац в README C08; числа по config.json.
- UI-фикс: Monaco-редактор уничтожается раньше своей модели (ошибка «Model is disposed!» при уходе со страницы
  задачи с открытым автодополнением).
- Тесты платформы: 129 (+1 weights); задач: 50; вопросов: 229.

**Ревью (2026-10-17):**
- `docs/REVIEW-2026-10-17.md`: сверка с config.json 26 моделей (таблица архитектур, пресеты viz, 171 + 37
  утверждений в текстах и вопросах, пересчёт производных чисел) — ошибок нет; 36 / 36 arXiv-ссылок верны;
  мутационное тестирование 27 задач — 1271 мутант, 88 % убито, 4 реальные дыры; обход тестов импортом эталона
  из `llmlab.core`; сравнение кода с `rasbt/LLMs-from-scratch` (макс. 2 общие строки подряд); поиск англоязычных
  цитат; проверка инвариантов CLAUDE.md. Скрипты (мутатор и др.) — вне репозитория.

**Исправления по ревью (2026-10-17):**
- P1: `check_own_code` в фикстуре `impl` — объекты, которые эталон определяет сам (минус импортируемое заготовкой),
  не могут прийти из `llmlab.*`; тест платформы `test_importing_reference_from_llmlab_is_rejected`; TASK_FORMAT §4.2.
- P1: C14/C14d `test_route_with_bias` — сумма выбранных scores 0.8, отдельная подсказка «не перенормировано».
- P2: явные `atol/rtol` во всех 43 `assert_close` (данные — `0/0`, численные — `1e-6/1e-5`); тест ветки `k_rope`
  в C16/C16d; «Формула» в README C24 и C25; `TODO-verify` C02 снят (копии `original/params.json`); симптомы C16d и
  C01d больше не называют баг; C07 — линейное масштабирование RoPE ×8 у Gemma 3 4B/12B/27B (сверено с config.json).
- Сэмплирование вынесено в **C25a-sampling** (25 мин, 6 тестов: + граница top-p при сумме = p, `top_k > V`);
  C25 — загрузка и генерация (40 мин, `sample_next` импортируется готовым); `llmlab generate --impl workspace`
  берёт `sample_next` из рабочей копии C25a, если она есть; `tests/test_core_hf.py` сверяет и C25a.
- Paper-задачи > 60 мин разделены: C-P2 (50) + **C-P2b** Xiong (25, +2 вопроса), C-P3 (45) + **C-P3b** DeepSeek-V2/V3
  (50), C-P4 (55) + **C-P4b** Gated Attention + DSA (40); вопросы перепривязаны; regex id допускает `C-P2b`.
- P3: ячейка `deepseek-v3.moe` — «bias вместо основного aux loss».
- Проверено: мутанты C14 и C25a, прежде выживавшие, теперь убиты; C25 на реальных весах 13/13; KaTeX новых текстов —
  без ошибок. Тесты платформы: 130 (+1 weights); задач: 54.

**Фаза A1 (блок A, модуль 1 «Autograd»):**
- Платформа: `Block.optional`, `Module.number()` (из id `module-<N>-…`), номера «A1.2» / «C1.3» в `Course.labels()`;
  API `/api/course` отдаёт `optional` и `number` модулей; карта: буква блока, пометка «необязательный», «следующая»
  не уводит в необязательный блок тех, кто уже решает обязательные; «☰ Курс» — номера модулей с буквой блока;
  ссылки на задачи в Markdown — для любого блока (`[A-Z]\d{2}`, `[A-Z]-[A-Z]\d+`); загрузчик: id модулей уникальны во
  всём курсе. Тесты платформы: два блока, уникальность модулей (138 + 1 weights).
- `content/blocks/A/block.yaml` (4 модуля, baseline `scalar-math`), `llmlab/core/autograd.py` (= эталон A01,
  страховка — `tests/test_core_autograd.py`, сравнивает и исходник).
- Задачи: **A01** Value (18 тестов: примитивы, числа с обеих сторон, накопление, порядок, 4 составных выражения vs
  torch, глубина 10 000 без рекурсии), **A01d** (`=` вместо `+=`), **A02** gradcheck (9), **A03** MLP на Value (7: порядок
  parameters, forward/backward vs torch, XOR, кольца), **A04** backprop Linear/ReLU/softmax-CE (7; autograd и готовые
  softmax/CE запрещены), **A04d** (softmax без сдвига), **A05** backward LayerNorm (6, диагност «какой путь забыт»),
  **A-P1** (Rumelhart 1986, Baydin et al. 2018), **A-Q1**. Мутационная проверка: 23 мутанта, все убиты.
- Вопросы: `content/questions/autograd.yaml` (16), `papers-block-a.yaml` (5). Статьи `backprop-1986`, `autodiff-survey`.
- Визуализации: `autograd-graph` (граф по шагам backward), `gradcheck` (ошибка от h в double), `mlp-boundary` (MLP
  учится в браузере), `softmax-ce` (p − y, переполнение наивного softmax), `layernorm-grad` (пути через μ и σ).
- README «Что внутри», CHANGELOG → Unreleased, TASK_FORMAT (`optional`, номер модуля, порядок блоков).
- Проверено в браузере (Playwright, вне репо): карта с двумя блоками, страницы A01–A05, «☰ Курс», визуализации,
  шаги backward, обучение MLP; ошибок в консоли нет.

**Фаза A2 (блок A, модуль 2 «Обучение нейросети»):**
- Задачи: **A06** инициализация (6 тестов: те же числа, что `nn.init`, no_grad, эксперимент по глубине), **A07** SGD /
  momentum / Adam / AdamW (8: совпадение с `torch.optim` за 10 шагов, bias correction, decoupled decay, пропуск
  параметров без grad; `torch.optim.*.step` подменён на время кода пользователя), **A07d** (L2 вместо decoupled), **A08**
  warmup + cosine и clipping (7), **A09** цикл обучения (6: история лоссов = эталон, шпион-оптимизатор проверяет lr и
  норму в момент step(), eval-режим, overfit one batch), **A09d** (нет `zero_grad`), **A10** LM на символах (6: словарь,
  окна, forward = эталон, перплексия равномерной модели = V, MLP лучше биграммной на 10%+), **A-P2**, **A-P2b**, **A-Q2**.
  Мутационная проверка: 28 мутантов, все убиты (один выживший в A07 — подстановка нулевого grad — убит усиленным тестом).
- Корпус `content/data/corpus-ru.txt` (~8.4 тыс. символов, текст написан для курса) + `content/data/README.md`.
- Вопросы: `content/questions/training.yaml` (16), `papers-block-a.yaml` (+7); user-db `iv-training-01/04` → A07,
  `iv-training-02/03` → A09 (через глобы в task.yaml).
- Статьи: `glorot-2010`, `kaiming-2015`, `adam`, `adamw`, `sgdr`, `bengio-2003`.
- Визуализации: `init-depth`, `optimizers`, `lr-schedule`, `bigram-table`; A09 использует `mlp-boundary` и `lr-schedule`.
- Проверено в браузере: карта с модулем 2, вкладки визуализаций A06–A10, ошибок в консоли нет.

**Фаза A3 (блок A, модуль 3 «Токенизация»):**
- `llmlab/core/bpe.py` — эталоны A11–A13 (подсчёт пар, слияние, обучение, словарь, encode/decode, регулярка GPT-2,
  `bytes_to_unicode`, спецтокены); `tests/test_core_bpe.py` сверяет исходник каждой функции с `solution.py` задач.
- Задачи: **A11** обучение byte-level BPE (7 тестов: перекрывающиеся пары, без пар через границы, слева направо без
  перекрытий, ничья → меньшая пара, ранги 256 + k, 120 слияний на корпусе = эталон, сжатие > 2×), **A12** encode по
  рангам + decode через байты (8: «пробел + половина буквы», `errors="replace"`, ранг против частоты на «bcbcabc», encode
  обучающего текста = обучение, roundtrip на эмодзи/иероглифах/случайном Unicode), **A12d** (слияния по частоте в тексте),
  **A13** пре-токенизация GPT-2, спецтокены, byte ↔ unicode (8: 9 примеров регулярки, `Ġ`/`Ċ`, слияния не пересекают
  куски, `|` в спецтокене экранирован, encode по кускам даже при «чужом» merge), **A13d** (decode по одному токену),
  **A14** токенизатор на `vocab.json`/`merges.txt` GPT-2 (5 без сети на игрушечных файлах с перемешанными id + 2 с
  `--weights`: 50 000 слияний, «Hello world» → [15496, 995], совпадение с `tokenizers` на всём корпусе), **A-P3**, **A-Q3**.
  Мутационная проверка: 35 мутантов, все убиты, кроме одного эквивалентного (A11: `continue` вместо `break` при пустых парах).
- Вопросы: `content/questions/tokenization.yaml` (17), `papers-block-a.yaml` (+5); user-db `iv-tokenization-01/03` → A11,
  `iv-tokenization-02` → A13, все три — в A-Q3 (через глобы в task.yaml).
- Статья `bpe-sennrich` (+ существующая `gpt2`). Зависимость `regex>=2023.0` в основных (нужна для `\p{L}`/`\p{N}`).
- Визуализация `bpe` (V-bpe) у всех задач модуля: обучение в браузере, подсветка токенов (‹D0› — несобранный байт),
  последние слияния, кривые «словарь → токенов» с регуляркой и без.
- Проверено в браузере (Playwright из прошлой сессии, вне репо): V-bpe в светлой (1440) и тёмной (860) теме, переключение
  режима, ошибок в консоли нет; 34 формулы новых README/summary проходят KaTeX; вопросы всех задач модуля отдаются API.

**Правка к A3 (2026-10-18):** корпус `content/data/corpus-ru.txt` расширен с ~8.4 до ~23.1 тыс. символов (42 КБ, 67
символов словаря; новый текст написан для курса): разделы про эмбеддинги, токены, attention, слои, позиции, шаг
обучения, генерацию, переобучение и три короткие истории. Пересчитаны: словарь A10 (61 → 67), запас A10 (ppl 10.8 против
14.4 у биграмм), числа README A11/A14, симптомы A12d/A13d, вопрос `charlm-logits-01`.

**Фаза A4 (блок A, модуль 4 «Свой GPT»):**
- `llmlab/core/minigpt.py` — эталоны A15–A17 (TokenPositionEmbedding, tie_weights, count_params, EmbeddingLM,
  causal_mask, attention, AttentionHead, MLP, Block; имена как в `core/gpt2.py`); `tests/test_core_minigpt.py` сверяет
  исходник каждой функции и класса с `solution.py`.
- Задачи: **A15** эмбеддинги + позиции + weight tying (8 тестов: позиции с нуля и общие для батча, ValueError при
  T > block_size, один параметр и сумма градиентов двух путей, `count_params` без двойного счёта, стартовый loss ≈ ln V,
  обучение на корпусе), **A16** одноголовое attention (6: маска, совпадение с SDPA без маски и с ней, нули над
  диагональю, v[0] на позиции 0, Dv ≠ D, отсутствие утечки; диагност: без √D, деление на D, маска после softmax,
  softmax по запросам; SDPA и nn.MultiheadAttention запрещены), **A16d** (маска после softmax — утечка через знаменатель),
  **A17** блок (7: tanh-GELU, pre/post-LN при неединичных LN, тождество pre-LN при нулевых подслоях, нормализованный выход
  post-LN, causal, градиент через 8 блоков; диагност 5 типичных перестановок), **A18** проект мини-GPT (7: forward = эталон
  в обоих режимах, ln V, causal, get_batch с крайними началами, история лоссов train = эталон, overfit one batch, generate
  greedy/сэмпл/обрезка контекста/режим), **A-P4** (Transformer §3), **A-Q4**, **A-Q4b** (итоговый квиз блока + таблица
  «A → C»). Мутационная проверка: 46 мутантов, все убиты, кроме одного некорректно составленного (A15).
- A18 `main`: обучение ~26 с на CPU (190 тыс. параметров, 4 блока, n_embd 64, lr 1e-3), ранняя остановка по val
  (patience 3), лучшие веса; лучший val 2.246 на шаге 1000. `--record` пишет прогон для визуализации.
- Вопросы: `content/questions/gpt.yaml` (18), `papers-block-a.yaml` (+5); user-db `iv-positional-01/02` → A15,
  `iv-attention-01/02/06`, `iv-architecture-03` → A16, `iv-architecture-01` → A17, `iv-generation-01/07` → A18 (глобы).
- Статья `weight-tying` (Press & Wolf 2017). Визуализации: `training-curve` (новая, данные —
  `content/viz/training-curve.data.json`, 3000 шагов без ранней остановки), повторно `bigram-table` (A15), `causal-mask`
  (A16, A16d), `norm-placement` (A17), `sampling` (A18).
- Платформа: поле `recommended_after` (рекомендация, не блокирует, не влияет на «следующую»; проверяется загрузчиком,
  показывается «лучше после: …»); C00 → A16, A18. Тест загрузчика. `resolveJsonModule` в web/tsconfig.
- README: таблица блока A, пересняты скриншоты и GIF (номера «C1.4», блок A на карте), картинка обучения мини-GPT;
  CHANGELOG v0.2.0, версия 0.2.0, тег.
- Проверено в браузере: V-training-curve (обе темы), карта с блоком A, страница C04, Interview; ошибок в консоли нет;
  100 формул модуля 4 проходят KaTeX; все внешние ссылки модуля отвечают 200.

**Фаза B1 (блок B, модуль 1 «Transformer: энкодер и декодер»):**
- План блока B в PLAN.md (фазы B1–B4: энкодер-декодер, BERT, DistilBERT/T5, ModernBERT и эмбеддинги), порядок
  `/next` → B1…B4. `content/blocks/B/block.yaml` (optional, baseline `rnn-seq2seq`); на карте блок B между A и C.
- `llmlab/core/encdec.py` — эталоны B01–B05 (padding_mask, attention, MultiHeadAttention, sinusoidal_positions,
  shift_matrix, TransformerEmbedding, FeedForward, EncoderLayer, Encoder, causal_mask, target_mask, DecoderLayer,
  Decoder, shift_right, label_smoothed_ce, Seq2SeqConfig, Seq2Seq, greedy_decode); `tests/test_core_encdec.py` сверяет
  исходники с `solution.py` задач.
- Задачи: **B01** padding-маска + MHA (8 тестов: маска [B,1,1,S], T ≠ S, оракулы SDPA и nn.MultiheadAttention,
  padding-инвариантность, эквивариантность к перестановке), **B01d** (`view(B, H, L, D)` без transpose), **B02**
  синусоиды (5: формула в float64, ‖PE‖² = d/2, PE·PE зависит от k, `shift_matrix` — поворот, × √d), **B03** энкодер
  (8: == nn.TransformerEncoderLayer в Post-/Pre-LN с неединичными LN, нормированный выход Post-LN, стопка, PAD,
  двунаправленность), **B04** декодер (8: target_mask, == nn.TransformerDecoderLayer при T ≠ S, cross видит весь
  источник, causal), **B04d** (causal-маска в cross-attention), **B05** seq2seq (10: shift_right, лосс ==
  F.cross_entropy(label_smoothing) при ε 0/0.1/0.3, градиент p − q, forward, greedy на «сценарной» фальшивой модели,
  overfit «развернуть»), **B05d** (`torch.roll` вместо сдвига с BOS), **B06** проект «число → слова» (5: encode_pair,
  collate, train_step == эталон за 3 шага, правила exact match, запоминание 12 чисел), **B-P1**, **B-Q1**.
  Мутационная проверка: 61 мутант, все убиты, кроме одного эквивалентного (B06: сравнение «без EOS» с обеих сторон).
- B06 `main`: 2000 чисел из миллиона, 1500 шагов, 0.24 М параметров, ~22 с на CPU → 99.4 % exact match на 500
  отложенных; ошибки только на коротких числах (761 → «семь тысяч шестьсот одиннадцать»): 90 % случайных чисел
  шестизначные. `--record` пишет `content/viz/cross-attention.data.json`.
- Статьи: `seq2seq-sutskever`, `bahdanau-2014`, `label-smoothing` (Szegedy et al. §7), `label-smoothing-when`.
  Вопросы: `content/questions/encdec.yaml` (19), `papers-block-b.yaml` (5); user-db `iv-architecture-02/03`,
  `iv-attention-03/05`, `iv-positional-01/02`, `iv-generation-01` — через глобы task.yaml.
- Визуализации: `encdec-masks` (три маски, две пары батча, переключатель ошибки «маска по запросам» → nan),
  `sinusoidal` (тепловая карта PE на canvas, PE(p)·PE(p+k) от k для двух p), `cross-attention` (предзаписанный
  прогон B06: карта весов для 11 примеров, включая ошибки, и кривая exact match).
- README (таблица блока B), CHANGELOG → Unreleased.
- Проверено: 168 формул KaTeX без ошибок; 19 внешних ссылок отвечают 200; скриншоты трёх визуализаций (светлая
  1440, тёмная 860) и карты с блоком B — ошибок в консоли нет; `pytest tests/` 153 + 1 skip; `validate --all` — 100 задач.

**Фаза B2 (блок B, модуль 2 «BERT»):**
- `llmlab/core/bert.py` — эталоны B07–B11 (is_punctuation, clean_text, strip_accents, basic_tokenize, wordpiece, tokenize,
  truncate_pair, build_inputs, pad_batch, BertConfig, BertEmbeddings, mlm_mask, BertSelfAttention, BertLayer, BertModel,
  BertForMaskedLM, SequenceClassifier, align_labels, qa_loss, best_span); имена параметров как в HF — state_dict
  `transformers` грузится strict; `tests/test_core_bert.py`.
- Задачи: **B07** WordPiece (7 тестов: strip_accents с «й», basic_tokenize, жадность, [UNK] на слово, `##`, == `tokenizers` на
  корпусе, weights — словарь bert-base-uncased), **B08** вход BERT (6: обрезка == `tokenizers` на 1568 сочетаниях, типы
  сегментов, батч, эмбеддинги == transformers), **B09** MLM (5: labels, спецтокены, доли 15 % и 80/10/10 на ~60 тыс.
  токенов, детерминизм с generator, whole word), **B09d** (labels до исключения спецтокенов), **B10** модель (8:
  self-attention, слой с диагностом GELU-tanh/ReLU/Pre-LN, pooler, паддинг, MLM-голова, == transformers BertForMaskedLM,
  smoke «заполни [MASK]», weights: bert-base-uncased в своей модели → paris), **B11** головы (7: классификатор ==
  BertForSequenceClassification, обучение за 60 шагов, align_labels, qa_loss, best_span == перебор пар), **B11d**
  (независимые argmax), **B-P2** (BERT, RoBERTa), **B-Q2**. Мутационная проверка: 61 мутант, все убиты, кроме двух
  эквивалентных (B07 `split(' ')` после clean_text; B09 позиции word_ids = −1 и так спецтокены).
- Статьи: `bert`, `roberta`, `wordpiece-gnmt`, `gelu`, `squad`. Вопросы: `content/questions/bert.yaml` (15),
  `papers-block-b.yaml` (+5); user-db `iv-bert-01…05`, `iv-tokenization-03` — глобами (B-Q2 берёт все `iv-bert-*`).
- Визуализации: `wordpiece` (нормализация, слова, разбор по шагам на игрушечном словаре, cased/uncased), `mlm-masking`
  (вход и labels, whole word, доли на 200 масках, переключатель ошибки «спецтокены выбираются»); общий игрушечный
  токенизатор `web/src/viz/wordpieceToy.ts`.
- README (строка B2), CHANGELOG → Unreleased. Устаревший скриншот `docs/img/2026-10-17 17.05.42.jpg` (нумерация без буквы
  блока, нигде не используется) не закоммичен, перенесён в Корзину.
- Проверено: 44 формулы KaTeX без ошибок; 17 внешних ссылок — 200; weights-тесты B07 и B10 проходят (bert-base-uncased:
  «paris» 0.42, «coffee» 0.40); скриншоты двух визуализаций в обеих темах, ошибок в консоли нет; `pytest tests/` 159 + 1
  skip; `validate --all` — 109 задач.

**Фаза B3 (блок B, модуль 3 «DistilBERT и T5»):**
- `llmlab/core/t5.py` — эталоны B13–B15 (relative_position_bucket, relative_position_bias, T5Config, T5LayerNorm, T5Attention,
  T5Block, T5Stack, T5Model, random_segmentation, random_spans_noise_mask, noise_span_to_unique_sentinel, span_corruption);
  имена параметров как в HF — state_dict `transformers.T5Model` грузится strict; `tests/test_core_t5.py` (+ debug-копии).
- Задачи: **B12** дистилляция (9 тестов: soft targets, T²·KL == `F.kl_div(batchmean)` при T = 1/2/5, градиент = T·(q − p)/N и
  не идёт в учителя, масштаб при T = 8 и 2 одного порядка, смесь с CE и пустыми метками, cosine-loss без PAD, ученик из слоёв
  0, 2, … учителя (копии, tying), 60 шагов дистилляции: KL ↓ в 4+ раза, top-1 ≥ 80 %), **B13** относительный bias (11: формула на
  −300 … 300, ручные примеры, монотонность, == `T5Attention._relative_position_bucket` при трёх конфигах, Тёплиц, causal + offset,
  == `compute_bias`), **B13d** (сдвиг «запрос − ключ»: бакеты верны, bias нет; падает 4/11), **B14** слой T5 (9: RMSNorm с eps под
  корнем и fp16, attention без 1/√D + bias / переданный bias / нули / cross, блоки энкодера и декодера с диагностом Post-LN /
  GELU / без остатка, общий bias стека (шпион на блоке 1), PAD и causal, == `transformers.T5Model`; weights: t5-small в своей
  модели == `T5ForConditionalGeneration`, ответ с sentinel-ами), **B15** span corruption (10: разбиение (все 36 разбиений 10 на 3),
  маска на L = 10 … 568, случайность и равномерность, пример из рисунка 2 статьи T5, восстановление текста и длины на
  случайных масках, лимит sentinel-ов), **B15d** (prev_is_noise дополнен единицей: вход верный, цель съезжает; падает 2/10),
  **B-P3** (Hinton, DistilBERT, T5), **B-Q3**. Мутационная проверка: B12 — 16, B14 — 16, B15 — 12 мутантов; все убиты, кроме
  одного эквивалентного (B14: bias после маски: −inf + b = −inf). Найдено и закрыто: «eps после корня» (добавлен тест на
  |x| ~ 1e-3), «ученик делит модули с учителем» (снимок state_dict клонируется).
- Статьи: `hinton-distillation`, `distilbert`, `t5`, `shaw-relative-2018`. Вопросы: `content/questions/distil-t5.yaml` (16),
  `papers-block-b.yaml` (+6); user-db `iv-positional-01/02` → B13 и B-Q3. Ответы к статьям сверены с текстом PDF
  (абляция DistilBERT −3.69 / −2.96 / −1.46 / −0.31; фильтры C4; L + L слоёв энкодер-декодера = 2L параметров, L вычислений;
  раздел 3.6 T5; MNIST без троек у Хинтона).
- Визуализации: `t5-buckets` (кривая бакета от сдвига, оба режима, ползунки num_buckets / max_distance / T, матрица бакетов с
  диапазонами, неиспользуемый бакет 16), `distill-temperature` (мягкие метки на игрушечном примере, энтропия, масса вне top-1,
  T²·KL, ‖∇‖ от T с T² и без).
- README (строка B3), CHANGELOG → Unreleased.
- Проверено: 114 формул KaTeX без ошибок; ссылки на HF / GitHub / PyTorch отвечают 200; weights-тест B14 (t5-small, ~240 МБ)
  проходит; скриншоты страниц B12–B15, B-Q3 и обеих визуализаций (светлая 1440, тёмная 860) — ошибок в консоли нет (исправлены
  широкая формула бакетов и широкий пример t5-small); `pytest tests/` 165 + 1 skip; `validate --all` — 117 задач.

## Не сделано / долги
- A4: тесты A16d — копия тестов A16 без ветки диагноста про маску после softmax; при правке тестов A16 синхронизировать.
  `llmlab/core/minigpt.py` дублирует эталоны A15–A17 (страховка — `tests/test_core_minigpt.py`).
- A4: мини-GPT переобучается после ~1000 шагов (данных ~21 тыс. символов обучения); dropout в блоке нет. Числа в README
  A18, вопросах `minigpt-*` и `training-curve.data.json` зависят от корпуса и гиперпараметров `main` — при правке
  перезаписать: `python content/blocks/A/module-4-gpt/A18-minigpt/solution.py --steps 3000 --every 100 --record
  content/viz/training-curve.data.json`.
- A4: тест A18 `test_train_matches_reference` требует точного порядка шага и AdamW(weight_decay=0.01) + clip 1.0 —
  строго, но описано в docstring и README.
- A4: в блоке A одна голова attention; многоголовое — в C00 (сознательно, см. «Решения»).
- A3: тесты A12d/A13d — копии тестов A12/A13 без подсказок про баг; при правке тестов A12/A13 синхронизировать.
  `llmlab/core/bpe.py` дублирует эталоны A11–A13 (страховка — `tests/test_core_bpe.py`).
- A3: числа в README A14 (24 978 токенов GPT-2 на корпусе, 9 717 у BPE на 500 слияний; 28 против 152 токенов на
  английской/русской фразе от корпуса не зависят), симптомы A12d/A13d (≈240-й токен, «��аметки») и проверки сжатия в тестах A11/A12 зависят от `corpus-ru.txt` — при правке корпуса пересчитать.
- A3: weights-тесты A14 качают vocab.json/merges.txt/tokenizer.json GPT-2 (~2 МБ) через `hf_hub_download`; без сети с
  `--weights` падают (как C25), без флага — skip.
- A3: encode в A12–A14 наивный (пересчёт пар после каждого слияния, без кеша кусков) — на корпусе ~0.02 с, для больших
  текстов медленно; оптимизация описана в README A12 и вопросе `bpe-encode-speed-01`, но не реализована.
- A2: факты в README A07/A08 (β2 = 0.95, λ = 0.1, warmup 2000, cosine до 10%, clip 1.0 у Llama 2) — по статье Llama 2
  (раздел 2.2), не по config.json (в конфигах HF гиперпараметров обучения нет).
- A2: тесты A07d/A09d — копии тестов A07/A09 без подсказок про баг; при правке тестов A07/A09 синхронизировать.
- A10 зависит от корпуса: при правке `corpus-ru.txt` пересчитать число символов словаря (тест ждёт 67) и запас
  «MLP лучше биграммной на 10%» (сейчас ppl ≈ 10.8 против ≈ 14.4).
- ~~Блок A: README-скриншоты и GIF со старыми номерами~~ — пересняты в Фазе A4 (2026-10-18).
- A-P1: раздел о logsumexp из PLAN не стал отдельной статьёй — logsumexp разобран в A04/A04d и вопросах.
- Тесты A01d/A04d — копии тестов A01/A04 без подсказок про баг; при правке тестов A01/A04 синхронизировать.
- `llmlab/core/autograd.py` дублирует эталон A01 (тест сверяет исходник целиком, кроме шапки).
- Обучение в тестах A03 зависит от порядка `random.Random` в готовых конструкторах: менять конструкторы нельзя
  (запас: XOR 0.005 при пороге 0.05; кольца 100 % при 3 сидах).
- E2E-сценарий UI не в репозитории (Playwright ставился во временную папку). Если UI начнёт
  ломаться — завести `web/e2e/` с Playwright и отдельный CI-job.
- Отслеживание внешних правок — опросом (1.5 с), не SSE/watchdog: проще и без зависимостей.
- Подсветка кода в README (```python) — простая моноширинная, без раскраски.
- Задачи не блокируются: зависимости только рекомендуются («после: …», «следующая» на карте).
- `llmlab validate` гоняет задачи в 4 потока; при 40+ задачах может понадобиться кеш по хэшу файлов.
- Ссылка из конспекта на карточку — абсолютная `http://127.0.0.1:8765/#/interview/q/<id>`: при другом порте
  `llmlab serve --port` не сработает (Obsidian не знает порт).
- Конспекты только для задач с тестами; для paper/quiz — решить в Фазе 7 (квиз → конспект?).
- Граф связей в «Конспектах» глазами не смотрел (граф зависимостей на карте — проверен, 15 узлов).
- Конспектов для paper/quiz нет (решено в Фазе 7: их вопросы живут в Interview и в SRS).
- UI-рендер новых README в браузере глазами не смотрел; все 70 формул проходят `katex.renderToString` без ошибок (проверка
  скриптом, вне репо).
- ~~`ffn_dim_multiplier` / `multiple_of` Llama 3 8B — TODO-verify~~ — подтверждено по копиям `original/params.json` (ревью).
- Тесты debug-задач — копии тестов implement-задачи (с вырезанным диагностом). При правке тестов
  C01–C06 не забыть синхронизировать C01d–C06d.
- `llmlab/core/llama.py` дублирует эталоны C01–C05 (сверяется тестом) — при правке эталона править оба.
- Локальный `.venv` был скопирован из другого проекта (activate указывал на чужой venv) —
  пересоздан `rm -rf .venv && uv sync`. Если `llmlab` «не находится» — первым делом это.

- Фаза 4: тесты C07d/C09d/C10d — копии тестов C07/C09/C10 без подсказок, называющих баг. При правке тестов
  C07, C09, C10 синхронизировать debug-копии. `llmlab/core/gemma_olmo.py` дублирует эталоны C07–C11
  (страховка — `tests/test_core_gemma_olmo.py`).
- ~~Ячейки `gemma-3.ffn` / `olmo-2.ffn`~~ — закрыто в Фазе 9 (открывает C08).
- Разные θ RoPE для local/global слоёв Gemma 3 описаны в README C07, но не реализованы ни в одной задаче
  (в C07 один θ) — учесть в Фазе 8 (конфигурируемая модель).
- Тест градиента C08 опирается на статистику (3 сида, пороги 0.7–1.4 и > 2; фактически ≈ 1.0 и ≈ 2.9). При
  смене версии torch/инициализации пересчитать запас.

- Фаза 5a: тесты C12d/C13d/C14d — копии тестов C12/C13/C14 без подсказок про баг; при правке тестов
  синхронизировать. `llmlab/core/moe.py` дублирует эталоны C12–C15 (страховка — `tests/test_core_moe.py`).
- Тест C14 `test_bias_balancing_simulation` статистический (сид фиксирован; фактически max 0.145 / min 0.113
  при порогах 0.18 / 0.08) — при смене версии torch пересчитать запас.
- Фаза 6a: тесты C20d — копия тестов C20 без диагноста `_diagnose`; при правке тестов C20 синхронизировать.
  `llmlab/core/efficient.py` дублирует эталоны C18–C20 (страховка — `tests/test_core_efficient.py`).
- C18 без KV-cache индексатора и без инкрементального декодирования (только prefill/обучение); RoPE в индексаторе
  и MLA-основа DSA не реализованы — описаны в README. Обучение индексатора (KL к плотному attention) — только текст
  и вопрос, кода нет.
- C19 без KV-cache; RMSNorm Qwen3-Next вида (1 + w) не реализован (как у Gemma, C08) — учесть в Фазе 8.
- Фаза 6b: тесты C21d — копия тестов C21 без диагноста вариантов; при правке тестов C21 синхронизировать.
- C21 без conv1d, выходного гейта `RMSNormGated(o)·SiLU(z)` и 32 value / 16 key голов; чанковая (WY) форма не реализована —
  только описана. Для загрузки весов Qwen3-Next в Фазе 8 всё это понадобится.
- C22 без генерации с кешем (у `GatedAttention` нет KV-cache); память считается формулой, conv-состояние не учтено.
- `segsum` в C23 — неустойчивый вариант (разность cumsum); для длинных последовательностей нужен маскированный cumsum.
- Фаза 7: новые карточки в «Повторении» идут в порядке файлов (сначала вопросы курса); перемешивания нет. Лимит
  новых — на сессию, а не на сутки. E2E Interview по-прежнему вне репозитория.
- Вопросы user-db модулей 2–3 (Gemma/OLMo, MoE) в базе пользователя отсутствуют — квизы C-Q2/C-Q3 только из
  вопросов курса.
- Импорт `.xlsx` не поддержан (нужен openpyxl) — просим сохранить как csv.
- Визуальную проверку UI в 5a не делал (новых viz нет); README отдаются API, KaTeX проверен скриптом.
- Фаза 8: из долгов прошлых фаз закрыты нормы (1 + w), свой θ у local-слоёв Gemma 3, KV-cache у gated attention
  (общая `Attention`). **Не сделано:** загрузка весов Gemma 3 / OLMo 2 / DeepSeek / Qwen3-Next (`config_from_hf`
  знает только qwen3, llama, gpt2); эксперты MoE в 3D-тензорах HF (Mixtral, Qwen3-MoE); conv1d, выходной гейт и
  32:16 головы Gated DeltaNet; MTP и сжатие запроса в `deepseek-tiny`. Конфиги `*-like`/`*-tiny` проверяются
  только на случайных весах.
- `solution.py` C24/C25 дублируют `llmlab/core/hf.py` (страховка — `tests/test_core_hf.py`) — при правке менять оба.
- Weights-тесты C25 качают ~1.5 ГБ при первом запуске и без сети падают ошибкой `snapshot_download` (не skip) —
  сознательно: флаг `--weights` пользователь ставит сам.
- UI-страницу C24/C25 в браузере глазами не смотрел (README отдаются API как у всех задач).
- Скриншоты и GIF README сняты скриптом из scratchpad (Playwright вне репо, кадры → GIF через Pillow), последний раз — после правок UI 2026-10-17; при заметных изменениях UI
  переснять. Проверка установки на Linux/Windows — только в CI (локального Docker не было).

- B1: тесты B01d/B04d/B05d — копии тестов B01/B04/B05 без диагностов, называющих баг; при правке синхронизировать.
  `llmlab/core/encdec.py` дублирует эталоны B01–B05 (страховка — `tests/test_core_encdec.py`).
- B1: dropout в энкодер-декодере не реализован (детерминированные тесты) — описан в вопросе к статье; KV-cache
  декодера и кеш K/V cross-attention не реализованы (greedy пересчитывает префикс) — упомянуты в вопросе
  `encdec-seq2seq-encode-once-01`; beam search нет.
- B06: числа в README/вопросах/визуализации (99.4 %, 3 ошибки из 500, 22 с) зависят от сида и версии torch; при
  правке перезаписать: `python content/blocks/B/module-1-encdec/B06-num2words/solution.py --record
  content/viz/cross-attention.data.json`.

- B2: тесты B09d/B11d — копии тестов B09/B11 без диагностов, называющих баг; при правке синхронизировать.
  `llmlab/core/bert.py` дублирует эталоны B07–B11 (страховка — `tests/test_core_bert.py`).
- B2: CJK-иероглифы (BERT окружает их пробелами) в basic_tokenize не обрабатываются — оговорено в solution.py; NSP-голова
  и предобучение BERT целиком не реализованы (только MLM-маскирование и smoke-обучение); голова token classification
  описана, но отдельного класса нет (только align_labels); SQuAD 2.0 («нет ответа») — только в вопросе.
- B2: тесты MLM статистические (пороги 4–6σ на фиксированных сидах; whole word ±2 % — ≈3.4σ).
- B2: weights-тест B10 качает ~440 МБ (bert-base-uncased) и без сети с `--weights` падает (как C25).

- B3: тесты B13d/B15d — копии тестов B13/B15 с нейтральными сообщениями; при правке синхронизировать. `llmlab/core/t5.py`
  дублирует эталоны B13–B15 (страховка — `tests/test_core_t5.py`, сверяет и debug-копии `solution.py`).
- B3: у B12 нет core-копии (дальше дистилляция не используется). Ученик берёт слои 0, 2, 4, …, а скрипт HF — 0, 2, 4, 7, 9, 11
  (написано в README); без `token_type_embeddings` и pooler ученик не собирается — используется тот же `BertForMaskedLM`.
- B3: T5 без KV-cache, без dropout, без `T5ForConditionalGeneration` (голова с d_model^−½ — только в weights-тесте и README),
  без GeGLU v1.1; span corruption — для одного текста (батч и подбор длины 568 → 512 только в README / вопросе).
- B3: weights-тест B14 качает t5-small (~240 МБ) и токенизатор через `AutoTokenizer`; без сети с `--weights` падает (как C25).
- B3: тесты B15 статистические на фиксированных сидах (все 36 разбиений за 1500 попыток; средняя длина первого куска ±0.25;
  доля шума на позициях 10–89 в 0.05 … 0.3 за 300 масок).
- B3: в базе вопросов пользователя нет вопросов про дистилляцию и T5 — в B-Q3 только `iv-positional-01/02`.

## Решения (почему сделано так, а не иначе)
- **Фаза 8:** одна общая `Attention` в `core/model.py` из функций эталонов (rope, partial rope,
  repeat_kv, KVCache, RingKVCache) вместо обёрток над 6 классами — у Qwen3 head_dim ≠ d/H, нужны сочетания идей; её
  частные случаи сверяются с LlamaAttention/QKNorm/SWA/NoPE/GatedAttention/блоками C08 в `tests/test_core_model.py`.
  MLA, Gated DeltaNet, MoE, FFN — классы эталонов. Имена параметров = HF без `model.`. Задача-проект разбита на
  C24 (config.json + имена весов, вкл. GPT-2 Conv1D) и C25 (safetensors-шарды, meta-загрузка, сэмплирование,
  генерация, сверка с transformers). Бонус — GPT-2 и SmolLM2-135M одной `load_pretrained`. Поддержан формат
  config.json transformers 5.x (`rope_parameters`). Допуск логитов 1e-3 в fp32 (факт 6e-5 на Qwen3-0.6B).
  Новая строка таблицы архитектур `qwen3` (открывает C25). Weights-тесты: флаг `--weights`/`LLMLAB_WEIGHTS=1`.
  **Дополнительные разделы summary.md идут в конспект** (после четырёх обязательных), а не отдельный файл «Моя
  модель»: итоговый конспект = конспект C25 с картой «деталь → задача» и wiki-ссылками на все задачи. Генерация в
  CLI — без `transformers` (только `tokenizers`), transformers нужен лишь для сверки в тестах. Чат-шаблон Qwen3 в
  `--chat` — строкой, с пустым `<think>` (режим без размышлений).

- **Реальные веса второй архитектуры (Gemma 3 / OLMo 2) — не в Фазе 9** (ответ пользователя после Фазы 8): оставлено
  блоку «Позже»; sandwich-нормы и SWA проверяются только на случайных весах.

- **Фаза 9: `doctor` гоняет эталон C00 по умолчанию** (~2 с): это единственная проверка, которая ловит сломанный
  subprocess/pytest/torch целиком; `--quick` — без него. Нет пакетов `weights` и занятый порт — предупреждения, не ошибки.
- **Linux/Windows проверены через CI**, а не локально: job `install` повторяет быстрый старт из README на трёх ОС.
- **Ячейки FFN Gemma 3 / OLMo 2 открывает C08** (а не новая задача): FFN в модуле 2 не меняется, GeGLU — одна
  строка отличия от SwiGLU; объяснение — абзац в README C08.
- **Версия 0.1.0 и тег `v0.1.0`** (PLAN: «тег v0.1») — semver с тремя числами, как в pyproject.
- **Блок A (план, 2026-10-17):** необязательный, идёт перед C; autograd на чистом Python-скалярах, дальше
  `torch.Tensor` без `nn`/`optim`; номера навигации получают букву блока («A1.2», «C1.3»), потому что «1.3» в двух
  блоках неоднозначен; межблочные зависимости — только рекомендации; корпус для обучения пишем сами (≤ 50 КБ);
  после блока — v0.2.0.
- **Фаза A1: номер модуля — из id** (`module-1-autograd` → 1), а не позиция: у блока C модули с `module-0`, у A — с
  `module-1`; так номера совпадают с названиями папок. Порядок блоков — алфавитный по папкам (поле `order` из плана
  не понадобилось: A < B < C < D).
- **Фаза A1: `Value` хранит локальные производные** (`_children`, `_local_grads`), а не замыкание `_backward`, как в
  micrograd: backward — один общий цикл chain rule, ошибка «= вместо +=» живёт в одном месте (A01d), и код заметно
  отличается от micrograd (мы на него только ссылаемся).
- **Фаза A1: A02/A03 импортируют готовый `Value` из `llmlab.core.autograd`**, а не код пользователя из A01 — как
  `core/llama.py` в блоке C.
- **Фаза A1: в A04/A05 autograd и готовые softmax/CE/LayerNorm запрещены через monkeypatch**, оракул вызывает
  сохранённые до запрета функции.
- **Фаза A1: визуализация у каждой implement-задачи** (как в блоке C после правок UI), а не только V-autograd-graph из
  PLAN: добавлены gradcheck, mlp-boundary, softmax-ce, layernorm-grad.
- **Фаза A2: A-P2 разделена на A-P2 (Adam, AdamW) и A-P2b (Glorot, He, Bengio)** — пять статей в одной задаче не
  укладываются в 60 минут (правило CLAUDE.md), как C-P2/C-P2b.
- **Фаза A2: в A09 можно `torch.optim` и `clip_grad_norm_`** — свои версии уже написаны в A07/A08; задача про порядок
  шага, и тест сверяет историю лоссов с эталоном точно.
- **Фаза A2: корпус свой и маленький (~8 тыс. символов)**: хватает, чтобы MLP с окном 4 обогнал биграммы за 0.2 с;
  больше текста — медленнее тесты. В модуле 4 (свой GPT) при нехватке данных — дописать корпус.
- **Фаза A2: у A09 нет своей визуализации** — используются `mlp-boundary` (тот же цикл обучения) и `lr-schedule`.
- **Фаза A3: общий `llmlab/core/bpe.py`, A12–A14 импортируют из него** готовые части (как `core/autograd.py`); сверка —
  по исходнику каждой функции через `ast`, а не целиком файла: в core лежат эталоны трёх задач.
- **Фаза A3: `count_pairs`/`train_bpe` принимают список последовательностей с первой задачи** — в A11 это один кусок
  `[list(text.encode())]`, а в A13 обучение с регуляркой сводится к одной строке (`train_regex_bpe`).
- **Фаза A3: ничья при выборе пары → меньшая пара (сравнение кортежей)**, а не «первая встреченная», как в minbpe:
  результат не зависит от порядка обхода словаря и проверяется тестом.
- **Фаза A3: decode через байты требуется уже в A12** (тест «пробел + половина буквы»), а A13d повторяет ту же ошибку в
  decode со спецтокенами — на корпусе её видно по «��» в тексте. Так урок закрепляется дважды, а A13d остаётся «найди баг».
- **Фаза A3: A14 без сети проверяется на игрушечных файлах GPT-2-формата** (свой BPE, id в vocab.json перемешаны), чтобы
  инвариант «starter падает на всех тестах» и обучение работали офлайн; настоящие файлы — только с `--weights`,
  только `tokenizers` (без transformers). В отличие от `core/hf.download` качаем три файла, а не snapshot с весами.
- **Фаза A3: `regex` — основная зависимость**, а не extra: регулярка GPT-2 нужна задачам без весов, а `re` не знает
  `\p{L}`/`\p{N}`. Пакет маленький и с колёсами под все ОС.
- **Фаза A3: в A13 пользователь пишет регулярку сам по таблице правил** (полный шаблон — в третьей подсказке): так
  понятен смысл каждой альтернативы; тесты на 9 примерах ловят типичные отступления (пробел в конце слова, «3.14» одним
  куском, `  b`).
- **Фаза A4: в блоке A attention одноголовое** (голова размера n_embd) — PLAN: «дальше — multi-head в C00». Блок A
  отличается от GPT-2 ровно числом голов, словарём и размерами; C00 остаётся первой задачей обязательного блока.
- **Фаза A4: C00 связана с A16/A18 через новое поле `recommended_after`, а не `depends_on`**: с `depends_on` C00 никогда
  не стала бы «следующей» для тех, кто пропускает необязательный блок A (кандидаты блока A исключаются, а зависимость не
  решена).
- **Фаза A4: у A15 и A17 нет новых визуализаций** — `bigram-table` (EmbeddingLM — по сути биграммная модель) и
  `norm-placement` (pre/post-LN из C08) показывают ровно их механизм; новая — только `training-curve`.
- **Фаза A4: модель A18 уменьшена до 190 тыс. параметров (n_embd 64, lr 1e-3)** после сравнения 5 конфигураций: все
  переобучаются после 750–1000 шагов, а лучший val почти одинаков (2.19–2.27); ранняя остановка в `main` вместо
  dropout — без изменения блока A17 и с наглядной кривой переобучения в визуализации.
- **Фаза A4: данные V-training-curve лежат в `content/viz/training-curve.data.json`** и импортируются в бандл при сборке
  (JSON рядом с yaml-описанием, как требует PLAN), а не запрашиваются через API.
- **Фаза A4: итоговый квиз блока — отдельная карточка A-Q4b** (22 вопроса списком id), а не глобы всего блока
  (~100 вопросов); там же таблица «что ты построил → где это в блоке C».
- **v0.2.0 — минорная версия**: новый блок и новое поле task.yaml; прогресс v0.1.x совместим.
- **Правка к A3: корпус дописан до 23 тыс. символов** (≤ 50 КБ по PLAN) по согласию пользователя; алфавит вырос на 6
  символов (?, —, Ж, Р, Ц, Я).
- **v0.1.1 — патч, а не 0.2.0**: новых блоков нет, id задач не менялись, прогресс v0.1.0 совместим. 0.2.0 — с блоком A.

- **Исправления ревью:** C25 не переименована (сэмплирование — новая `C25a`, а не перенумерация C25 → C26), и
  paper-задачи разделены с сохранением старых id — чтобы не сломать прогресс, конспекты и отметки «пройдено» у тех,
  кто уже начал курс. Paper-задачи раздроблены по правилу CLAUDE.md, а не выведены из-под него.
- **Защита от импорта эталона — по AST эталона, а не по списку `entrypoint`**: проверяются все `def`/`class`
  верхнего уровня `solution.py`, кроме импортируемых заготовкой; так покрыты и вспомогательные функции задачи.
- **Ревью без правок** (по промпту «Ревью»): находки только записаны, исправления — отдельной сессией.
  Мутационный скрипт в репозиторий не добавлен — разовый инструмент; если ревью станет регулярным, завести
  `tools/mutate.py`.

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

- **Фаза 2, конспекты:** файл — источник правды, БД — кеш + страховка. «Главное, что я понял»
  при пересборке берётся из файла (правка в Obsidian побеждает), иначе из БД. Записывается
  списком `- строка`, чтобы строки не склеивались в абзац в react-markdown/строгом Markdown.
- **Конспект создаётся только после ввода вывода** (модалку можно закрыть «Позже» — тогда в
  шапке задачи кнопка «Создать конспект»). Повторный зелёный прогон молча обновляет конспект
  (код, ошибки), вывод не трогается.
- **«Мой код»** = рабочая копия, если её sha совпадает с какой-то зелёной попыткой; иначе
  снимок кода, сохранённый в БД при последней сборке. Так сломанный после решения код не
  попадёт в конспект.
- **«Мои ошибки по пути»** группируются по тексту сообщения (числа нормализуются), а не по
  тесту: при ошибке импорта 12 тестов падают с одним текстом → одна запись. Под записью —
  названия тестов, число попыток и строка кода пользователя, если известна. Сообщения
  `NotImplementedError` выброшены (это не ошибка, а нетронутая заготовка). Берутся попытки до
  последней зелёной.
- **`attempts` во frontmatter** = номер попытки, с которой задача впервые решена (как в
  примере PLAN.md), а не общее число прогонов.
- **Diff и вопросы — свёрнутые callout-ы Obsidian** (`> [!example]-`, `> [!question]-`), а не
  HTML `<details>`: работают в Obsidian нативно, а UI рендерит их сам (без rehype-raw).
- **Таблица архитектур:** строки/столбцы — `content/architectures.yaml`, значения — поле
  `arch` в task.yaml задачи, объясняющей компонент («данные из task.yaml», как в промпте).
  Ячейка открыта, если задача решена (не обязательно с конспектом). Закрытые ячейки API не
  раскрывает. Модели: GPT-2, Llama 3 8B, Gemma 3, OLMo 2, DeepSeek-V3, Qwen3-Next — значения
  появятся вместе с задачами модулей.
- **Поиск:** FTS5 `unicode61 remove_diacritics 2`, каждое слово — префикс (`"слово"*`): без
  стемминга русские словоформы ищутся по началу слова. Индекс синхронизируется с файлами по
  mtime перед каждым поиском — правки из Obsidian находятся без пересборки.
- **`notes/index.md`** — оглавление vault'а, не считается конспектом (не в списке/поиске).
- **Маршруты UI:** `#/notes`, `#/notes/<id>`, `#/notes/~graph`, `#/notes/~arch` (тильда —
  чтобы не пересекаться с id задач).

- **Фаза 3a, контент:** Llama-гиперпараметры сверены по `NousResearch/Meta-Llama-3-8B` и
  `NousResearch/Llama-2-7b-hf` — открытые зеркала gated-репозиториев Meta (config.json те же).
  В комментарии task.yaml указаны оба.
- **RoPE — раскладка «половинами» (HF/GPT-NeoX)**, не интерливная (Meta/RoFormer): так проще
  сверять с `transformers` и грузить HF-веса в Фазе 8. В README объяснено, что это одно и то же
  с точностью до перестановки строк W_Q/W_K. `rope_cos_sin` принимает тензор позиций (не длину),
  чтобы KV-cache в C05 переиспользовал его со смещением.
- **RMSNorm как в HF Llama:** нормировка в fp32, `.to(dtype)` ДО умножения на weight.
- **C02 включает `llama_ffn_hidden_dim`** — правило 8/3·d с множителем и округлением; это
  «что поменялось» в цифрах и проверяется на реальных конфигах.
- **Debug-задачи:** `diff_from` и `depends_on` = implement-задача; `arch` не задают (ячейку
  открывает implement-задача); тесты — копия, где диагност и подсказки, называющие именно этот
  баг, заменены на описание симптома. C02d (SiLU на ветке up) — свой баг, в PLAN он не назван
  (там 5 багов на 6 debug-задач); остальные: C01d — eps, C03d — половины в RoPE.
- **Вопросы модуля** пишутся сразу в 3a (по 2–3 на задачу), чтобы конспекты имели «Вопросы для
  повторения»; квиз C-Q1 в 3b дополнит файл до 8–10+ и соберёт по тегам.
- **Готовые функции запрещены через monkeypatch:** `F.rms_norm`/`nn.RMSNorm` (C01), `F.silu`/`nn.SiLU`
  (C02). В C03 запрещать нечего.

- **Фаза 3b: поздние задачи импортируют эталоны из `llmlab/core/llama.py`**, а не копию в starter и
  не код пользователя: ошибка в C03 не ломает C05/C06, а Фаза 8 соберёт модель из того же модуля.
  Цена — дублирование с `solution.py`, страховка — `tests/test_core_llama.py`.
- **C04 без RoPE**, RoPE + кеш — в C05: каждая задача про одну идею. Конструкторы в C04/C05 даны
  (имена HF), в C06 конструкторы пишет пользователь — сборка и есть суть задачи; имена проверяются
  загрузкой `state_dict` эталона с понятной ошибкой.
- **C05: в кеше k после RoPE и до repeat_kv** — это проверяется тестом (как в HF). `F.sdpa` в C05
  разрешён: его `is_causal=True` при T < S — отдельный урок, диагност его называет.
- **Debug-баги модуля 1 отличаются от списка PLAN**: C04d — `.repeat` вместо `repeat_interleave`
  (GQA-специфичный баг, а «пропущен 1/√d» уже ловит диагност C00/C04), C06d — остаток от norm(x)
  (маски в коде C06 нет — она в `core`). «Утечка через маску» будет в Фазе 4 (off-by-one в SWA).
- **Paper/quiz без отдельного UI**: вопросы в `content/questions/*.yaml`, показываются вкладкой
  «Вопросы» (mcq с проверкой, open с эталоном). Полноценный Interview/SRS — Фаза 7.
- **mcq: поле `correct`** (индекс в `options`), загрузчик требует его для mcq и запрещает для других.
- **Порядок в `topo_order`**: paper/quiz в конце, иначе «C-P1» (дефис < цифры) шёл перед «C01».
- **V-kvcache без весов модели для сравнения**: размеры весов проверены только для Llama 3 8B
  (8 030 261 248, посчитано по config.json), остальные не выдумывал. Mistral не в пресетах —
  у него SWA, добавить в Фазе 4. Llama 3.2 1B сверен по зеркалу `unsloth/Llama-3.2-1B`.

- **Фаза 4: гиперпараметры Gemma** сверены по открытым зеркалам `unsloth/gemma-2-9b`,
  `unsloth/gemma-3-{1b,4b,27b}-pt` (оригинальные репозитории Google закрыты соглашением); SmolLM3, OLMo 2,
  Mistral — по оригинальным config.json. Правило «слой i full ⇔ (i+1) % pattern == 0» и семантика окна
  (`kv_idx > q_idx − W`, окно включает сам токен) — из `configuration_gemma3.py` / `masking_utils.py`.
- **C08: эффект градиентов проверяется при Xavier-инициализации**, а не при дефолтной `nn.Linear`: с
  дефолтной (выходы подслоёв малы) разница Pre/Post почти не видна, с Xavier — ровно картина Xiong et al.
  (Post ≈ константа, Pre ~ 1/√L; проверено экспериментом на 3 сидах). Конструкторы блоков даны, пользователь
  пишет только `forward` (по две строки) — суть задачи в сравнении, а не в сборке.
- **C08: у Gemma `RMSNorm` умножает на (1 + w)** — в задаче обычный RMSNorm, отличие описано в README.
  Учесть в Фазе 8 при загрузке весов Gemma.
- **C09: метод `scores()`** вынесен в интерфейс, чтобы проверить ограниченность логитов напрямую.
  Режим OLMo 2 (`full`) тоже в задаче: разница «по голове / по всей проекции» — частый вопрос.
- **C10: расписание NoPE «последний в группе»** проверено по `no_rope_layers` SmolLM3-3B. Debug
  «global-слои по неверному индексу» из PLAN сделан на расписании NoPE (`C10d`), а не Gemma: у C07 уже
  есть debug (окно), а формула расписания та же; ловушка Gemma описана в C07 (диагност `i % pattern`).
- **C11: `update` возвращает буфер ДО записи + новые токены** (`[W + T]` ключей, пустые с позицией −1), а
  не «последние W»: иначе кусок из нескольких токенов теряет ключи, вытесненные собственной записью.
  Тест `test_update_returns_needed_keys` проверяет семантику (нужные позиции есть, позиции ↔ ключи), а не
  точный формат, — допускаются и другие корректные реализации.
- **C11 без debug-задачи**: три бага PLAN распределены по C07d/C09d/C10d.
- **V-masks: три режима в одном компоненте** (маска / расписание / поле зрения), а не три viz: одна
  тема, одна вкладка. Полоса слоёв — сетка по 12 (кратно 6, 4, 2), чтобы паттерн выстраивался столбцами.
- **Таблица архитектур: добавлена строка SmolLM3** (модуль назван в честь трёх моделей, NoPE — её фича).

- **Фаза 5a: эксперты — `SwiGLUFFN` в `nn.ModuleList`** (имена `experts.{i}.gate_proj` как у DeepSeek в
  HF), а не 3D-параметры `gate_up_proj` нового Mixtral в HF: так понятнее и проще диспетчеризация. Для
  загрузки весов Mixtral в Фазе 8 понадобится конвертация.
- **C12 включает sigmoid-роутер и `routed_scaling_factor`** (DeepSeek-V3), но не групповой выбор
  (`n_group`/`topk_group`) — он нужен для распределения по устройствам, описан в README.
- **C14: aux loss нормирован так, что минимум = 1** (f_i делится на N·k, как в статье Switch для k = 1);
  HF Mixtral делит на N (минимум = k) — тест сверяет `HF / k` и объясняет разницу.
- **Capacity factor / drop токенов не реализованы**: современные открытые MoE (Mixtral, DeepSeek-V3,
  Qwen3 в HF) — dropless. Debug «токены теряются» из PLAN сделан как C13d (вклады экспертов
  перезаписываются → токен теряет эксперта). Capacity упомянуть в C-P3 (Switch).
- **Debug «нет перенормировки top-k» из PLAN** не отдельная задача: его ловит диагност C12; вместо него
  C14d (знак шага bias) — баг, специфичный для DeepSeek-V3.
- **C15 считает только FFN-часть** параметров (+ роутер): attention (MLA) и эмбеддинги — в 5b, поэтому
  671B / 37B в тестах не проверяется, только упоминается.
- **C15 открывает «moe: нет — dense»** у Gemma 3 / OLMo 2 / SmolLM3 — решение долга Фазы 4 без новой
  задачи (ячейку открывает задача, которая объясняет MoE vs dense).

- **Фаза 5b: MLA разбит на две задачи** — `C16-mla` (латент, кеш, поглощение, без позиций) и `C16b-mla-rope`
  (decoupled RoPE): вместе это > 60 минут, а PLAN называет их «затем». Id `C16b` (id C18+ заняты Фазой 6).
- **MLA без сжатия запроса** (как DeepSeek-V2-Lite, `q_lora_rank = null`); `q_lora_rank` V3 описан в README
  и в вопросе к статье. YaRN и `mscale` не реализованы — описаны.
- **MTP: порядок конкатенации [emb; h]** — по vLLM `deepseek_mtp.py` (в формуле статьи наоборот); λ = 0.3 → 0.1
  и 85–90 % / ×1.8 для спекулятивного декодирования — сверено по тексту отчёта DeepSeek-V3 (arXiv HTML).
  MTP-блок — Pre-Norm блок из `core/gemma_olmo.py` (C08), не MoE-блок V3.
- **Debug 5b — только C16d** (баг PLAN «в MLA кешируется не латент»); MTP без debug-задачи.
- **V-moe-routing обучает только смещение логитов каждого эксперта** (вектор [E]), а не матрицу роутера:
  наглядно и считается мгновенно; оговорено в подписи. Bias-режим выбирает по log p + bias (= логиты + bias).
- **V-kvcache для MLA**: «эквивалентное» число KV-голов (576 / 320), чтобы не менять общую формулу; MQA
  для этого пресета не показан (подписи наезжали, а главное сравнение — с MHA тех же голов).

- **Фаза 6a: DSA без MLA и RoPE** — основное attention в C18 обычное MHA, индексатор без RoPE/LayerNorm/Адамара/FP8:
  суть задачи — индексатор и выбор top-k; формула и масштабы (`n_heads ** -0.5`, `softmax_scale`) — по
  `inference/model.py` DeepSeek-V3.2-Exp (`Indexer`, `fp8_index`: ReLU(q·k) · вес головы, сумма по головам).
  Схема обучения индексатора (прогрев: только индексатор, KL к сумме весов голов; затем detach) сверена по
  тексту arXiv 2512.02556.
- **DSA в таблице — отдельная строка `deepseek-v3.2`**: ячейку `deepseek-v3.attention` уже открывает C16, а DSA —
  это другая модель (V3.2-Exp).
- **C18 зависит от C07** (diff_from — «фиксированное окно → выбор по содержанию»), а не от C16.
- **C19: QK-Norm и GQA оставлены** (как в Qwen3-Next), diff_from = C09; раскладка гейта по головам — главный
  подвох задачи, на него два теста (оракул с диагностом и «обнулённые строки гейта → ½»).
- **C20 с некаузальной формой** — четвёртая функция, чтобы ассоциативность (kernel trick) была реализована явно, а
  не только упомянута. Чанковая форма — в C21 (DeltaNet).
- **Debug 6a — только C20d** (нормировка); «утечка будущего в рекуррентной форме» из PLAN — в 6b (C21d).
- **V-masks: оценки индексатора игрушечные** (не обученная модель) — оговорено в подписи.

- **Фаза 6b: C21 одной задачей** (DeltaNet → Gated DeltaNet через `g=None`), а не двумя: разница — одна строка, а слой
  (`gates`, проекции) добавляет то, что связывает формулу с весами HF. Время 55 мин — на верхней границе.
- **Debug «утечка будущего в рекуррентной форме» — C21d** (чтение итогового состояния после цикла): правдоподобная
  «векторизация», не ловится генерацией по одному токену.
- **C22 diff_from = C11** (Gemma 3: SWA + кольцевой кеш): сравнение «дешёвый слой с окном / с состоянием».
- **Новый столбец таблицы `layers`**: чередование слоёв — отдельный компонент архитектуры; ячейку `attention` у Qwen3-Next
  уже открывает C19. Ячейки `layers` у DeepSeek/OLMo/SmolLM3/GPT-2 пустые (ни одна задача их не объясняет).
- **C23 сделана** (PLAN: опционально): двойственность скан ↔ attention замыкает модуль и объясняет, откуда гейт в Gated DeltaNet.
  Конфиг Mamba-2 2.7B: в `config.json` нет `ssm_cfg` кроме типа слоя — d_state/headdim/expand взяты из значений по
  умолчанию класса `Mamba2` (`mamba_ssm/modules/mamba2.py`), это оговорено в task.yaml и README.
- **V-hybrid в геометрии Qwen3-Next для всех трёх расписаний** (а не реальные Gemma/Llama): сравнение честное только при
  одинаковых головах; реальные модели показаны полосой слоёв.

- **Фаза 7: разметку делал агент в сессии, без вызова LLM из кода**: импортёр детерминированный, а «размечено» =
  `source_ref` в `content/questions` — пачки можно продолжать в новой сессии (`llmlab questions batch`).
- **Смысловые дубли объединены**: 244 сырых → 105 вопросов (один вопрос курса покрывает 1–8 сырых). Вопросы о
  конкретных проектах (Confluence, чат на 600 человек, RAGAS 0,91) обобщены, детали проекта в формулировки не
  попали.
- **topic: other + tags** для всего вне архитектуры/инференса LLM (RAG, агенты, evals, retrieval, BERT, word2vec,
  LoRA, классический ML) — по промпту фазы; LoRA/дообучение тоже `other` (в курсе нет задачи), а токенизация,
  генерация/сэмплирование и квантизация — темы курса без задач.
- **source_ref — хэши (sha1[:10]) нормализованного сырого текста** в публичном репо: текста не раскрывают, но
  короткий вопрос теоретически подбирается перебором формулировок; сочтено приемлемым (вопросы общие).
- **SM-2 с четырьмя кнопками и повтором через 10 минут** вместо дня при провале (как в Anki, удобнее для
  подготовки к собесу); оценки mcq не ставятся автоматически — подсвечивается рекомендованная.
- **Состояние Interview — в SQLite на сервере**, во фронтенде (localStorage) только настройки фильтров.
- **paper/quiz «пройдено» — попытка `impl = 'manual'`**, а не отдельная таблица: статусы на карте и в CLI
  считаются тем же запросом.

- **Номера «модуль.шаг» считаются, а не хранятся** (`Course.labels()`): новая задача сдвигает номера следующих —
  зато id и пути (workspace, конспекты, ссылки [[C03-rope]]) не меняются. Отладка получает номер родителя.
- **`short_title` — необязательное поле task.yaml** (заполнено у всех 54): длинные названия не помещаются в узел графа
  и в хлебные крошки; без него — title без префикса «Отладка: / Статья: / Квиз модуля: ».
- **Схемы форм — свой мини-формат ` ```shapes `, а не Mermaid/SVG-файлы**: без зависимостей и CDN, текст читается и
  в GitHub/Obsidian, а в UI оси окрашены по смыслу и подсвечиваются. В `summary.md` схемы остались текстовыми.
- **Упоминания задач в README пишутся кодом (`C05`) и превращаются в ссылки на лету** (remark-плагин, только вне
  кода): тексты не пришлось переписывать, при перенумерации ссылки не устаревают.
- **Визуализации в debug-задачах — только во вкладке**, не в тексте; для C04d — `kv-cache`, а не `gqa-groups`
  (подпись про repeat_kv подсказала бы баг).
- **Новые визуализации — упрощённые модели с явной оговоркой в подписи** (рост RMS остатка — независимые вклады;
  память состояния — без φ и проекций). Числа реальных моделей — только из config.json (Qwen3-0.6B, Llama 2/3).

- **В .py-файлах задач (starter/buggy/solution/test_task) задачи названы по `short_title` в кавычках** («RoPE»),
  а не номером «1.3»: Python-код не проходит через UI, а номера вычисляемые и могут сдвинуться. В Markdown (README,
  подсказки, вопросы, конспекты) по-прежнему пишем код `C03` — UI превращает его в ссылку.
- **Синтаксис ` ```shapes ` проверяет загрузчик** (`check_shapes`): ошибка — issue уровня error на README задачи,
  её показывает `llmlab validate`. Грамматика продублирована в Python и в ShapeFlow.tsx — менять обе.

- **Блок B: необязательный, между A и C.** Не нужен для C; B01 рекомендует A16, задачи блока B импортируют
  только эталоны блока B (`core/encdec.py`). Для BERT/T5/ModernBERT план сверки с `transformers` на случайных весах.
- **B01d — нарезка голов, а не «маска по запросам»** (как было в первом варианте плана): ошибка оси маски ловится
  первым же тестом формы и очевидна, а `view(B, H, L, D)` правдоподобна, не видна при H = 1 и учит проверять
  эквивариантность. «Маска по запросам» осталась в диагносте B01 и в переключателе V-encdec-masks. PLAN исправлен.
- **Маски — bool, True = «можно смотреть»** (как в A16, C00); различие с `key_padding_mask` PyTorch вынесено в
  README B01, диагност и вопрос.
- **Label smoothing — соглашение PyTorch (ε/V всем классам)**, а не fairseq (ε/(V−1) неверным); диагност узнаёт
  вариант fairseq.
- **Инициализация эмбеддинга std = 1/√d** (как в fairseq; статья не указывает) и паддинг-строка нулевая: иначе
  множитель √d из §3.4 бессмыслен при общей с выходным слоем матрице.
- **B06 — «число → слова», а не перевод текста:** обучается за ~25 с на CPU, есть разная длина, выравнивание,
  «молчащие» нули и согласование («тысяча / тысячи / тысяч»), данные генерируются кодом (ничего не скачиваем).
  Перекос по длине чисел оставлен намеренно: учит сдвигу распределения (разобрано в README и вопросе).
- **B06: пользователь пишет `train_step`, а не `train`** — цикл с валидацией в `main` готов; тест `train_step` ==
  эталон за 3 шага (лосс и веса) ловит порядок clip/step.

- **B08: обрезка пары — правило быстрых токенизаторов HF**, а не цикл «по одному у длинного» из исходного BERT: при равных
  длинах они расходятся; диагност узнаёт цикл и объясняет. Проверено против `tokenizers` на 1568 сочетаниях.
- **B10: модель с именами HF и вложенными `nn.Module()`-контейнерами** (`attention.self`, `attention.output`) вместо
  своих имён: настоящие веса грузятся strict без маппинга, а `attention` из B01 переиспользован. MLM-bias — один
  Parameter под двумя именами, как в HF.
- **B09d — порядок строк (labels до исключения спецтокенов)**, а не «спецтокены не исключены вовсе»: вход остаётся
  правильным, и баг виден только в labels — так он и встречается в реальном коде.
- **Тесты с `transformers` / `tokenizers` — через `importorskip`** (как C24): без extras остаются тесты против эталона и
  ручные примеры; в CI с weights-extras они идут.

- **B3: эталоны модуля — в `core/t5.py`, а B12 без core-копии**: B14 импортирует бакеты B13, а дистилляцию дальше никто не
  импортирует; лишняя копия только добавила бы синхронизацию.
- **B12: `distill_kl` сам делает `detach` учителя** и принимает логиты любой формы `[..., V]`; мягкая часть — по всем
  позициям, жёсткая — только по labels ≠ −100 (как в DistilBERT: KL по attention_mask, MLM по закрытым). Без меток в батче
  hard = 0, а не nan. Учитель в тестах — случайный BERT из `core/bert.py`, ничего не скачиваем.
- **B13: две функции — бакеты и bias**, а не класс: B14 держит таблицу `relative_attention_bias` внутри `T5Attention` с именем
  HF, а B13 даёт ей функцию `relative_position_bias(embedding, …, offset)`. `offset` — заготовка под генерацию с кешем.
- **B13d — знак сдвига в `relative_position_bias`**, а не в формуле бакетов: бакеты остаются верными, ломается только
  связка, как в реальном коде; в causal-режиме ошибка особенно заметна (все прошлые ключи — бакет 0).
- **B14: пользователь пишет forward четырёх классов** (норма, attention, блок, стек), `T5Model` дан. Подслои — `nn.Module()`-
  контейнеры с именами HF (`layer[0].SelfAttention`, `DenseReluDense.wi`), как в B10. Только ReLU-FFN (t5-small, исходный T5);
  GeGLU v1.1 — в README. Сравнение с `transformers` — только по настоящим позициям (PAD-строки не сравниваем).
- **B15: алгоритм маски — как в T5** (ровно round(0.15 L) токенов шума, round(·/3) спанов, текст первым, шум последним),
  а не независимые броски: так тесты проверяют точные числа, а длина входа предсказуема (568 → 512). Sentinel-ы передаются
  списком id (`<extra_id_i>` = 32 099 − i у T5). Цель — та же функция над `~mask`, как в T5; README объясняет, почему это
  работает только при маске, начинающейся с текста.
- **B15d — граничное значение сдвига маски (`new_ones` вместо `new_zeros`)**, а не «перевёрнутый порядок sentinel-ов»: вход
  остаётся правильным, а цель съезжает на кусок — правдоподобная ошибка на единицу.

## Известные проблемы
- Холодный старт subprocess с torch ≈ 1–1.5 с на прогон — приемлемо для UI, но заметно.

## Журнал сессий
| Дата | Фаза | Что сделано | Коммит |
|------|------|-------------|--------|
| 2026-10-16 | 0 | Каркас, загрузчик, раннер, validate, CLI, GPT-2, C00, TASK_FORMAT, CI, 32 теста | см. `git log` |
| 2026-10-16 | 1 | FastAPI API, `llmlab serve`, React/Monaco/KaTeX UI, viz-реестр + causal-mask, E2E на C00, 42 теста | см. `git log` |
| 2026-10-16 | 2 | Автоконспекты, FTS5-поиск, экспорт, таблица архитектур, вкладка «Конспекты», модалка вывода, 57 тестов | см. `git log` |
| 2026-10-16 | 3a | C01 RMSNorm, C02 SwiGLU, C03 RoPE + 3 debug-задачи, 5 статей, 8 вопросов, строка Llama 3 | см. `git log` |
| 2026-10-16 | 3b | C04 GQA, C05 KV-cache, C06 Llama + 3 debug, C-P1, C-Q1, V-rope, V-kvcache, вкладка «Вопросы», core/llama.py, 64 теста | см. `git log` |
| 2026-10-16 | 4 | C07 SWA, C08 нормы, C09 QK-Norm, C10 NoPE, C11 кольцевой KV-cache + 3 debug, C-P2, C-Q2, V-masks, SWA в V-kvcache, core/gemma_olmo.py, 69 тестов | см. `git log` |
| 2026-10-16 | 5a | C12 роутер, C13 MoE-слой, C14 балансировка, C15 DeepSeekMoE + 3 debug, 10 статей, 15 вопросов, core/moe.py, 73 теста | см. `git log` |
| 2026-10-16 | 5b | C16 MLA, C16b decoupled RoPE, C17 MTP + C16d, C-P3, C-Q3, V-moe-routing, MLA в V-kvcache, core/mla.py, 76 тестов | см. `git log` |
| 2026-10-17 | 6a | C18 DSA, C19 gated attention + partial RoPE, C20 линейное внимание + C20d, 3 статьи, 12 вопросов, строка DeepSeek-V3.2, sparse-режим V-masks, core/efficient.py, 79 тестов | см. `git log` |
| 2026-10-17 | 6b | C21 Gated DeltaNet + C21d, C22 гибрид 3:1, C23 Mamba-2 SSD, C-P4, C-Q4, 3 статьи, 16 вопросов, столбец `layers`, V-hybrid, 82 теста | см. `git log` |
| 2026-10-17 | 7 | Импортёр вопросов + CLI, 244 сырых → 105 вопросов (7 файлов), SM-2 + SQLite v3, API и вкладка Interview, самооценка в задачах, «пройдено» для paper/quiz, 94 теста | см. `git log` |
| 2026-10-17 | 8 | core/model.py из YAML + реестр, 9 конфигов, core/hf.py, C24 + C25 (Qwen3-0.6B, логиты = transformers), `llmlab model/generate`, `--weights`, docs/MODEL.md, 122 теста | см. `git log` |
| 2026-10-17 | 9 | `llmlab doctor`, установка с нуля (macOS локально, Linux/macOS/Windows в CI), UTF-8 для Windows, README с GIF и скриншотами, CONTRIBUTING, NOTICE, CHANGELOG, ячейки FFN Gemma 3/OLMo 2, фикс Monaco, v0.1.0, 129 тестов | см. `git log` |
| 2026-10-17 | ревью | `docs/REVIEW-2026-10-17.md`: факты (26 config.json, arXiv), мутационное тестирование (1271 мутант, 88 %), понятность, заимствования, инварианты; 12 находок | см. `git log` |
| 2026-10-17 | ревью-фиксы | 12 находок ревью: защита от импорта эталона, C14, допуски, C16 k_rope, C25a-sampling, paper-задачи → C-P2b/P3b/P4b, тексты; 54 задачи, 130 тестов | см. `git log` |
| 2026-10-17 | правки UI | Навигация «1.1 · RMSNorm» вместо id (`short_title`, `Course.labels()`), ```shapes + ShapeFlow во всех 44 README, ссылки на задачи в теории, 11 новых визуализаций (у всех задач, кроме квизов), 131 тест | см. `git log` |
| 2026-10-17 | правки UI | «☰ Курс» на странице задачи (модули, задачи, статусы), ← / → к соседним задачам | см. `git log` |
| 2026-10-17 | выпуск | v0.1.1: версия, CHANGELOG, тег, GitHub Release; 134 теста, 54 задачи; блок A расписан в PLAN (A1–A4) | см. `git log` |
| 2026-10-18 | A1 | Платформа под несколько блоков (A1.2/C1.3, optional), A01–A05 + A01d, A04d, A-P1, A-Q1, core/autograd.py, 5 визуализаций, 21 вопрос, 138 тестов, 63 задачи | см. `git log` |
| 2026-10-18 | A2 | A06–A10 + A07d, A09d, A-P2, A-P2b, A-Q2, корпус corpus-ru.txt, 4 визуализации, 23 вопроса, 73 задачи | см. `git log` |
| 2026-10-18 | A3 | A11–A14 + A12d, A13d, A-P3, A-Q3, core/bpe.py, V-bpe, 22 вопроса, зависимость regex, 142 теста, 81 задача | см. `git log` |
| 2026-10-18 | правка A3 | корпус 8.4 → 23.1 тыс. символов, пересчёт чисел A10–A14 | см. `git log` |
| 2026-10-18 | A4 | A15–A18 + A16d, A-P4, A-Q4, A-Q4b, core/minigpt.py, V-training-curve, `recommended_after`, 23 вопроса, README-скриншоты/GIF, v0.2.0, 147 тестов, 89 задач | см. `git log` |
| 2026-10-18 | B1 | План блока B (B1–B4), B01–B06 + B01d, B04d, B05d, B-P1, B-Q1, core/encdec.py, 3 визуализации, 24 вопроса, 4 статьи, 153 теста, 100 задач | см. `git log` |
| 2026-10-18 | B2 | B07–B11 + B09d, B11d, B-P2, B-Q2, core/bert.py (== transformers/tokenizers), 2 визуализации, 20 вопросов, 5 статей, weights-тесты bert-base-uncased, 159 тестов, 109 задач | см. `git log` |
| 2026-10-18 | B3 | B12–B15 + B13d, B15d, B-P3, B-Q3, core/t5.py (== transformers T5), 2 визуализации, 22 вопроса, 4 статьи, weights-тест t5-small, 165 тестов, 117 задач | см. `git log` |
