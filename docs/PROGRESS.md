# Прогресс

> Обновляется в конце КАЖДОЙ сессии Claude Code. Пиши кратко и конкретно.

## Текущая фаза
Правки пользователя (2026-10-17, после ревью): понятная навигация без кодов задач, схемы форм тензоров вместо
ASCII, визуализации у всех задач — **сделано**, ждёт просмотра пользователем.

Предыдущее: Исправления по ревью — **завершены** (2026-10-17). Все 12 находок `docs/REVIEW-2026-10-17.md` закрыты; задач 54
(+ C25a-sampling, C-P2b, C-P3b, C-P4b), вопросов 231. Все фазы PLAN.md выполнены; изменения — в CHANGELOG → Unreleased.

## Следующий шаг (точный)
Фазы плана закончились. Варианты (выбор за пользователем): (1) выпустить v0.1.1 из CHANGELOG → Unreleased (тег +
GitHub Release); (2) новый блок из PLAN «Позже» (A — фундамент, B — энкодеры, D — обучение и инференс) — сначала
расписать его фазы в PLAN.md; (3) долги из «Не сделано» (E2E UI в репозитории, загрузка весов Gemma 3 / OLMo 2 —
отложено пользователем).

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

## Не сделано / долги
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
- Фаза 9: скриншоты README сняты вручную скриптом из scratchpad (Playwright вне репо) — при заметных изменениях UI
  переснять. Проверка установки на Linux/Windows — только в CI (локального Docker не было).

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
