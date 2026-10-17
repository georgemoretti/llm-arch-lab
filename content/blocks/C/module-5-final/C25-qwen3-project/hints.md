## Подсказка 1

`load_safetensors`: `index = Path(model_dir) / "model.safetensors.index.json"`; если он есть —
`shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))`, иначе `["model.safetensors"]`;
`sd.update(load_file(model_dir / shard))` для каждого. `load_pretrained`: `hf = json.loads(...)`,
`cfg = config_from_hf(hf)`, `model = Model(cfg)`, `sd = convert_state_dict(load_safetensors(d), cfg, hf["model_type"], dtype)`,
`check_state_dict(model, sd)`, `model.load_state_dict(sd)`, `model.to(dtype).eval()`.

## Подсказка 2

Связь `lm_head` ↔ `embed_tokens`: обычный `load_state_dict` копирует значения в существующие параметры, и связь
сохраняется. Если строишь модель под `with torch.device("meta"):` (быстро: нет случайной инициализации 0.6 B
параметров) и грузишь `load_state_dict(sd, assign=True)`, параметры подменяются тензорами из `sd` — и связь
рвётся. Тогда после загрузки: `model.lm_head.weight = model.embed_tokens.weight`.

## Подсказка 3

Генерация: `caches = model.new_caches()`, `logits = model(idx, caches)`; в цикле `nxt = sample_next(logits[:, -1], ...)`; при eos — `nxt.masked_fill(done[:, None], eos)`,
`done |= nxt[:, 0] == eos`; выход, если `done.all()` или это последний шаг; иначе `logits = model(nxt, caches)`.
