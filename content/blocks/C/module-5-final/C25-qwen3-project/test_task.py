"""Тесты C25 (проект): Qwen3-0.6B на своих деталях — загрузка, сэмплирование, генерация.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Без весов (по умолчанию): игрушечный чекпойнт в формате HF (config.json + bf16-шарды + index.json) пишется
во временную папку; загрузка даёт ровно те логиты, что исходная модель; связь lm_head ↔ embed_tokens
сохранена; сэмплирование соблюдает temperature / top-k / top-p (проверяется частотами на 4000 выборок);
генерация совпадает с полным перепрогоном, использует KV-cache и останавливается на eos.

С весами (`llmlab test C25-qwen3-project --weights`, нужен `uv sync --extra weights`, ~1.5 ГБ): логиты
настоящей Qwen3-0.6B совпадают с transformers, жадная генерация — токен в токен, модель знает столицу Франции.
Бонус: то же для GPT-2 и SmolLM2-135M.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.testing import assert_close

from llmlab.core import hf as core_hf
from llmlab.core.model import Model
from llmlab.core.model import generate as full_recompute_generate

TINY_QWEN3 = {
    "model_type": "qwen3", "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 48, "hidden_act": "silu", "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0, "rope_scaling": None, "attention_bias": False, "use_sliding_window": False,
    "tie_word_embeddings": True, "max_position_embeddings": 128,
}
TINY_GPT2 = {"model_type": "gpt2", "vocab_size": 64, "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 32,
             "layer_norm_epsilon": 1e-5, "activation_function": "gelu_new"}


def _tiny_model(hf_cfg=TINY_QWEN3, seed=0) -> Model:
    """Случайная модель с весами, точно представимыми в bf16, и «живыми» нормами (веса ≈ 1)."""
    torch.manual_seed(seed)
    m = Model(core_hf.config_from_hf(hf_cfg)).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3 if p.ndim == 1 else 0.15)
            if p.ndim == 1 and hf_cfg["model_type"] == "qwen3":
                p.add_(1.0)
            p.copy_(p.to(torch.bfloat16).float())
    return m


def _write_qwen3_checkpoint(model: Model, path: Path, shards: int = 2) -> Path:
    """Чекпойнт как у HF: config.json + model-0000i-of-0000n.safetensors (bf16) + index.json."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(TINY_QWEN3), encoding="utf-8")
    sd = {("" if k == "lm_head.weight" else "model.") + k: v.to(torch.bfloat16).clone()
          for k, v in model.state_dict().items()}
    names = sorted(sd)
    if shards == 1:
        save_file(sd, str(path / "model.safetensors"))
        return path
    weight_map = {}
    for s in range(shards):
        fname = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        part = {n: sd[n] for n in names[s::shards]}
        save_file(part, str(path / fname))
        weight_map.update({n: fname for n in part})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    return path


def _write_gpt2_checkpoint(model: Model, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(TINY_GPT2), encoding="utf-8")
    s = model.state_dict()
    sd = {"wte.weight": s["embed_tokens.weight"], "wpe.weight": s["embed_positions.weight"],
          "ln_f.weight": s["norm.weight"], "ln_f.bias": s["norm.bias"]}
    for i in range(model.cfg.n_layers):
        p, h = f"layers.{i}.", f"h.{i}."
        sd[h + "ln_1.weight"], sd[h + "ln_1.bias"] = s[p + "input_layernorm.weight"], s[p + "input_layernorm.bias"]
        sd[h + "ln_2.weight"] = s[p + "post_attention_layernorm.weight"]
        sd[h + "ln_2.bias"] = s[p + "post_attention_layernorm.bias"]
        sd[h + "attn.c_attn.weight"] = torch.cat([s[p + f"self_attn.{n}_proj.weight"] for n in "qkv"]).t()
        sd[h + "attn.c_attn.bias"] = torch.cat([s[p + f"self_attn.{n}_proj.bias"] for n in "qkv"])
        sd[h + "attn.c_proj.weight"], sd[h + "attn.c_proj.bias"] = s[p + "self_attn.o_proj.weight"].t(), \
            s[p + "self_attn.o_proj.bias"]
        sd[h + "mlp.c_fc.weight"], sd[h + "mlp.c_fc.bias"] = s[p + "mlp.c_fc.weight"].t(), s[p + "mlp.c_fc.bias"]
        sd[h + "mlp.c_proj.weight"], sd[h + "mlp.c_proj.bias"] = s[p + "mlp.c_proj.weight"].t(), \
            s[p + "mlp.c_proj.bias"]
    save_file({k: v.contiguous().clone() for k, v in sd.items()}, str(path / "model.safetensors"))
    return path


def _idx(B=2, T=6, V=64, seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def test_load_safetensors_shards(impl, tmp_path):
    """load_safetensors: один файл и шарды по model.safetensors.index.json — все тензоры, ни одного лишнего"""
    model = _tiny_model()
    want = core_hf.load_safetensors(_write_qwen3_checkpoint(model, tmp_path / "one", shards=1))
    got = impl.load_safetensors(tmp_path / "one")
    assert set(got) == set(want), f"model.safetensors: ключи не совпали; лишние {sorted(set(got) - set(want))[:3]}."
    got3 = impl.load_safetensors(_write_qwen3_checkpoint(model, tmp_path / "three", shards=3))
    missing = sorted(set(want) - set(got3))
    if missing:
        pytest.fail(f"Из шардированного чекпойнта загружено {len(got3)} тензоров из {len(want)} (нет, например, "
                    f"{missing[:2]}). Большие модели режут на файлы model-0000i-of-0000n.safetensors; список — в "
                    "model.safetensors.index.json → weight_map (имя тензора → файл). Прочитай все файлы.", pytrace=False)
    for k in want:
        assert torch.equal(got3[k], want[k]), f"{k}: тензор отличается от записанного."


def test_load_pretrained_logits(impl, tmp_path):
    """load_pretrained: игрушечный Qwen3-чекпойнт (bf16, 2 шарда) → Model с теми же логитами, eval, float32"""
    orig = _tiny_model()
    model = impl.load_pretrained(_write_qwen3_checkpoint(orig, tmp_path / "ckpt"))
    assert isinstance(model, Model), f"Нужна llmlab.core.model.Model, получено {type(model).__name__}."
    assert not model.training, "Верни модель в режиме eval(): у загруженной для инференса модели не должно быть dropout."
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}, f"По умолчанию параметры — float32 (dtype=torch.float32), получено {dtypes}."
    with torch.no_grad():
        assert_close(model(_idx()), orig(_idx()), atol=1e-5, rtol=1e-5,
                     msg=lambda m: f"Логиты загруженной модели не совпали с исходной.\n{m}")


def test_load_pretrained_keeps_tie(impl, tmp_path):
    """load_pretrained: при tie_word_embeddings lm_head и embed_tokens — один и тот же тензор"""
    model = impl.load_pretrained(_write_qwen3_checkpoint(_tiny_model(), tmp_path / "ckpt"))
    if model.lm_head.weight is not model.embed_tokens.weight:
        pytest.fail("После загрузки lm_head.weight и embed_tokens.weight — разные тензоры. Логиты пока совпадают, но "
                    "память удвоена (у Qwen3-0.6B это +155 M параметров = +0.6 ГБ в fp32), а дообучение разведёт "
                    "матрицы. load_state_dict(..., assign=True) подменяет параметры и рвёт связь — восстанови её: "
                    "model.lm_head.weight = model.embed_tokens.weight.", pytrace=False)


def test_load_pretrained_dtype_and_gpt2(impl, tmp_path):
    """load_pretrained: dtype=bfloat16 → параметры bf16; GPT-2 (Conv1D-веса) грузится той же функцией"""
    model = impl.load_pretrained(_write_qwen3_checkpoint(_tiny_model(), tmp_path / "q"), dtype=torch.bfloat16)
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}, "dtype=torch.bfloat16 → все параметры bf16."
    gpt = _tiny_model(TINY_GPT2)
    loaded = impl.load_pretrained(_write_gpt2_checkpoint(gpt, tmp_path / "g"))
    with torch.no_grad():
        assert_close(loaded(_idx()), gpt(_idx()), atol=1e-5, rtol=1e-5,
                     msg=lambda m: f"GPT-2: логиты загруженной модели не совпали (model_type берётся из config.json).\n{m}")


def test_load_pretrained_matches_transformers(impl, tmp_path):
    """С transformers: Qwen3ForCausalLM.save_pretrained → load_pretrained → те же логиты, что у HF"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    torch.manual_seed(0)
    hf_model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(**{k: v for k, v in TINY_QWEN3.items() if k != "model_type"})).eval()
    hf_model.save_pretrained(tmp_path / "hf")
    model = impl.load_pretrained(tmp_path / "hf")
    idx = _idx()
    with torch.no_grad():
        assert_close(model(idx), hf_model(idx).logits, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Сэмплирование
# ---------------------------------------------------------------------------

def _draws(impl, logits, n=4000, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    out = impl.sample_next(logits.expand(n, -1).clone(), generator=g, **kw)
    assert out.shape == (n, 1), f"sample_next возвращает [B, 1], получено {tuple(out.shape)}."
    assert out.dtype == torch.long, f"Токены — int64 (long), получено {out.dtype}."
    return torch.bincount(out[:, 0], minlength=logits.shape[-1]).float() / n


PROBS = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
LOGITS = PROBS.log() + 3.0                                  # логиты определены с точностью до сдвига


def test_sample_greedy_and_top_k(impl):
    """sample_next: temperature = 0 → argmax (без случайности); top_k = 1 → тоже argmax при любой температуре"""
    logits = torch.randn(5, 64, generator=torch.Generator().manual_seed(3))
    got = impl.sample_next(logits, temperature=0.0)
    assert got.shape == (5, 1), f"sample_next возвращает [B, 1], получено {tuple(got.shape)}."
    assert torch.equal(got[:, 0], logits.argmax(-1)), "temperature = 0 — жадный выбор: argmax по последней оси."
    freq = _draws(impl, LOGITS, temperature=1.5, top_k=1)
    assert freq[0] == 1.0, f"top_k = 1 оставляет только самый вероятный токен, а выпадали и другие: {freq.tolist()}."
    freq = _draws(impl, LOGITS, temperature=1.0, top_k=2)
    assert freq[2:].sum() == 0, f"top_k = 2: токены 2 и 3 вне двух лучших, но выпадали: {freq.tolist()}."
    assert abs(freq[0] - 0.625) < 0.03, (
        f"top_k = 2: после отсечения вероятности перенормируются: 0.5 / 0.8 = 0.625, получено {freq[0]:.3f}.")


def test_sample_temperature(impl):
    """sample_next: логиты делятся на T до softmax — T = 0.5 делает распределение острее, T = 2 — площе"""
    logits = torch.tensor([[2.0, 0.0]])
    sharp = _draws(impl, logits, temperature=0.5)[0].item()     # e⁴ / (e⁴ + 1) ≈ 0.982
    flat = _draws(impl, logits, temperature=2.0)[0].item()      # e / (e + 1) ≈ 0.731
    assert abs(sharp - 0.982) < 0.015 and abs(flat - 0.731) < 0.03, (
        f"P(токен 0) при T = 0.5: {sharp:.3f} (нужно ≈ 0.982), при T = 2: {flat:.3f} (нужно ≈ 0.731). "
        "softmax(logits / T): T < 1 обостряет, T > 1 сглаживает. Умножение вместо деления даёт обратный эффект.")


def test_sample_top_p(impl):
    """sample_next: top-p — наименьший набор самых вероятных токенов с суммой ≥ p; токен, перешедший порог, остаётся"""
    freq = _draws(impl, LOGITS, temperature=1.0, top_p=0.7)     # 0.5 < 0.7 ≤ 0.5 + 0.3 → ядро {0, 1}
    if freq[1] == 0:
        pytest.fail("top_p = 0.7: выпадал только токен 0. Ядро — наименьший набор с суммой ≥ p: 0.5 < 0.7, поэтому "
                    "токен 1 (накопленная сумма 0.8) тоже входит. Типичная ошибка — маска cumsum > p выкидывает "
                    "токен, который пересёк порог; сдвинь cumsum на один (масса токенов ДО текущего).", pytrace=False)
    if freq[2:].sum() > 0:
        pytest.fail(f"top_p = 0.7: выпадали токены 2–3 (частоты {freq.tolist()}), а ядро — {{0, 1}}: суммы 0.5 и 0.8.",
                    pytrace=False)
    assert abs(freq[0] - 0.625) < 0.03, f"Внутри ядра вероятности перенормируются: 0.5 / 0.8 = 0.625, получено {freq[0]:.3f}."
    freq = _draws(impl, LOGITS, temperature=1.0, top_p=0.4)
    assert freq[0] == 1.0, "top_p = 0.4 < 0.5: в ядре один самый вероятный токен (он остаётся всегда)."
    # порядок токенов в словаре не важен: ядро — по вероятностям, а не по индексам
    perm = torch.tensor([2, 0, 3, 1])
    freq = _draws(impl, LOGITS[:, perm], temperature=1.0, top_p=0.7)
    assert freq[0] == 0 and freq[2] == 0 and freq[1] > 0 and freq[3] > 0, (
        f"Логиты переставлены (вероятные токены — 1 и 3), частоты {freq.tolist()}. Сортируй по вероятности и "
        "возвращай маску на исходные места (scatter по индексам сортировки).")


def test_sample_reproducible(impl):
    """sample_next: один и тот же torch.Generator с тем же сидом → те же токены"""
    logits = torch.randn(8, 64, generator=torch.Generator().manual_seed(5))
    a = impl.sample_next(logits, temperature=1.0, top_p=0.9, generator=torch.Generator().manual_seed(7))
    b = impl.sample_next(logits, temperature=1.0, top_p=0.9, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "Передай generator в torch.multinomial — иначе генерацию не воспроизвести."


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------

def test_generate_greedy_matches_full_recompute(impl):
    """generate: жадная генерация с KV-cache == перепрогон всего префикса на каждом шаге"""
    model = _tiny_model()
    idx = _idx(T=5)
    got = impl.generate(model, idx, 10)
    assert got.shape == (2, 15), f"Ожидалась форма [2, 5 + 10], получено {tuple(got.shape)}."
    assert torch.equal(got[:, :5], idx), "Первые T токенов результата — это промпт без изменений."
    want = full_recompute_generate(model, idx, 10, use_cache=False)
    assert torch.equal(got, want), f"Жадная генерация разошлась с перепрогоном:\n{got}\n{want}"


def test_generate_uses_cache(impl):
    """generate: после prefill модель получает по одному токену; прогонов ровно 1 + (новых токенов − 1)"""
    model = _tiny_model()
    seen = []
    handle = model.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[1]))
    try:
        impl.generate(model, _idx(T=5), 6)
    finally:
        handle.remove()
    assert seen and seen[0] == 5, f"Первый прогон — prefill всего промпта (5 токенов), получено {seen[:1]}."
    if any(t != 1 for t in seen[1:]):
        pytest.fail(f"Длины входов по шагам: {seen}. После prefill с KV-cache в модель подаётся только новый токен "
                    "[B, 1]: model(next_token, caches). Перепрогон всего префикса — квадратичная стоимость (C05).",
                    pytrace=False)
    assert len(seen) == 6, (f"Прогонов модели: {len(seen)}, нужно 6 = prefill + 5. Логиты для 6-го токена уже есть "
                            "после 5-го прогона; прогон после последнего токена лишний.")


def test_generate_stops_on_eos(impl, ref):
    """generate: строка, выдавшая eos, дальше дополняется eos; когда eos у всех — генерация останавливается"""
    model = _tiny_model()
    idx = _idx(B=1, T=5)
    free = full_recompute_generate(model, idx, 8, use_cache=False)[0, 5:]
    eos = int(free[2])
    first = int((free == eos).nonzero()[0])                                 # первое появление eos
    got = impl.generate(model, idx, 8, eos_token_id=eos)
    assert got.shape[1] == 5 + first + 1, (
        f"eos = {eos} выпал на шаге {first + 1}: генерация должна остановиться сразу (длина {5 + first + 1}), "
        f"получено {got.shape[1]}.")
    batch = torch.cat((idx, _idx(B=1, T=5, seed=9)))
    got = impl.generate(model, batch, 8, eos_token_id=eos)
    want = ref.generate(model, batch, 8, eos_token_id=eos)
    assert torch.equal(got, want), (
        f"Батч из двух строк, eos = {eos}:\nполучено {got.tolist()}\nнужно    {want.tolist()}\n"
        "После eos строка продолжает получать eos (а не случайные токены), пока не закончат все строки.")


# ---------------------------------------------------------------------------
# Настоящие веса (--weights)
# ---------------------------------------------------------------------------

PROMPT = "The capital of France is Paris. The capital of Germany is"


@pytest.fixture(scope="module")
def qwen3(impl):
    pytest.importorskip("huggingface_hub", reason="нужен huggingface_hub: uv sync --extra weights")
    path = core_hf.download("Qwen/Qwen3-0.6B")
    return path, impl.load_pretrained(path), core_hf.load_tokenizer(path)


@pytest.mark.weights
def test_qwen3_param_count(qwen3):
    """Qwen3-0.6B: 596 049 920 параметров (0.44 B без эмбеддингов), lm_head связан с embed_tokens"""
    _, model, _ = qwen3
    assert model.num_params() == 596_049_920 and model.num_params(exclude_embeddings=True) == 440_467_456
    assert model.lm_head.weight is model.embed_tokens.weight


@pytest.mark.weights
def test_qwen3_logits_match_transformers(qwen3):
    """Qwen3-0.6B: логиты своей модели == transformers (fp32, |Δ| ≤ 1e-3 при логитах до ~20; фактически ~6e-5)"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    path, model, tok = qwen3
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                                 attn_implementation="eager").eval()
    idx = torch.tensor([tok.encode(PROMPT).ids])
    with torch.no_grad():
        ours, theirs = model(idx), hf_model(idx).logits
    # Допуск: обе модели в fp32, разница — только порядок суммирования в matmul / softmax / RMSNorm
    # (ошибка ~1e-7 относительно на операцию, накопленная через 28 слоёв). Любая настоящая ошибка (не та норма,
    # не тот θ, перепутанные головы) даёт расхождение порядка единиц.
    assert_close(ours, theirs, atol=1e-3, rtol=1e-3)
    assert torch.equal(ours.argmax(-1), theirs.argmax(-1))


@pytest.mark.weights
def test_qwen3_generation(impl, qwen3):
    """Qwen3-0.6B: жадная генерация знает столицу Германии и совпадает с transformers токен в токен"""
    path, model, tok = qwen3
    idx = torch.tensor([tok.encode(PROMPT).ids])
    out = impl.generate(model, idx, 12)
    text = tok.decode(out[0, idx.shape[1]:].tolist())
    assert "Berlin" in text, f"Продолжение: {text!r} — модель с правильными весами отвечает «Berlin»."
    transformers = pytest.importorskip("transformers", reason="для сверки токенов нужен transformers")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    want = hf_model.generate(idx, max_new_tokens=12, do_sample=False)
    assert torch.equal(out, want), (f"Жадная генерация разошлась с transformers:\n{tok.decode(out[0].tolist())!r}\n"
                                    f"{tok.decode(want[0].tolist())!r}")


@pytest.mark.weights
@pytest.mark.parametrize("repo", ["openai-community/gpt2", "HuggingFaceTB/SmolLM2-135M"])
def test_bonus_other_architectures(impl, repo):
    """Бонус: та же load_pretrained грузит GPT-2 (LayerNorm, learned позиции, Conv1D) и SmolLM2 (Llama)"""
    pytest.importorskip("huggingface_hub", reason="нужен huggingface_hub: uv sync --extra weights")
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    path = core_hf.download(repo)
    model, tok = impl.load_pretrained(path), core_hf.load_tokenizer(path)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                                 attn_implementation="eager").eval()
    idx = torch.tensor([tok.encode(PROMPT).ids])
    with torch.no_grad():
        assert_close(model(idx), hf_model(idx).logits, atol=1e-3, rtol=1e-3)
    text = tok.decode(impl.generate(model, idx, 3)[0, idx.shape[1]:].tolist())
    assert "Berlin" in text, f"{repo}: продолжение {text!r}."
