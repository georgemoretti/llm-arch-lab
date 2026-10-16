"""Тесты C24: config.json и веса чекпойнта Hugging Face → своя модель.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: config.json Qwen3-0.6B / SmolLM2-135M / GPT-2 превращается в те же конфиги, что лежат в
llmlab/configs (каждое число сверено с официальным config.json); неподдерживаемые опции не проглатываются
молча; после конвертации весов модель даёт РОВНО те же логиты, что исходная. «Чекпойнт» в тестах без весов
строится из случайной модели обратным преобразованием (имена и раскладка — как в HF), а если установлен
transformers — берётся из настоящих Qwen3ForCausalLM / GPT2LMHeadModel игрушечного размера.
"""

import copy

import pytest
import torch
from torch.testing import assert_close

from llmlab.core.model import Model, ModelConfig

# Официальные config.json (только поля, которые читает курс + пара лишних, как в реальном файле).
# https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
QWEN3_06B = {
    "architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 1024,
    "initializer_range": 0.02, "intermediate_size": 3072, "max_position_embeddings": 40960, "max_window_layers": 28,
    "model_type": "qwen3", "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 1000000, "sliding_window": None,
    "tie_word_embeddings": True, "torch_dtype": "bfloat16", "use_cache": True, "use_sliding_window": False,
    "vocab_size": 151936,
}
# https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json — Llama без поля head_dim
SMOLLM2_135M = {
    "architectures": ["LlamaForCausalLM"], "attention_bias": False, "hidden_act": "silu", "hidden_size": 576,
    "intermediate_size": 1536, "max_position_embeddings": 8192, "mlp_bias": False, "model_type": "llama",
    "num_attention_heads": 9, "num_hidden_layers": 30, "num_key_value_heads": 3, "rms_norm_eps": 1e-05,
    "rope_scaling": None, "rope_theta": 100000, "tie_word_embeddings": True, "vocab_size": 49152,
}
# https://huggingface.co/openai-community/gpt2/blob/main/config.json
GPT2 = {
    "activation_function": "gelu_new", "architectures": ["GPT2LMHeadModel"], "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2", "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024,
    "vocab_size": 50257,
}

# Игрушечные config.json в тех же форматах. head_dim = 16 ≠ hidden_size / heads = 8 — как у Qwen3.
TINY_QWEN3 = {
    "model_type": "qwen3", "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 48, "hidden_act": "silu", "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0, "rope_scaling": None, "attention_bias": False, "use_sliding_window": False,
    "tie_word_embeddings": True, "max_position_embeddings": 128,
}
TINY_LLAMA = {
    "model_type": "llama", "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 1, "intermediate_size": 48, "hidden_act": "silu", "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0, "rope_scaling": None, "attention_bias": False, "mlp_bias": False,
    "tie_word_embeddings": False, "max_position_embeddings": 128,
}
TINY_GPT2 = {"model_type": "gpt2", "vocab_size": 64, "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 32,
             "layer_norm_epsilon": 1e-5, "activation_function": "gelu_new"}


def _fields(cfg: ModelConfig) -> dict:
    return cfg.model_dump(exclude={"name", "source"})


def _diff(got: dict, want: dict, prefix="") -> list[str]:
    out = []
    for k, v in want.items():
        if isinstance(v, dict) and isinstance(got.get(k), dict):
            out += _diff(got[k], v, f"{prefix}{k}.")
        elif got.get(k) != v:
            out.append(f"{prefix}{k}: получено {got.get(k)!r}, нужно {v!r}")
    return out


_HINTS = {
    "attention.head_dim": " У Qwen3 head_dim задан в config.json явно (128) и НЕ равен hidden_size / heads (64): "
                          "q_proj выдаёт 16 · 128 = 2048 каналов при d_model = 1024. Бери hf['head_dim'], если он есть.",
    "attention.qk_norm": " У Qwen3 есть q_norm / k_norm по голове (их веса лежат в чекпойнте) — qk_norm = 'head'; "
                         "у Llama и GPT-2 их нет.",
    "tie_embeddings": " tie_word_embeddings: у Qwen3-0.6B и SmolLM2 выход связан с эмбеддингами; если ключа нет "
                      "(GPT-2) — у GPT-2 связь есть всегда.",
    "attention.n_kv_heads": " n_kv_heads = num_key_value_heads (GQA); у GPT-2 — MHA, n_kv_heads = n_heads (или None).",
    "position.type": " GPT-2 — обучаемые позиции (learned, wpe), Llama / Qwen3 — RoPE.",
}


def _check_config(impl, hf: dict, name: str):
    got = impl.config_from_hf(copy.deepcopy(hf))
    assert isinstance(got, ModelConfig), f"config_from_hf должна вернуть ModelConfig, получено {type(got).__name__}."
    want = _fields(ModelConfig.named(name))
    got_f = _fields(got)
    if want["attention"]["n_kv_heads"] is None:                       # MHA: None и n_heads — одно и то же
        got_f["attention"]["n_kv_heads"] = None if got_f["attention"]["n_kv_heads"] in (None, want["attention"]["n_heads"]) \
            else got_f["attention"]["n_kv_heads"]
    if want["attention"]["head_dim"] is None:                         # None = d_model // n_heads
        got_f["attention"]["head_dim"] = None if got.head_dim == ModelConfig.named(name).head_dim \
            else got_f["attention"]["head_dim"]
    problems = _diff(got_f, want)
    if problems:
        hints = "".join(h for key, h in _HINTS.items() if any(p.startswith(key) for p in problems))
        pytest.fail(f"Конфиг {name} из config.json не совпал с llmlab/configs/{name}.yaml:\n  "
                    + "\n  ".join(problems) + hints, pytrace=False)


# ---------------------------------------------------------------------------
# config.json → ModelConfig
# ---------------------------------------------------------------------------

def test_config_qwen3(impl):
    """config_from_hf: официальный config.json Qwen3-0.6B → конфиг qwen3-0.6b (head_dim 128, QK-Norm, tied)"""
    _check_config(impl, QWEN3_06B, "qwen3-0.6b")


def test_config_llama_and_gpt2(impl):
    """config_from_hf: SmolLM2-135M (llama, без head_dim в config.json) и GPT-2 (свои имена полей)"""
    _check_config(impl, SMOLLM2_135M, "smollm2-135m")
    _check_config(impl, GPT2, "gpt2")


@pytest.mark.parametrize("patch,why", [
    ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "YaRN меняет частоты RoPE — без него длинный контекст сломается"),
    ({"use_sliding_window": True}, "окно меняет маску attention"),
    ({"attention_bias": True}, "bias у q/k/v/o — лишние веса, которых в модели нет"),
    ({"hidden_act": "gelu"}, "другая функция активации в FFN"),
    ({"model_type": "gemma3_text"}, "другая архитектура целиком"),
], ids=["rope_scaling", "sliding_window", "attention_bias", "hidden_act", "model_type"])
def test_config_rejects_unsupported(impl, patch, why):
    """config_from_hf: неподдерживаемые опции → ValueError, а не модель, которая молча считает не то"""
    hf = {**QWEN3_06B, **patch}
    try:
        impl.config_from_hf(hf)
    except ValueError:
        return
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{patch}: нужна ValueError с объяснением, получено {type(exc).__name__}: {exc}", pytrace=False)
    pytest.fail(f"{patch} молча принят. Нужно ValueError: {why}. Модель с такими весами загрузится без ошибок, "
                "но будет выдавать мусор — худший вид бага.", pytrace=False)


# ---------------------------------------------------------------------------
# Веса: «чекпойнт» в формате HF из случайной модели
# ---------------------------------------------------------------------------

def _bf16_model(cfg: ModelConfig, seed=0) -> Model:
    """Случайная модель, чьи веса точно представимы в bf16 (чекпойнты Qwen3 хранятся в bf16)."""
    torch.manual_seed(seed)
    m = Model(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3 if p.ndim == 1 else 0.1)
            if p.ndim == 1 and p.shape[0] <= 64:
                p.add_(1.0)                                                       # веса норм ≈ 1
            p.copy_(p.to(torch.bfloat16).float())
    return m


def _to_hf_llama(model: Model, include_lm_head=True) -> dict:
    """Обратное преобразование: наши имена → имена Qwen3/Llama в HF (префикс model.), dtype bf16."""
    sd = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight":
            if include_lm_head:
                sd[k] = v.to(torch.bfloat16).clone()
            continue
        sd["model." + k] = v.to(torch.bfloat16).clone()
    return sd


def _to_hf_gpt2(model: Model) -> dict:
    """Наши имена → GPT-2 в HF: Conv1D хранит веса как [in, out], q/k/v склеены в c_attn, есть буфер attn.bias."""
    s = model.state_dict()
    T = model.cfg.position.max_positions
    sd = {"wte.weight": s["embed_tokens.weight"], "wpe.weight": s["embed_positions.weight"],
          "ln_f.weight": s["norm.weight"], "ln_f.bias": s["norm.bias"]}
    for i in range(model.cfg.n_layers):
        p, h = f"layers.{i}.", f"h.{i}."
        sd[h + "ln_1.weight"], sd[h + "ln_1.bias"] = s[p + "input_layernorm.weight"], s[p + "input_layernorm.bias"]
        sd[h + "ln_2.weight"], sd[h + "ln_2.bias"] = (s[p + "post_attention_layernorm.weight"],
                                                      s[p + "post_attention_layernorm.bias"])
        sd[h + "attn.c_attn.weight"] = torch.cat([s[p + f"self_attn.{n}_proj.weight"] for n in "qkv"]).t().contiguous()
        sd[h + "attn.c_attn.bias"] = torch.cat([s[p + f"self_attn.{n}_proj.bias"] for n in "qkv"])
        sd[h + "attn.c_proj.weight"] = s[p + "self_attn.o_proj.weight"].t().contiguous()
        sd[h + "attn.c_proj.bias"] = s[p + "self_attn.o_proj.bias"]
        sd[h + "mlp.c_fc.weight"] = s[p + "mlp.c_fc.weight"].t().contiguous()
        sd[h + "mlp.c_fc.bias"] = s[p + "mlp.c_fc.bias"]
        sd[h + "mlp.c_proj.weight"] = s[p + "mlp.c_proj.weight"].t().contiguous()
        sd[h + "mlp.c_proj.bias"] = s[p + "mlp.c_proj.bias"]
        sd[h + "attn.bias"] = torch.tril(torch.ones(T, T)).view(1, 1, T, T)          # маска, а не вес
    return {k: v.clone() for k, v in sd.items()}


def _load_converted(impl, ref, hf_sd: dict, hf_cfg: dict) -> tuple[Model, ModelConfig]:
    cfg = ref.config_from_hf(hf_cfg)
    sd = impl.convert_state_dict(hf_sd, cfg, hf_cfg["model_type"])
    assert isinstance(sd, dict), f"convert_state_dict должна вернуть dict, получено {type(sd).__name__}."
    model = Model(cfg).eval()
    try:
        ref.check_state_dict(model, sd)
    except ValueError as exc:
        pytest.fail(f"{exc}", pytrace=False)
    bad_dtype = sorted({str(v.dtype) for v in sd.values() if v.dtype != torch.float32})
    assert not bad_dtype, (f"Тензоры после конвертации должны быть float32 (параметр dtype), есть {bad_dtype}. "
                           "Чекпойнт хранится в bf16 — его нужно привести.")
    model.load_state_dict(sd)
    return model, cfg


def _same_logits(a: Model, b: Model, V: int, msg: str):
    idx = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert_close(a(idx), b(idx), atol=1e-5, rtol=1e-5, msg=lambda m: f"{msg}\n{m}")


def test_convert_qwen3(impl, ref):
    """convert_state_dict: Qwen3 — снять префикс model., привести bf16 → fp32, связанный lm_head; логиты те же"""
    cfg = ref.config_from_hf(TINY_QWEN3)
    orig = _bf16_model(cfg)
    hf_sd = _to_hf_llama(orig)
    loaded, _ = _load_converted(impl, ref, hf_sd, TINY_QWEN3)
    _same_logits(loaded, orig, cfg.vocab_size, "Модель после конвертации считает не так, как исходная.")


def test_convert_tied_head(impl, ref):
    """convert_state_dict: при tie_word_embeddings lm_head = embed_tokens; расхождение в чекпойнте — ValueError"""
    cfg = ref.config_from_hf(TINY_QWEN3)
    orig = _bf16_model(cfg)
    # 1) SmolLM2-стиль: lm_head.weight в файле нет вовсе
    loaded, _ = _load_converted(impl, ref, _to_hf_llama(orig, include_lm_head=False), TINY_QWEN3)
    _same_logits(loaded, orig, cfg.vocab_size, "Чекпойнт без lm_head.weight (tie_word_embeddings = true): "
                 "выходная матрица = embed_tokens.weight.")
    # 2) lm_head есть, но отличается — конфиг и веса противоречат друг другу
    bad = _to_hf_llama(orig)
    bad["lm_head.weight"] = bad["lm_head.weight"] + 1
    try:
        impl.convert_state_dict(bad, cfg, "qwen3")
    except ValueError:
        return
    pytest.fail("tie_word_embeddings = true, а lm_head.weight в файле отличается от embed_tokens.weight — конфиг и "
                "веса противоречат друг другу. Нужна ValueError: молча выбрать одну из матриц — значит получить "
                "модель, которая считает не то, что обучали.", pytrace=False)


def test_convert_untied_and_junk(impl, ref):
    """convert_state_dict: Llama без связи — свой lm_head; служебные rotary_emb.inv_freq игнорируются"""
    cfg = ref.config_from_hf(TINY_LLAMA)
    orig = _bf16_model(cfg)
    hf_sd = _to_hf_llama(orig)
    for i in range(cfg.n_layers):                                       # так выглядят старые чекпойнты Llama 2
        hf_sd[f"model.layers.{i}.self_attn.rotary_emb.inv_freq"] = torch.ones(4)
    loaded, _ = _load_converted(impl, ref, hf_sd, TINY_LLAMA)
    assert loaded.lm_head.weight is not loaded.embed_tokens.weight
    _same_logits(loaded, orig, cfg.vocab_size, "Llama с раздельными embed/lm_head: lm_head.weight берётся из файла.")


def test_convert_gpt2(impl, ref):
    """convert_state_dict: GPT-2 — Conv1D [in, out] → Linear [out, in], c_attn → q/k/v, буфер attn.bias не вес"""
    cfg = ref.config_from_hf(TINY_GPT2)
    torch.manual_seed(0)
    orig = Model(cfg).eval()
    with torch.no_grad():
        for p in orig.parameters():
            p.normal_(0, 0.2)
    hf_sd = _to_hf_gpt2(orig)
    sd = impl.convert_state_dict(hf_sd, cfg, "gpt2")
    w_fc, w_o = sd.get("layers.0.mlp.c_fc.weight"), sd.get("layers.0.self_attn.o_proj.weight")
    if w_fc is not None and w_fc.shape == hf_sd["h.0.mlp.c_fc.weight"].shape:
        pytest.fail(f"mlp.c_fc.weight имеет форму {tuple(w_fc.shape)} = как в чекпойнте. У GPT-2 слои — Conv1D, вес "
                    "хранится как [in, out]; nn.Linear ждёт [out, in] → транспонируй.", pytrace=False)
    if w_o is not None and torch.equal(w_o, hf_sd["h.0.attn.c_proj.weight"]):
        pytest.fail("attn.c_proj.weight скопирован без транспонирования. Он квадратный [C, C], поэтому форма "
                    "совпала и load_state_dict не пожалуется, но матрица другая. Conv1D → .t().", pytrace=False)
    q = sd.get("layers.0.self_attn.q_proj.weight")
    if q is not None and q.shape == (cfg.d_model, cfg.d_model):
        want_q = orig.state_dict()["layers.0.self_attn.q_proj.weight"]
        if not torch.allclose(q, want_q):
            pytest.fail("q_proj не совпал с исходным. c_attn после транспонирования — [3C, C]: первые C строк — q, "
                        "следующие — k, потом v (split по dim=0). Bias [3C] режется так же.", pytrace=False)
    loaded, _ = _load_converted(impl, ref, hf_sd, TINY_GPT2)
    _same_logits(loaded, orig, cfg.vocab_size, "GPT-2 после конвертации считает не так, как исходная.")


# ---------------------------------------------------------------------------
# Настоящие модели transformers игрушечного размера (если transformers установлен)
# ---------------------------------------------------------------------------

def _hf_logits_match(impl, ref, hf_model, hf_cfg: dict):
    sd = {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
    loaded, cfg = _load_converted(impl, ref, sd, hf_cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 12), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = hf_model(idx).logits
        assert_close(loaded(idx), want, atol=1e-4, rtol=1e-4,
                     msg=lambda m: f"Логиты не совпали с {type(hf_model).__name__} из transformers.\n{m}")


def test_matches_transformers_qwen3(impl, ref):
    """С transformers: случайная Qwen3ForCausalLM → convert_state_dict → те же логиты, что у HF"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    torch.manual_seed(0)
    hf_cfg = transformers.Qwen3Config(**{k: v for k, v in TINY_QWEN3.items() if k != "model_type"})
    model = transformers.Qwen3ForCausalLM(hf_cfg).eval()
    _hf_logits_match(impl, ref, model, TINY_QWEN3)


def test_matches_transformers_gpt2(impl, ref):
    """С transformers: случайная GPT2LMHeadModel → convert_state_dict → те же логиты, что у HF"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    torch.manual_seed(0)
    hf_cfg = transformers.GPT2Config(**{k: v for k, v in TINY_GPT2.items() if k != "model_type"},
                                     attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0)
    model = transformers.GPT2LMHeadModel(hf_cfg).eval()
    _hf_logits_match(impl, ref, model, TINY_GPT2)
