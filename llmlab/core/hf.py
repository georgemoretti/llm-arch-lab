"""Загрузка реальных весов с Hugging Face в конфигурируемую модель (Фаза 8) — эталоны задач C24, C25a и C25.

C24: ``config_from_hf`` (config.json → ModelConfig) и ``convert_state_dict`` (имена и формы тензоров
чекпойнта → имена ``llmlab.core.model.Model``). C25a: ``sample_next``. C25: ``load_safetensors``,
``load_pretrained``, ``generate``. Утилиты ``download`` и ``load_tokenizer`` нужны CLI (``llmlab generate``) и
тестам с ``@pytest.mark.weights``.

⚠️ Это готовые ответы на задачи C24, C25a и C25. Если ещё не решил их — не подглядывай.

Код намеренно совпадает с ``solution.py`` задач C24/C25a/C25 (``content/blocks/C/module-5-final/``);
расхождение ловит ``tests/test_core_hf.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from llmlab.core.model import Model, ModelConfig

# ---------------------------------------------------------------------------- C24 config.json


def _reject(hf: dict, key: str, ok, why: str) -> None:
    value = hf.get(key)
    if not ok(value):
        raise ValueError(f"{key} = {value!r} не поддерживается: {why}")


def config_from_hf(hf: dict) -> ModelConfig:
    """config.json модели Hugging Face (dict) → ModelConfig. Неподдерживаемые опции → ValueError."""
    model_type = hf.get("model_type")
    if model_type == "gpt2":
        _reject(hf, "activation_function", lambda v: v in (None, "gelu_new"), "в курсе только GELU с tanh")
        d = hf["n_embd"]
        return ModelConfig.model_validate({
            "name": hf.get("_name_or_path") or "gpt2",
            "vocab_size": hf["vocab_size"], "d_model": d, "n_layers": hf["n_layer"],
            "tie_embeddings": hf.get("tie_word_embeddings", True),
            "norm": {"type": "layernorm", "eps": hf.get("layer_norm_epsilon", 1e-5), "placement": "pre"},
            "position": {"type": "learned", "max_positions": hf["n_positions"]},
            "attention": {"type": "gqa", "n_heads": hf["n_head"], "bias": True},
            "ffn": {"type": "gelu", "d_hidden": hf.get("n_inner") or 4 * d, "bias": True},
        })
    if model_type not in ("qwen3", "llama"):
        raise ValueError(f"model_type = {model_type!r} не поддерживается: есть qwen3, llama, gpt2")
    _reject(hf, "hidden_act", lambda v: v == "silu", "FFN курса для Llama/Qwen3 — SwiGLU (silu)")
    _reject(hf, "rope_scaling", lambda v: v is None, "масштабирование RoPE (YaRN, llama3) в курсе не реализовано")
    # transformers ≥ 5 пишет RoPE иначе: "rope_parameters": {"rope_theta": …, "rope_type": "default" | "yarn" | …}
    rope = hf.get("rope_parameters") or {}
    _reject(rope, "rope_type", lambda v: v in (None, "default"), "масштабирование RoPE в курсе не реализовано")
    theta = rope.get("rope_theta", hf.get("rope_theta", 10000.0))
    _reject(hf, "attention_bias", lambda v: not v, "bias у q/k/v/o в Llama/Qwen3 не используется")
    _reject(hf, "mlp_bias", lambda v: not v, "bias в SwiGLU не используется")
    _reject(hf, "use_sliding_window", lambda v: not v, "у Qwen3 dense окно выключено; SWA — см. gemma3-like")
    H = hf["num_attention_heads"]
    return ModelConfig.model_validate({
        "name": hf.get("_name_or_path") or model_type,
        "vocab_size": hf["vocab_size"], "d_model": hf["hidden_size"], "n_layers": hf["num_hidden_layers"],
        "tie_embeddings": hf.get("tie_word_embeddings", False),
        "norm": {"type": "rmsnorm", "eps": hf["rms_norm_eps"], "placement": "pre"},
        "position": {"type": "rope", "theta": theta},
        "attention": {
            "type": "gqa", "n_heads": H, "n_kv_heads": hf.get("num_key_value_heads") or H,
            "head_dim": hf.get("head_dim") or hf["hidden_size"] // H,
            "qk_norm": "head" if model_type == "qwen3" else None,
        },
        "ffn": {"type": "swiglu", "d_hidden": hf["intermediate_size"]},
    })


# ---------------------------------------------------------------------------- C24 веса

_GPT2_RENAMES = {"ln_1": "input_layernorm", "ln_2": "post_attention_layernorm", "attn.c_proj": "self_attn.o_proj",
                 "mlp.c_fc": "mlp.c_fc", "mlp.c_proj": "mlp.c_proj"}


def _convert_gpt2(hf_sd: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    out = {}
    C = cfg.d_model
    for name, t in hf_sd.items():
        name = name.removeprefix("transformer.")
        if name.endswith((".attn.bias", ".attn.masked_bias")):     # буферы causal-маски, не веса
            continue
        if name == "lm_head.weight":
            continue                                                # в GPT-2 выход связан с wte
        if name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
            out[{"wte": "embed_tokens", "wpe": "embed_positions", "ln_f": "norm"}[name.split(".")[0]]
                + "." + name.split(".")[1]] = t
            continue
        _, i, rest = name.split(".", 2)                             # h.{i}.<rest>
        prefix = f"layers.{i}."
        module, kind = rest.rsplit(".", 1)                          # "attn.c_attn", "weight"
        if module == "attn.c_attn":
            # Conv1D хранит вес как [in, out] → транспонируем в [out, in] и режем на q, k, v
            parts = (t.t() if kind == "weight" else t).split(C, dim=0)
            for proj, p in zip(("q_proj", "k_proj", "v_proj"), parts):
                out[f"{prefix}self_attn.{proj}.{kind}"] = p.contiguous()
            continue
        if kind == "weight" and module in ("attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            t = t.t().contiguous()                                  # Conv1D → Linear
        out[prefix + _GPT2_RENAMES[module] + "." + kind] = t
    out["lm_head.weight"] = out["embed_tokens.weight"]
    return out


def _convert_llama(hf_sd: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    out = {}
    for name, t in hf_sd.items():
        if name.endswith("rotary_emb.inv_freq"):                   # старые чекпойнты хранят частоты RoPE
            continue
        out[name.removeprefix("model.")] = t
    if cfg.tie_embeddings:
        head = out.pop("lm_head.weight", None)
        if head is not None and not torch.equal(head, out["embed_tokens.weight"]):
            raise ValueError("tie_word_embeddings = true, но lm_head.weight в чекпойнте отличается от embed_tokens")
        out["lm_head.weight"] = out["embed_tokens.weight"]
    return out


def convert_state_dict(hf_sd: dict[str, torch.Tensor], cfg: ModelConfig, model_type: str,
                       dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Тензоры чекпойнта HF → state_dict ``Model(cfg)``: имена, Conv1D → Linear, связанный lm_head, dtype."""
    convert = _convert_gpt2 if model_type == "gpt2" else _convert_llama
    return {k: v.to(dtype) for k, v in convert(hf_sd, cfg).items()}


def check_state_dict(model: torch.nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """Понятная ошибка вместо длинного отчёта load_state_dict: чего не хватает, что лишнее, где не те формы."""
    expected = model.state_dict()
    missing = sorted(set(expected) - set(sd))
    extra = sorted(set(sd) - set(expected))
    shapes = [f"{k}: в чекпойнте {tuple(sd[k].shape)}, в модели {tuple(expected[k].shape)}"
              for k in expected if k in sd and sd[k].shape != expected[k].shape]
    problems = []
    if missing:
        problems.append(f"нет {len(missing)} тензоров, например {missing[:3]}")
    if extra:
        problems.append(f"лишние {len(extra)} тензоров, например {extra[:3]}")
    if shapes:
        problems.append("не совпадают формы: " + "; ".join(shapes[:3]))
    if problems:
        raise ValueError("state_dict не подходит к модели: " + " | ".join(problems))


# ---------------------------------------------------------------------------- C25 загрузка


def load_safetensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Все тензоры из model.safetensors или из шардов по model.safetensors.index.json."""
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        sd.update(load_file(model_dir / shard))
    return sd


def load_pretrained(model_dir: str | Path, dtype: torch.dtype = torch.float32) -> Model:
    """Папка с config.json и *.safetensors → Model с этими весами (eval-режим)."""
    model_dir = Path(model_dir)
    hf = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = config_from_hf(hf)
    with torch.device("meta"):                                      # не тратим время на случайную инициализацию
        model = Model(cfg)
    sd = convert_state_dict(load_safetensors(model_dir), cfg, hf["model_type"], dtype)
    check_state_dict(model, sd)
    model.load_state_dict(sd, assign=True)
    if cfg.tie_embeddings:
        model.lm_head.weight = model.embed_tokens.weight            # assign=True разрывает связь — восстанавливаем
    return model.eval()


# ---------------------------------------------------------------------------- C25a сэмплирование, C25 генерация


def sample_next(logits: torch.Tensor, temperature: float = 0.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] → следующий токен [B, 1]. temperature = 0 — argmax; иначе softmax(logits / T),
    обрезанный top-k и top-p (nucleus: наименьший набор токенов с суммарной вероятностью ≥ p)."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k is not None:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        before = probs.cumsum(dim=-1) - probs                       # масса токенов СТРОГО выше текущего
        sorted_logits = sorted_logits.masked_fill(before >= top_p, float("-inf"))   # первый токен остаётся всегда
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator)


@torch.no_grad()
def generate(model: Model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
             top_k: int | None = None, top_p: float | None = None, eos_token_id: int | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """idx [B, T] → [B, T + n], n ≤ max_new_tokens. С KV-cache. После eos строка дополняется eos;
    генерация останавливается, когда eos выдали все строки."""
    caches = model.new_caches()
    logits = model(idx, caches)
    done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    for step in range(max_new_tokens):
        nxt = sample_next(logits[:, -1], temperature, top_k, top_p, generator)        # [B, 1]
        if eos_token_id is not None:
            nxt = nxt.masked_fill(done[:, None], eos_token_id)                    # закончившие строки — eos
            done = done | (nxt[:, 0] == eos_token_id)
        idx = torch.cat((idx, nxt), dim=1)
        if bool(done.all()) or step + 1 == max_new_tokens:
            break                                                                 # лишний прогон не нужен
        logits = model(nxt, caches)                                               # с кешем — только новый токен
    return idx


# ---------------------------------------------------------------------------- утилиты (не задачи)


def download(repo_id: str) -> Path:
    """Скачать config.json, токенизатор и *.safetensors в кеш HF (~/.cache/huggingface) → путь к папке."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Нужен huggingface_hub: uv sync --extra weights (или pip install -e '.[weights]')") from exc
    return Path(snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


def load_tokenizer(model_dir: str | Path):
    """tokenizer.json → tokenizers.Tokenizer (без transformers)."""
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Нужен tokenizers: uv sync --extra weights (или pip install -e '.[weights]')") from exc
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
