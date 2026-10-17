"""Эталон: config.json и веса чекпойнта Hugging Face → ModelConfig и state_dict своей модели."""

import torch

from llmlab.core.model import ModelConfig   # конфиг модели из YAML (Фаза 8)


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
