"""Конфигурируемая модель (llmlab/core/model.py): реестр, конфиги, совпадение с эталонами блока C, кеши."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from llmlab.core import efficient, gemma_olmo, gpt2, llama
from llmlab.core.model import (REGISTRY, Attention, Block, Model, ModelConfig, available_configs, build, generate,
                               layer_specs)

TOY = ["llama-like", "gemma3-like", "olmo2-like", "smollm3-like", "deepseek-tiny", "hybrid-tiny"]
REAL = {"gpt2": (124_439_808, 85_056_000), "qwen3-0.6b": (596_049_920, 440_467_456),
        "smollm2-135m": (134_515_008, 106_203_456)}


def _tiny_gpt2_cfg() -> ModelConfig:
    data = ModelConfig.named("gpt2").model_dump()
    data.update(vocab_size=97, d_model=32, n_layers=2)
    data["attention"]["n_heads"] = 4
    data["position"]["max_positions"] = 32
    data["ffn"]["d_hidden"] = 128
    return ModelConfig.model_validate(data)


def _toy(name: str) -> Model:
    torch.manual_seed(0)
    return Model.from_config(name).eval()


def test_all_configs_present():
    expected = {"gpt2", "llama-like", "gemma3-like", "qwen3-0.6b", "deepseek-tiny", "hybrid-tiny"}  # PLAN.md
    assert expected <= set(available_configs())


@pytest.mark.parametrize("name", sorted(REAL))
def test_real_configs_param_count(name):
    """Число параметров реальных моделей = как у оригинала (посчитано по весам HF)."""
    with torch.device("meta"):
        model = Model.from_config(name)
    total, non_embed = REAL[name]
    assert model.num_params() == total
    assert model.num_params(exclude_embeddings=True) == non_embed


@pytest.mark.parametrize("name", TOY + ["gpt2-tiny"])
def test_forward_cache_and_causality(name):
    model = Model(_tiny_gpt2_cfg()).eval() if name == "gpt2-tiny" else _toy(name)
    V = model.cfg.vocab_size
    idx = torch.randint(0, V, (2, 20))
    with torch.no_grad():
        logits = model(idx)
        assert logits.shape == (2, 20, V)
        assert torch.isfinite(logits).all()
        # будущее не влияет на прошлое
        idx2 = idx.clone()
        idx2[:, 12:] = (idx2[:, 12:] + 1) % V
        torch.testing.assert_close(model(idx2)[:, :12], logits[:, :12], atol=1e-5, rtol=1e-5)
        # prefill 7 + по токену == полный прогон (KV-cache, кольцевой кеш, латент, состояние DeltaNet)
        caches = model.new_caches()
        parts = [model(idx[:, :7], caches)] + [model(idx[:, t:t + 1], caches) for t in range(7, 20)]
        torch.testing.assert_close(torch.cat(parts, dim=1), logits, atol=1e-4, rtol=1e-4)
    out = generate(model, idx[:, :5], 6)
    torch.testing.assert_close(out, generate(model, idx[:, :5], 6, use_cache=False))


def test_layer_specs_schedules():
    g = layer_specs(ModelConfig.named("gemma3-like"))
    assert [s.window for s in g] == [8, 8, 8, 8, 8, None]
    assert [s.rope_theta for s in g] == [10000.0] * 5 + [1e6]
    s3 = layer_specs(ModelConfig.named("smollm3-like"))
    assert [s.use_rope for s in s3] == [True, True, True, False] * 2
    ds = layer_specs(ModelConfig.named("deepseek-tiny"))
    assert [s.ffn for s in ds] == ["dense", "moe", "moe", "moe"]
    hy = layer_specs(ModelConfig.named("hybrid-tiny"))
    assert [s.mixer for s in hy] == ["gated_deltanet"] * 3 + ["gqa"] + ["gated_deltanet"] * 3 + ["gqa"]
    model = Model.from_config("hybrid-tiny")
    assert hasattr(model.layers[0], "linear_attn") and hasattr(model.layers[3], "self_attn")


def test_config_errors_are_explicit():
    data = ModelConfig.named("llama-like").model_dump()
    data["attention"]["n_kv_head"] = 2                                   # опечатка
    with pytest.raises(ValidationError, match="n_kv_head"):
        ModelConfig.model_validate(data)
    data = ModelConfig.named("llama-like").model_dump()
    data["attention"]["n_kv_heads"] = 3
    with pytest.raises(ValidationError, match="не делится"):
        ModelConfig.model_validate(data)
    with pytest.raises(KeyError, match="Есть"):
        build("ffn", "relu", None)
    assert {"rmsnorm", "gemma_rmsnorm", "layernorm"} <= set(REGISTRY["norm"])


def test_moe_router_logits_collected():
    model = _toy("deepseek-tiny")
    model(torch.randint(0, 256, (2, 5)))
    logits = model.router_logits()
    assert len(logits) == 3 and logits[0].shape == (10, 8)


# ---------------------------------------------------------------------------- совпадение с эталонами


def _same(a: torch.nn.Module, b: torch.nn.Module, x: torch.Tensor, **kw):
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(a(x, **kw) if kw else a(x), b(x), atol=1e-5, rtol=1e-5)


def test_attention_matches_references():
    torch.manual_seed(0)
    x = torch.randn(2, 9, 32)
    _same(Attention(32, 4, 2, 8, rope_theta=500000.0), llama.LlamaAttention(32, 4, 2, 500000.0), x)
    for mode in ("head", "full"):
        _same(Attention(32, 4, 2, 8, qk_norm=mode), gemma_olmo.QKNormAttention(32, 4, 2, mode), x)
    _same(Attention(32, 4, 2, 8, window=3), gemma_olmo.SlidingWindowAttention(32, 4, 2, 3), x)
    _same(Attention(32, 4, 2, 8, use_rope=False), gemma_olmo.NoPEAttention(32, 4, 2, use_rope=False), x)
    _same(Attention(32, 4, 2, 16, qk_norm="head", partial_rotary_factor=0.25, output_gate=True),
          efficient.GatedAttention(32, 4, 2, 16), x)


def test_ring_cache_matches_reference():
    torch.manual_seed(0)
    ours, ref = Attention(32, 4, 2, 8, window=4), gemma_olmo.SlidingWindowCachedAttention(32, 4, 2, 4)
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(1, 10, 32)
    c1, c2 = ours.new_cache(), gemma_olmo.RingKVCache(4)
    with torch.no_grad():
        for t in range(10):
            torch.testing.assert_close(ours(x[:, t:t + 1], c1), ref(x[:, t:t + 1], c2), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("placement,ref_cls", [("pre", gemma_olmo.PreNormBlock), ("post", gemma_olmo.PostNormBlock),
                                               ("olmo", gemma_olmo.OlmoPostNormBlock),
                                               ("sandwich", gemma_olmo.SandwichNormBlock)])
def test_block_matches_c08(placement, ref_cls):
    cfg = ModelConfig.model_validate({
        "vocab_size": 8, "d_model": 32, "n_layers": 1, "norm": {"placement": placement, "eps": 1e-6},
        "attention": {"n_heads": 4, "n_kv_heads": 2}, "ffn": {"d_hidden": 48}})
    torch.manual_seed(0)
    ours = Block(cfg, layer_specs(cfg)[0])
    for p in ours.parameters():
        torch.nn.init.normal_(p, std=0.3)
    _same(ours, ref_cls(32, 4, 2, 48), torch.randn(2, 7, 32))


def test_gpt2_config_matches_core_gpt2():
    """Модель из gpt2.yaml (q/k/v раздельно) == llmlab/core/gpt2.py (общая c_attn) на тех же весах."""
    torch.manual_seed(0)
    ref = gpt2.GPT2(gpt2.GPT2Config.tiny()).eval()
    ours = Model(_tiny_gpt2_cfg()).eval()
    sd = {"embed_tokens.weight": ref.wte.weight, "embed_positions.weight": ref.wpe.weight,
          "norm.weight": ref.ln_f.weight, "norm.bias": ref.ln_f.bias, "lm_head.weight": ref.wte.weight}
    for i, blk in enumerate(ref.h):
        p = f"layers.{i}."
        for ours_name, mod in (("input_layernorm", blk.ln_1), ("post_attention_layernorm", blk.ln_2),
                               ("self_attn.o_proj", blk.attn.c_proj), ("mlp.c_fc", blk.mlp.c_fc),
                               ("mlp.c_proj", blk.mlp.c_proj)):
            sd[p + ours_name + ".weight"], sd[p + ours_name + ".bias"] = mod.weight, mod.bias
        for name, w, b in zip(("q_proj", "k_proj", "v_proj"), blk.attn.c_attn.weight.chunk(3),
                              blk.attn.c_attn.bias.chunk(3)):
            sd[p + f"self_attn.{name}.weight"], sd[p + f"self_attn.{name}.bias"] = w, b
    ours.load_state_dict(sd)
    idx = torch.randint(0, 97, (2, 16))
    with torch.no_grad():
        torch.testing.assert_close(ours(idx), ref(idx)[0], atol=1e-5, rtol=1e-5)


def test_tied_embeddings():
    model = Model.from_config("gemma3-like")
    assert model.lm_head.weight is model.embed_tokens.weight
    untied = Model.from_config("llama-like")
    assert untied.lm_head.weight is not untied.embed_tokens.weight
