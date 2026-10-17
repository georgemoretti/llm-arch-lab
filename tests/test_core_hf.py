"""Загрузка весов HF (llmlab/core/hf.py): совпадение с эталонами C24/C25, transformers, флаг --weights."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from llmlab.core import hf
from llmlab.core.model import Model, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "content" / "blocks" / "C" / "module-5-final"
TINY_QWEN3 = {
    "model_type": "qwen3", "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 48, "hidden_act": "silu", "rms_norm_eps": 1e-6,
    "rope_theta": 1e6, "tie_word_embeddings": True,
}


def _load(task: str):
    spec = importlib.util.spec_from_file_location(f"sol_{task.replace('-', '_')}", FINAL / task / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_core_matches_task_solutions():
    """core/hf.py дублирует solution.py задач C24 и C25 — функции должны вести себя одинаково."""
    c24, c25 = _load("C24-hf-checkpoint"), _load("C25-qwen3-project")
    for cfg in (TINY_QWEN3, {**TINY_QWEN3, "model_type": "llama", "tie_word_embeddings": False},
                {"model_type": "gpt2", "vocab_size": 64, "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 16}):
        assert c24.config_from_hf(cfg) == hf.config_from_hf(cfg)
    torch.manual_seed(0)
    model = Model(hf.config_from_hf(TINY_QWEN3)).eval()
    sd = {("" if k == "lm_head.weight" else "model.") + k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    a, b = c24.convert_state_dict(sd, model.cfg, "qwen3"), hf.convert_state_dict(sd, model.cfg, "qwen3")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    logits = torch.randn(6, 64, generator=torch.Generator().manual_seed(1))
    for kw in ({"temperature": 0.0}, {"temperature": 0.8, "top_k": 5}, {"temperature": 1.0, "top_p": 0.6}):
        assert torch.equal(c25.sample_next(logits, generator=torch.Generator().manual_seed(2), **kw),
                           hf.sample_next(logits, generator=torch.Generator().manual_seed(2), **kw))
    idx = torch.randint(0, 64, (2, 4))
    assert torch.equal(c25.generate(model, idx, 5, eos_token_id=3), hf.generate(model, idx, 5, eos_token_id=3))


def test_named_configs_roundtrip_through_hf_format():
    """qwen3-0.6b.yaml == config_from_hf(официальный config.json), включая формат transformers 5.x."""
    official = {"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 1024, "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
                "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 1000000, "rope_scaling": None,
                "attention_bias": False, "use_sliding_window": False, "tie_word_embeddings": True}
    want = ModelConfig.named("qwen3-0.6b").model_dump(exclude={"name", "source"})
    assert hf.config_from_hf(official).model_dump(exclude={"name", "source"}) == want
    v5 = {k: v for k, v in official.items() if k not in ("rope_theta", "rope_scaling")}
    v5["rope_parameters"] = {"rope_theta": 1e6, "rope_type": "default"}
    assert hf.config_from_hf(v5).model_dump(exclude={"name", "source"}) == want


@pytest.mark.parametrize("arch", ["qwen3", "llama", "gpt2"])
def test_save_pretrained_roundtrip(tmp_path, arch):
    """Случайная модель transformers → save_pretrained → load_pretrained: те же логиты (без скачивания)."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    if arch == "gpt2":
        cfg = transformers.GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=32,
                                      attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0)
        ref = transformers.GPT2LMHeadModel(cfg)
    else:
        kw = {k: v for k, v in TINY_QWEN3.items() if k != "model_type"}
        if arch == "qwen3":
            ref = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**kw))
        else:
            kw.pop("head_dim")
            kw["tie_word_embeddings"] = False
            ref = transformers.LlamaForCausalLM(transformers.LlamaConfig(**kw))
    ref.eval().save_pretrained(tmp_path, max_shard_size="20KB")       # несколько шардов + index.json
    assert (tmp_path / "model.safetensors.index.json").exists()
    model = hf.load_pretrained(tmp_path)
    idx = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(model(idx), ref(idx).logits, atol=1e-4, rtol=1e-4)
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["model_type"] == arch


def test_weights_marker_skipped_without_flag(tmp_path):
    """@pytest.mark.weights пропускается без --weights и запускается с ним."""
    (tmp_path / "test_w.py").write_text("import pytest\n\n@pytest.mark.weights\ndef test_w():\n    assert False\n", encoding="utf-8")
    base = [sys.executable, "-m", "pytest", str(tmp_path / "test_w.py"), "-p", "llmlab.pytest_plugin", "-q",
            "-p", "no:cacheprovider", f"--rootdir={tmp_path}"]
    env = {k: v for k, v in os.environ.items() if k != "LLMLAB_WEIGHTS"}   # иначе флаг придёт из окружения
    skipped = subprocess.run(base, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert skipped.returncode == 0 and "1 skipped" in skipped.stdout, skipped.stdout
    run = subprocess.run(base + ["--weights"], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert run.returncode == 1 and "1 failed" in run.stdout, run.stdout


@pytest.mark.weights
def test_real_qwen3_matches_transformers():
    """Qwen3-0.6B: логиты == transformers (fp32), 596 049 920 параметров."""
    transformers = pytest.importorskip("transformers")
    path = hf.download("Qwen/Qwen3-0.6B")
    model, tok = hf.load_pretrained(path), hf.load_tokenizer(path)
    assert model.num_params() == 596_049_920
    ref = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                            attn_implementation="eager").eval()
    idx = torch.tensor([tok.encode("The capital of France is Paris. The capital of Germany is").ids])
    with torch.no_grad():
        torch.testing.assert_close(model(idx), ref(idx).logits, atol=1e-3, rtol=1e-3)
