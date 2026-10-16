"""llmlab/core/llama.py совпадает с эталонами задач C01–C05 (страховка от расхождения копий)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.testing import assert_close

from llmlab.core import llama as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "C" / "module-1-llama"


def _solution(task_id: str):
    spec = importlib.util.spec_from_file_location(f"sol_{task_id.replace('-', '_')}", MODULE / task_id / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rmsnorm_matches_c01():
    sol = _solution("C01-rmsnorm")
    x, w = torch.randn(2, 5, 16), torch.randn(16)
    assert_close(core.rms_norm(x, w), sol.rms_norm(x, w))
    assert_close(core.rms_norm(x.half(), w.half()), sol.rms_norm(x.half(), w.half()))


def test_swiglu_matches_c02():
    sol = _solution("C02-swiglu")
    for args in [(4096, 256, None), (4096, 1024, 1.3), (64, 32, None)]:
        assert core.llama_ffn_hidden_dim(*args) == sol.llama_ffn_hidden_dim(*args)
    torch.manual_seed(0)
    a = core.SwiGLUFFN(16, 40)
    b = sol.SwiGLUFFN(16, 40)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 16)
    assert_close(a(x), b(x))


def test_rope_matches_c03():
    sol = _solution("C03-rope")
    pos = torch.arange(5, 17)
    for theta in (10000.0, 500000.0):
        for c, s in zip(core.rope_cos_sin(pos, 16, theta), sol.rope_cos_sin(pos, 16, theta)):
            assert_close(c, s)
    x = torch.randn(2, 3, 12, 16)
    cos, sin = sol.rope_cos_sin(pos, 16)
    assert_close(core.apply_rope(x, cos, sin), sol.apply_rope(x, cos, sin))


def test_gqa_parts_match_c04():
    sol = _solution("C04-gqa")
    x = torch.randn(2, 3, 5, 4)
    for n_rep in (1, 2, 4):
        assert_close(core.repeat_kv(x, n_rep), sol.repeat_kv(x, n_rep))
    assert core.kv_cache_bytes(32, 8, 128, 8192) == sol.kv_cache_bytes(32, 8, 128, 8192)


def test_attention_matches_c05():
    sol = _solution("C05-kv-cache")
    torch.manual_seed(0)
    a = core.LlamaAttention(32, 4, 2, rope_theta=10000.0)
    b = sol.LlamaAttention(32, 4, 2, rope_theta=10000.0)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 9, 32)
    ca, cb = core.KVCache(), sol.KVCache()
    for sl in (slice(0, 4), slice(4, 5), slice(5, 9)):
        assert_close(a(x[:, sl], ca), b(x[:, sl], cb))
    assert_close(ca.k, cb.k)
    assert_close(a(x), b(x))
