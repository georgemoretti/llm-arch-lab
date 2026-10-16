"""llmlab/core/gemma_olmo.py совпадает с эталонами задач C07–C11 (страховка от расхождения копий)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.testing import assert_close

from llmlab.core import gemma_olmo as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "C" / "module-2-gemma-olmo"


def _solution(task_id: str):
    spec = importlib.util.spec_from_file_location(f"sol_{task_id.replace('-', '_')}", MODULE / task_id / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same_module(a, b, *inputs):
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert_close(a.eval()(*inputs), b.eval()(*inputs))


def test_swa_matches_c07():
    sol = _solution("C07-swa")
    for T, S, W in [(6, 6, 3), (3, 10, 4), (8, 8, None), (5, 5, 1)]:
        assert torch.equal(core.sliding_window_mask(T, S, W), sol.sliding_window_mask(T, S, W))
    for n, p in [(62, 6), (42, 2), (5, 1)]:
        assert core.layer_types(n, p) == sol.layer_types(n, p)
    torch.manual_seed(0)
    for W in (3, None):
        _same_module(core.SlidingWindowAttention(32, 4, 2, W), sol.SlidingWindowAttention(32, 4, 2, W),
                     torch.randn(2, 10, 32))


def test_norm_blocks_match_c08():
    sol = _solution("C08-norm-placement")
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)
    for name in ("PostNormBlock", "PreNormBlock", "OlmoPostNormBlock", "SandwichNormBlock"):
        a = getattr(core, name)(32, 4, 2, 64)
        with torch.no_grad():
            for p in a.parameters():
                if p.dim() == 1:
                    p.uniform_(0.5, 1.5)
        _same_module(a, getattr(sol, name)(32, 4, 2, 64), x)


def test_qk_norm_matches_c09():
    sol = _solution("C09-qk-norm")
    torch.manual_seed(0)
    for mode in ("head", "full"):
        a = core.QKNormAttention(32, 4, 2, qk_norm=mode)
        with torch.no_grad():
            a.q_norm.weight.uniform_(0.5, 1.5)
            a.k_norm.weight.uniform_(0.5, 1.5)
        _same_module(a, sol.QKNormAttention(32, 4, 2, qk_norm=mode), torch.randn(2, 7, 32))


def test_nope_matches_c10():
    sol = _solution("C10-nope")
    for n, k in [(36, 4), (6, 100), (5, 1)]:
        assert core.rope_layers(n, k) == sol.rope_layers(n, k)
    torch.manual_seed(0)
    for use_rope in (True, False):
        _same_module(core.NoPEAttention(32, 4, 2, use_rope), sol.NoPEAttention(32, 4, 2, use_rope),
                     torch.randn(2, 7, 32))
    flags = [m.use_rope for m in sol.build_attention_layers(8, 32, 4, 2, 4)]
    assert flags == [m.use_rope for m in core.build_attention_layers(8, 32, 4, 2, 4)]


def test_ring_cache_matches_c11():
    sol = _solution("C11-swa-kv-cache")
    torch.manual_seed(0)
    a = core.SlidingWindowCachedAttention(32, 4, 2, 4)
    b = sol.SlidingWindowCachedAttention(32, 4, 2, 4)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 14, 32)
    ca, cb = core.RingKVCache(4), sol.RingKVCache(4)
    with torch.no_grad():
        for s, e in [(0, 5), (5, 6), (6, 12), (12, 14)]:
            assert_close(a(x[:, s:e], ca), b(x[:, s:e], cb))
    assert torch.equal(ca.pos, cb.pos)
    lt = core.layer_types(62, 6)
    assert core.swa_kv_cache_bytes(lt, 1024, 16, 128, 131072) == sol.swa_kv_cache_bytes(lt, 1024, 16, 128, 131072)
