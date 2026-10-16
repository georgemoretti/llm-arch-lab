"""llmlab/core/efficient.py совпадает с эталонами задач C18–C23 (страховка от расхождения копий)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.testing import assert_close

from llmlab.core import efficient as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "C" / "module-4-efficient"


def _solution(task_id: str):
    spec = importlib.util.spec_from_file_location(f"sol_{task_id.replace('-', '_')}", MODULE / task_id / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dsa_matches_c18():
    sol = _solution("C18-sparse-attention")
    torch.manual_seed(0)
    a, b = core.DSAttention(32, 4, 3, 8, 3), sol.DSAttention(32, 4, 3, 8, 3)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        oa, ka = a(x, return_mask=True)
        ob, kb = b(x, return_mask=True)
    assert_close(oa, ob)
    assert torch.equal(ka, kb)


def test_gated_attention_matches_c19():
    sol = _solution("C19-gated-attention")
    torch.manual_seed(0)
    a, b = core.GatedAttention(32, 4, 2, 16, 0.25), sol.GatedAttention(32, 4, 2, 16, 0.25)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        assert_close(a(x), b(x))


def test_linear_attention_matches_c20():
    sol = _solution("C20-linear-attention")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 2, 7, 4, generator=g) for _ in range(3))
    for name in ("linear_attention_noncausal", "linear_attention"):
        assert_close(getattr(core, name)(q, k, v), getattr(sol, name)(q, k, v))
    (oa, (sa, za)), (ob, (sb, zb)) = core.linear_attention_recurrent(q, k, v), sol.linear_attention_recurrent(q, k, v)
    assert_close(oa, ob)
    assert_close(sa, sb)
    assert_close(za, zb)


def test_gated_deltanet_matches_c21():
    sol = _solution("C21-gated-deltanet")
    torch.manual_seed(0)
    a, b = core.GatedDeltaNet(32, 2, 8, 6), sol.GatedDeltaNet(32, 2, 8, 6)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        (ya, sa), (yb, sb) = a(x), b(x)
    assert_close(ya, yb)
    assert_close(sa, sb)


def test_hybrid_matches_c22():
    sol = _solution("C22-hybrid-stack")
    cfg = (32, 4, 2, 48, 4, 2, 8, 2, 8, 8)
    torch.manual_seed(0)
    a, b = core.HybridStack(*cfg), sol.HybridStack(*cfg)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        assert_close(a(x), b(x))
    types = core.hybrid_layer_types(48, 4)
    assert types == sol.hybrid_layer_types(48, 4)
    assert core.hybrid_cache_bytes(types, 1000, 2, 256, 32, 128, 128) == sol.hybrid_cache_bytes(types, 1000, 2, 256, 32, 128, 128)


def test_ssd_matches_c23():
    sol = _solution("C23-mamba2-ssd")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 7, 3, 4, generator=g)
    dt = torch.rand(2, 7, 3, generator=g)
    A = -torch.rand(3, generator=g)
    B, C = torch.randn(2, 7, 5, generator=g), torch.randn(2, 7, 5, generator=g)
    D = torch.randn(3, generator=g)
    assert_close(core.ssd_quadratic(x, dt, A, B, C, D), sol.ssd_quadratic(x, dt, A, B, C, D))
    (ya, ha), (yb, hb) = core.ssd_recurrent(x, dt, A, B, C, D), sol.ssd_recurrent(x, dt, A, B, C, D)
    assert_close(ya, yb)
    assert_close(ha, hb)
