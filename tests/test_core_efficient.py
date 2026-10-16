"""llmlab/core/efficient.py совпадает с эталонами задач C18–C21 (страховка от расхождения копий)."""

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
