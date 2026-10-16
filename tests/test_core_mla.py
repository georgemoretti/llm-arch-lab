"""llmlab/core/mla.py совпадает с эталонами задач C16, C16b, C17 (страховка от расхождения копий)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.testing import assert_close

from llmlab.core import mla as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "C" / "module-3-moe"


def _solution(task_id: str):
    spec = importlib.util.spec_from_file_location(f"sol_{task_id.replace('-', '_')}", MODULE / task_id / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chunks(m, x, cache_cls, method="forward"):
    cache, outs, s = cache_cls(), [], 0
    with torch.no_grad():
        for t in (3, 1, 4):
            outs.append(getattr(m, method)(x[:, s:s + t], cache))
            s += t
    return torch.cat(outs, 1), cache


def test_mla_matches_c16():
    sol = _solution("C16-mla")
    torch.manual_seed(0)
    a, b = core.MLAttention(32, 4, 8, 8, 6), sol.MLAttention(32, 4, 8, 8, 6)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 8, 32)
    for method in ("forward", "forward_absorbed"):
        oa, ca = _chunks(a, x, core.LatentCache, method)
        ob, cb = _chunks(b, x, sol.LatentCache, method)
        assert_close(oa, ob)
        assert_close(ca.c, cb.c)


def test_mla_rope_matches_c16b():
    sol = _solution("C16b-mla-rope")
    torch.manual_seed(0)
    a, b = core.MLARopeAttention(32, 4, 8, 6, 4, 6), sol.MLARopeAttention(32, 4, 8, 6, 4, 6)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 8, 32)
    oa, ca = _chunks(a, x, core.LatentCache)
    ob, cb = _chunks(b, x, core.LatentCache)
    assert_close(oa, ob)
    assert_close(ca.k_rope, cb.k_rope)
    assert core.mla_kv_cache_bytes(61, 512, 64, 100) == sol.mla_kv_cache_bytes(61, 512, 64, 100)


def test_mtp_matches_c17():
    sol = _solution("C17-mtp")
    torch.manual_seed(0)
    embed, head = nn.Embedding(30, 32), nn.Linear(32, 30, bias=False)
    ma = nn.ModuleList(core.MTPModule(32, 4, 2, 48) for _ in range(2))
    mb = nn.ModuleList(sol.MTPModule(32, 4, 2, 48) for _ in range(2))
    mb.load_state_dict(ma.state_dict())
    tokens, h = torch.randint(0, 30, (2, 10)), torch.randn(2, 10, 32)
    with torch.no_grad():
        la = core.mtp_logits(h, tokens, embed, head, ma)
        lb = sol.mtp_logits(h, tokens, embed, head, mb)
        for u, v in zip(la, lb):
            assert_close(u, v)
        assert_close(core.mtp_loss(head(h), la, tokens, 0.3)[0], sol.mtp_loss(head(h), lb, tokens, 0.3)[0])
