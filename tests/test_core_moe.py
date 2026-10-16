"""llmlab/core/moe.py совпадает с эталонами задач C12–C15 (страховка от расхождения копий)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.testing import assert_close

from llmlab.core import moe as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "C" / "module-3-moe"


def _solution(task_id: str):
    spec = importlib.util.spec_from_file_location(f"sol_{task_id.replace('-', '_')}", MODULE / task_id / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_router_matches_c12():
    sol = _solution("C12-router")
    logits = torch.randn(20, 16)
    for args in [(2, "softmax", True, 1.0), (4, "softmax", False, 1.0), (8, "sigmoid", True, 2.5)]:
        for a, b in zip(core.route(logits, *args), sol.route(logits, *args)):
            assert_close(a, b)
    torch.manual_seed(0)
    a, b = core.TopKRouter(12, 6, 2), sol.TopKRouter(12, 6, 2)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 5, 12)
    for u, v in zip(a(x), b(x)):
        assert_close(u, v)


def test_moe_layer_matches_c13():
    sol = _solution("C13-moe-layer")
    torch.manual_seed(0)
    a, b = core.MoELayer(16, 24, 6, 2), sol.MoELayer(16, 24, 6, 2)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        for method in ("forward", "forward_naive"):
            for u, v in zip(getattr(a, method)(x), getattr(b, method)(x)):
                assert_close(u, v)


def test_balancing_matches_c14():
    sol = _solution("C14-load-balancing")
    logits = torch.randn(30, 8)
    assert_close(core.load_balancing_loss(logits, 2), sol.load_balancing_loss(logits, 2))
    idx = logits.topk(2, -1).indices
    assert_close(core.expert_load(idx, 8), sol.expert_load(idx, 8))
    scores, bias = logits.sigmoid(), torch.randn(8) * 0.1
    for u, v in zip(core.route_with_bias(scores, bias, 3), sol.route_with_bias(scores, bias, 3)):
        assert_close(u, v)
    load = core.expert_load(idx, 8)
    assert_close(core.update_bias(bias, load, 0.01), sol.update_bias(bias, load, 0.01))


def test_deepseek_moe_matches_c15():
    sol = _solution("C15-shared-expert")
    for args in [(61, 3), (27, 1), (6, 1, 2)]:
        assert core.mlp_layer_types(*args) == sol.mlp_layer_types(*args)
    args = (61, 7168, 18432, 2048, 256, 1, 8, 3)
    assert core.moe_param_counts(*args) == sol.moe_param_counts(*args)
    torch.manual_seed(0)
    a, b = core.DeepSeekMoE(16, 12, 6, 2, 2), sol.DeepSeekMoE(16, 12, 6, 2, 2)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        for u, v in zip(a(x), b(x)):
            assert_close(u, v)
    la = core.build_ffn_layers(4, 16, 40, 12, 6, 1, 2, 1)
    lb = sol.build_ffn_layers(4, 16, 40, 12, 6, 1, 2, 1)
    assert [type(m).__name__ for m in la] == [type(m).__name__ for m in lb]
