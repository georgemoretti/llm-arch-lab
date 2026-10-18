"""llmlab/core/autograd.py совпадает с эталоном задачи A01 (страховка от расхождения копий)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from llmlab.core import autograd as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "A" / "module-1-autograd"


def _solution(task_id: str):
    spec = importlib.util.spec_from_file_location(f"sol_{task_id.replace('-', '_')}", MODULE / task_id / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_value_matches_a01():
    sol = _solution("A01-value")

    def fn(V):
        x, y = V(0.7), V(-1.3)
        out = ((x * y).tanh() + (x**2).exp() / (y * y + 1) - (x + 2).log() + (y - x).relu()) * x
        out.backward()
        return out.data, x.grad, y.grad

    for a, b in zip(fn(core.Value), fn(sol.Value)):
        assert math.isclose(a, b, rel_tol=1e-12)


def test_core_source_is_a01_solution():
    """Код класса — один в один (кроме шапки модуля): правка эталона должна попасть в core."""
    core_src = Path(core.__file__).read_text(encoding="utf-8")
    sol_src = (MODULE / "A01-value" / "solution.py").read_text(encoding="utf-8")
    body = lambda s: s[s.index("from __future__"):]  # noqa: E731
    assert body(core_src) == body(sol_src)
