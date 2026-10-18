"""llmlab/core/t5.py совпадает с эталонами задач B13–B15 (страховка от расхождения копий)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from llmlab.core import t5 as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "B" / "module-3-distil-t5"
NAMES = {
    "B13-t5-relative-bias": ["relative_position_bucket", "relative_position_bias"],
    "B13d-t5-bias-debug": ["relative_position_bucket", "relative_position_bias"],
    "B14-t5-layer": ["T5LayerNorm", "T5Attention", "T5Block", "T5Stack", "T5Model"],
    "B15-span-corruption": ["random_segmentation", "random_spans_noise_mask", "noise_span_to_unique_sentinel",
                            "span_corruption"],
    "B15d-sentinel-debug": ["random_segmentation", "random_spans_noise_mask", "noise_span_to_unique_sentinel",
                            "span_corruption"],
}


def _sources(path: Path) -> dict[str, str]:
    src = path.read_text(encoding="utf-8")
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


@pytest.mark.parametrize("task_id", sorted(NAMES))
def test_core_is_task_solution(task_id):
    """Код функций и классов — один в один: правка эталона задачи должна попасть в core (и наоборот)."""
    core_src, sol_src = _sources(Path(core.__file__)), _sources(MODULE / task_id / "solution.py")
    for name in NAMES[task_id]:
        assert sol_src[name] == core_src[name], f"{name}: llmlab/core/t5.py расходится с {task_id}/solution.py"


def test_t5_smoke():
    torch.manual_seed(0)
    cfg = core.T5Config(vocab_size=120)
    ids = torch.arange(10, 40)
    mask = core.random_spans_noise_mask(30, generator=torch.Generator().manual_seed(0))
    inputs, targets = core.span_corruption(ids, mask, [119 - i for i in range(10)], eos_id=1)
    out = core.T5Model(cfg)(inputs[None], torch.cat([torch.tensor([0]), targets[:-1]])[None])
    assert out.shape == (1, targets.numel(), cfg.d_model)
