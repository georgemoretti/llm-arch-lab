"""llmlab/core/minigpt.py совпадает с эталонами задач A15–A17 (страховка от расхождения копий)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from llmlab.core import minigpt as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "A" / "module-4-gpt"
NAMES = {
    "A15-embeddings": ["TokenPositionEmbedding", "tie_weights", "count_params", "EmbeddingLM"],
    "A16-attention": ["causal_mask", "attention", "AttentionHead"],
    "A17-block": ["MLP", "Block"],
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
        assert sol_src[name] == core_src[name], f"{name}: llmlab/core/minigpt.py расходится с {task_id}/solution.py"


def test_block_stack_smoke():
    torch.manual_seed(0)
    emb = core.TokenPositionEmbedding(20, 8, 16)
    blocks = [core.Block(16, 8, pre_norm=p) for p in (True, False)]
    x = emb(torch.randint(0, 20, (2, 8)))
    for b in blocks:
        x = b(x)
    assert x.shape == (2, 8, 16) and torch.isfinite(x).all()
