"""llmlab/core/modernbert.py совпадает с эталонами задач B16–B19 (страховка от расхождения копий)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from llmlab.core import modernbert as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "B" / "module-4-modernbert"
B16 = ["layer_types", "local_window_mask", "ModernBertEmbeddings", "ModernBertMLP", "ModernBertAttention",
       "ModernBertEncoderLayer", "ModernBertModel", "ModernBertForMaskedLM"]
B17 = ["unpad_input", "pack", "unpack", "packed_position_ids", "varlen_mask", "varlen_attention", "padding_savings"]
B18 = ["mean_pool", "cls_pool", "l2_normalize", "cosine_matrix", "_hidden", "BiEncoder", "CrossEncoder"]
NAMES = {
    "B16-modernbert-layer": B16,
    "B17-unpadding": B17,
    "B17d-cu-seqlens-debug": B17,
    "B18-sentence-embeddings": B18,
    "B18d-mean-pool-debug": B18,
    "B19-infonce": ["info_nce", "symmetric_info_nce", "recall_at_k"],
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
        assert sol_src[name] == core_src[name], f"{name}: llmlab/core/modernbert.py расходится с {task_id}/solution.py"


def test_unpadded_model_smoke():
    torch.manual_seed(0)
    model = core.BiEncoder(core.ModernBertModel(core.ModernBertConfig(vocab_size=50, local_attention=4)))
    ids = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]])
    emb = model(ids, (ids != 0).long())
    assert emb.shape == (2, 32) and torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)
    idx, cu, _ = core.unpad_input((ids != 0).long())
    assert cu.tolist() == [0, 3, 8] and core.varlen_mask(cu).sum() == 9 + 25
    loss = core.info_nce(emb, emb.flip(0), 0.1)
    assert torch.isfinite(loss)
