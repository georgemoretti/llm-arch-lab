"""llmlab/core/encdec.py совпадает с эталонами задач B01–B05 (страховка от расхождения копий)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from llmlab.core import encdec as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "B" / "module-1-encdec"
NAMES = {
    "B01-padding-mask": ["padding_mask", "attention", "MultiHeadAttention"],
    "B02-sinusoidal": ["sinusoidal_positions", "shift_matrix", "TransformerEmbedding"],
    "B03-encoder": ["FeedForward", "EncoderLayer", "Encoder"],
    "B04-decoder": ["causal_mask", "target_mask", "DecoderLayer", "Decoder"],
    "B05-seq2seq": ["shift_right", "label_smoothed_ce", "Seq2SeqConfig", "Seq2Seq", "greedy_decode"],
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
        assert sol_src[name] == core_src[name], f"{name}: llmlab/core/encdec.py расходится с {task_id}/solution.py"


def test_seq2seq_smoke():
    torch.manual_seed(0)
    model = core.Seq2Seq(core.Seq2SeqConfig(vocab_size=12))
    src = torch.tensor([[5, 6, 7, 0], [3, 4, 5, 6]])
    tgt = torch.tensor([[8, 9, 2, 0], [8, 2, 0, 0]])
    logits = model(src, core.shift_right(tgt, 1))
    assert logits.shape == (2, 4, 12)
    core.label_smoothed_ce(logits, tgt, 0.1, 0).backward()
    assert model.encoder.embed.tok.weight is model.lm_head.weight
    assert core.greedy_decode(model.eval(), src, 5).shape[0] == 2
