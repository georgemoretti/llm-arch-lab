"""llmlab/core/bpe.py совпадает с эталонами задач A11–A13 (страховка от расхождения копий)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from llmlab.core import bpe as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "A" / "module-3-tokenization"
FUNCS = {
    "A11-bpe-train": ["count_pairs", "merge_pair", "train_bpe", "build_vocab"],
    "A12-bpe-encode": ["encode", "decode"],
    "A13-pretokenize": ["pretokenize", "bytes_to_unicode", "train_regex_bpe", "split_special",
                        "encode_with_special", "decode_with_special"],
}


def _sources(path: Path) -> dict[str, str]:
    """Исходник каждой функции верхнего уровня и строка присваивания GPT2_PATTERN."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.get_source_segment(src, node)
    return out


@pytest.mark.parametrize("task_id", sorted(FUNCS))
def test_core_functions_are_task_solutions(task_id):
    """Код функций — один в один: правка эталона задачи должна попасть в core (и наоборот)."""
    core_src = _sources(Path(core.__file__))
    sol_src = _sources(MODULE / task_id / "solution.py")
    for name in FUNCS[task_id] + (["GPT2_PATTERN"] if task_id == "A13-pretokenize" else []):
        assert sol_src[name] == core_src[name], f"{name}: llmlab/core/bpe.py расходится с {task_id}/solution.py"


def test_core_roundtrip_smoke():
    merges = core.train_regex_bpe("мама мыла раму, папа мыл пол " * 5, 280)
    vocab = core.build_vocab(merges)
    special = {"<|endoftext|>": 280}
    s = "мама<|endoftext|>мыла 🙂"
    assert core.decode_with_special(core.encode_with_special(s, merges, special), vocab, special) == s
