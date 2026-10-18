"""llmlab/core/bert.py совпадает с эталонами задач B07–B11 (страховка от расхождения копий)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from llmlab.core import bert as core

MODULE = Path(__file__).resolve().parents[1] / "content" / "blocks" / "B" / "module-2-bert"
NAMES = {
    "B07-wordpiece": ["is_punctuation", "clean_text", "strip_accents", "basic_tokenize", "wordpiece", "tokenize"],
    "B08-bert-input": ["truncate_pair", "build_inputs", "pad_batch", "BertConfig", "BertEmbeddings"],
    "B09-mlm-masking": ["mlm_mask"],
    "B10-bert-model": ["BertSelfAttention", "BertLayer", "BertModel", "BertForMaskedLM"],
    "B11-finetune-heads": ["SequenceClassifier", "align_labels", "qa_loss", "best_span"],
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
        assert sol_src[name] == core_src[name], f"{name}: llmlab/core/bert.py расходится с {task_id}/solution.py"


def test_bert_smoke():
    torch.manual_seed(0)
    cfg = core.BertConfig(vocab_size=30)
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "при", "##вет", "мир", ","])}
    ids = [vocab[t] for t in core.tokenize("Привет, мир", vocab)]
    batch = core.pad_batch([core.build_inputs(ids, [7], 2, 3, 16), core.build_inputs(ids[:1], None, 2, 3, 16)])
    inputs, labels = core.mlm_mask(batch["input_ids"], [0, 1, 2, 3, 4], 4, 30, prob=0.5,
                                   generator=torch.Generator().manual_seed(0))
    logits = core.BertForMaskedLM(cfg)(inputs, batch["token_type_ids"], batch["attention_mask"])
    assert logits.shape == (2, 8, 30)
    assert core.SequenceClassifier(cfg, 2)(inputs).shape == (2, 2)
