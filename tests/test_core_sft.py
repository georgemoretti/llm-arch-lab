"""llmlab/core/sft.py совпадает с эталонами задач D01–D03 (страховка от расхождения копий) + генератор данных."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from llmlab.core import sft as core

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "content" / "blocks" / "D" / "module-1-sft"
D01 = ["render_chat", "split_special", "encode_chat"]
D02 = ["assistant_mask", "sft_targets", "pack", "packed_causal_mask"]
D03 = ["masked_ce_sum", "accumulate_gradients", "sft_step"]
NAMES = {
    "D01-chat-template": D01,
    "D02-loss-mask": D02,
    "D02d-shift-debug": D02,
    "D03-sft-step": D03,
    "D03d-grad-accum-debug": D03,
}


def _sources(path: Path) -> dict[str, str]:
    src = path.read_text(encoding="utf-8")
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


@pytest.mark.parametrize("task_id", sorted(NAMES))
def test_core_is_task_solution(task_id):
    """Код функций — один в один: правка эталона задачи должна попасть в core (и наоборот)."""
    core_src, sol_src = _sources(Path(core.__file__)), _sources(MODULE / task_id / "solution.py")
    for name in NAMES[task_id]:
        assert sol_src[name] == core_src[name], f"{name}: llmlab/core/sft.py расходится с {task_id}/solution.py"


def test_lr_at_is_a08():
    a08 = _sources(ROOT / "content" / "blocks" / "A" / "module-2-training" / "A08-lr-schedule" / "solution.py")
    assert _sources(Path(core.__file__))["lr_at"] == a08["lr_at"], "lr_at в core/sft.py расходится с эталоном A08"


def test_instructions_split_and_format():
    train = core.make_instructions(600, seed=0)
    test = core.make_instructions(300, seed=1, split="test")
    assert [e["task"] for e in train[:3]] == list(core.TASKS)
    tr_keys = {e["prompt"].split(":")[-1].strip() for e in train if e["task"] == "upper"}
    te_keys = {e["prompt"].split(":")[-1].strip() for e in test if e["task"] == "upper"}
    assert not tr_keys & te_keys, "отложенные слова не должны встречаться в train"
    for e in train + test:
        if e["task"] == "json":
            obj = json.loads(e["answer"])
            assert set(obj) == {"name", "age"} and isinstance(obj["age"], int)
        elif e["task"] == "arith":
            assert 0 <= int(e["answer"]) <= 20
        else:
            assert e["answer"] == e["answer"].upper()
    tok = core.CharTokenizer()
    for e in train[:30]:
        ids = core.encode_chat(core.render_chat(core.to_messages(e)), tok)
        assert tok.decode(ids) == core.render_chat(core.to_messages(e))
        tok.encode(core.plain_text(e))                                  # все символы — в алфавите


def test_sft_pipeline_smoke():
    tok = core.CharTokenizer()
    torch.manual_seed(0)
    from llmlab.core.model import Model
    model = Model(core.toy_config(tok.vocab_size, d_model=32, n_layers=1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    header = core.assistant_header(tok)
    exs = []
    for e in core.make_instructions(6, seed=0):
        ids = core.encode_chat(core.render_chat(core.to_messages(e)), tok)
        exs.append((ids, core.sft_targets(ids, core.assistant_mask(ids, header, tok.eos_id))))
    st = core.sft_step(model, opt, [core.collate(exs[:3], tok.pad_id), core.collate(exs[3:], tok.pad_id)], 0,
                       max_lr=1e-3, warmup_steps=1, total_steps=10)
    assert st["tokens"] == sum(int((t != core.IGNORE).sum()) for _, t in exs) and st["loss"] > 0
    rows = core.pack(exs, max_len=512)
    assert len(rows) == 1 and rows[0]["cu_seqlens"].tolist()[-1] == sum(len(x) for x, _ in exs)
