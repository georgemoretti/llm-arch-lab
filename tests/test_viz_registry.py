"""Каждая визуализация из content/viz имеет компонент во фронтенде и наоборот."""

from __future__ import annotations

import re
from pathlib import Path

from toycourse import write_task
from llmlab.content_loader import load_course

REPO = Path(__file__).resolve().parents[1]


def test_viz_references_are_validated(toy_root):
    (toy_root / "content" / "viz").mkdir()
    (toy_root / "content" / "viz" / "demo.yaml").write_text("id: demo\ntitle: Демо\n", encoding="utf-8")
    write_task(toy_root, "Z01-add", meta={"viz": ["demo", "missing"]},
               files={"README.md": "# x\n\n:::viz demo:::\n\n:::viz ghost:::\n"})
    msgs = [i.message for i in load_course(toy_root).errors]
    assert "viz: нет content/viz/missing.yaml" in msgs
    assert "viz: нет content/viz/ghost.yaml" in msgs
    assert not any("demo" in m for m in msgs)


def test_repo_viz_registry_matches_content():
    registry = (REPO / "web" / "src" / "viz" / "registry.ts").read_text(encoding="utf-8")
    components = set(re.findall(r'^\s*"([a-z0-9-]+)":', registry, flags=re.MULTILINE))
    described = {p.stem for p in (REPO / "content" / "viz").glob("*.yaml")}
    assert described == components, (
        f"нет компонента для {sorted(described - components)}; нет content/viz/*.yaml для {sorted(components - described)}"
    )
