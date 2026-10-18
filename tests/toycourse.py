"""Игрушечный курс во временной папке для тестов платформы.

Задачи игрушечного курса — на чистом Python (без torch), чтобы прогоны раннера были быстрыми.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

TEST_FILE = '''
def test_add(impl):
    """add складывает два числа"""
    assert impl.add(2, 3) == 5, "2 + 3 должно быть 5"


def test_add_negative(impl):
    """add работает с отрицательными"""
    assert impl.add(-2, -3) == -5, "-2 + -3 должно быть -5"
'''

GOOD = "def add(a, b):\n    return a + b\n"
STARTER = "def add(a, b):\n    raise NotImplementedError('add')\n"
BUGGY = "def add(a, b):\n    return a + abs(b)\n"  # ловится только тестом с отрицательными


def write_task(root: Path, task_id: str, *, type_: str = "implement", module: str = "module-1", block: str = "Z",
               depends_on: list[str] | None = None, files: dict[str, str] | None = None,
               meta: dict | None = None) -> Path:
    task_dir = root / "content" / "blocks" / block / module / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": task_id, "title": f"Задача {task_id}", "block": block, "module": module,
        "type": type_, "difficulty": "easy", "depends_on": depends_on or [],
        "diff_from": None, "tags": ["toy"], "entrypoint": "add", "time_estimate_min": 15,
        **(meta or {}),
    }
    (task_dir / "task.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    defaults = {
        "README.md": f"# {task_id}\n", "hints.md": "## Подсказка 1\n", "summary.md": "## TL;DR\n",
        "solution.py": GOOD, "test_task.py": TEST_FILE,
    }
    if type_ == "debug":
        defaults["buggy.py"] = BUGGY
    elif type_ in ("implement", "project"):
        defaults["starter.py"] = STARTER
    if type_ in ("paper", "quiz"):
        defaults = {"README.md": f"# {task_id}\n"}
    for name, content in {**defaults, **(files or {})}.items():
        if content is None:
            (task_dir / name).unlink(missing_ok=True)
        else:
            (task_dir / name).write_text(textwrap.dedent(content), encoding="utf-8")
    return task_dir
