"""Рабочие копии решений пользователя: ``workspace/<task_id>/solution.py``."""

from __future__ import annotations

import shutil
from pathlib import Path

from llmlab.content_loader import Task
from llmlab.paths import workspace_dir


def workspace_file(task: Task, root: Path | None = None) -> Path:
    return workspace_dir(root) / task.id / "solution.py"


def init_workspace(task: Task, root: Path | None = None, force: bool = False) -> Path:
    """Копирует заготовку (или багованный код для debug-задачи) в workspace.

    Существующий файл не перезаписывается без ``force=True`` — это код пользователя.
    """
    if not task.has_code:
        raise ValueError(f"У задачи {task.id} (type: {task.type}) нет кода для рабочей копии")
    target = workspace_file(task, root)
    if target.exists() and not force:
        raise FileExistsError(f"{target} уже существует (используй --force, чтобы сбросить к заготовке)")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task.starter_file, target)
    return target
