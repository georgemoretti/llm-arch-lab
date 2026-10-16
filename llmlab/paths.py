"""Где лежит курс и где хранится состояние пользователя.

Корень проекта ищется так:
1. переменная окружения ``LLMLAB_ROOT``;
2. вверх от текущей директории — первая папка, где есть ``content/blocks``;
3. родитель пакета ``llmlab`` (editable-установка из клона репозитория).

Состояние (SQLite) лежит в ``<root>/.llmlab/``; переопределяется ``LLMLAB_HOME``.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def _looks_like_root(path: Path) -> bool:
    return (path / "content" / "blocks").is_dir()


def project_root() -> Path:
    env = os.environ.get("LLMLAB_ROOT")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _looks_like_root(candidate):
            return candidate
    fallback = PACKAGE_DIR.parent
    if _looks_like_root(fallback):
        return fallback
    raise RuntimeError(
        "Не найден корень курса (папка с content/blocks). "
        "Запусти команду из клона репозитория или задай LLMLAB_ROOT."
    )


def content_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "content"


def workspace_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "workspace"


def notes_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "notes"


def state_dir(root: Path | None = None) -> Path:
    env = os.environ.get("LLMLAB_HOME")
    path = Path(env).resolve() if env else (root or project_root()) / ".llmlab"
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path(root: Path | None = None) -> Path:
    return state_dir(root) / "progress.db"
