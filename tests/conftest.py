"""Фикстуры тестов платформы."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def toy_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Пустой курс с блоком Z; LLMLAB_ROOT/LLMLAB_HOME указывают во временную папку."""
    root = tmp_path / "course"
    block = root / "content" / "blocks" / "Z"
    block.mkdir(parents=True)
    (block / "block.yaml").write_text(yaml.safe_dump({
        "id": "Z", "title": "Тестовый блок",
        "modules": [{"id": "module-1", "title": "M1"}, {"id": "module-2", "title": "M2"}],
        "baselines": {"toy-baseline": {"title": "База"}},
    }, allow_unicode=True), encoding="utf-8")
    (root / "content" / "questions").mkdir()
    (root / "content" / "papers").mkdir()
    # Корневой conftest подключает плагин курса так же, как в настоящем репозитории.
    (root / "conftest.py").write_text('pytest_plugins = ["llmlab.pytest_plugin"]\n', encoding="utf-8")
    monkeypatch.setenv("LLMLAB_ROOT", str(root))
    monkeypatch.setenv("LLMLAB_HOME", str(tmp_path / "state"))
    return root
