"""llmlab doctor: проверки окружения на игрушечном курсе."""

from __future__ import annotations

import socket
import sys

from typer.testing import CliRunner

from toycourse import write_task
from llmlab import doctor
from llmlab.cli import app


def _by_name(checks) -> dict[str, doctor.Check]:
    return {c.name.split()[0]: c for c in checks}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_healthy_course_passes_and_runs_reference(toy_root):
    write_task(toy_root, "Z01-add")
    checks = doctor.run_checks(port=_free_port())
    by = _by_name(checks)
    assert not [c for c in checks if c.status == "fail"], [(c.name, c.detail) for c in checks]
    assert by["Курс"].status == "ok" and "задач 1" in by["Курс"].detail
    assert by["Запуск"].status == "ok" and "Z01-add: 2/2" in by["Запуск"].detail
    assert "progress" not in "".join(p.name for p in (toy_root.parent / "state").iterdir()), \
        "doctor не должен создавать историю попыток"


def test_broken_content_is_a_failure_with_fix(toy_root):
    write_task(toy_root, "Z01-add", meta={"depends_on": ["Z99-missing"]})
    by = _by_name(doctor.run_checks(port=_free_port(), run_tests=False))
    assert by["Курс"].status == "fail" and "llmlab validate" in by["Курс"].fix
    assert "Запуск" not in by


def test_busy_port_and_missing_bundle(toy_root, tmp_path, monkeypatch):
    import llmlab.server

    monkeypatch.setattr(llmlab.server, "STATIC_DIR", tmp_path / "no-static")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        by = _by_name(doctor.run_checks(port=port, run_tests=False))
    assert by["Порт"].status == "warn" and "--port" in by["Порт"].fix
    assert by["Веб-интерфейс"].status == "fail" and "npm run build" in by["Веб-интерфейс"].fix


def test_foreign_venv_is_reported(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/project/.venv")
    check = doctor.check_venv()
    assert check.status == "warn" and "uv sync" in check.fix


def test_old_python_fails(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 10, 4, "final", 0))
    assert doctor.check_python().status == "fail"


def test_cli_exit_code(toy_root):
    write_task(toy_root, "Z01-add")
    ok = CliRunner().invoke(app, ["doctor", "--quick", "--port", str(_free_port())])
    assert ok.exit_code == 0, ok.output
    assert "Всё готово" in ok.output
    write_task(toy_root, "Z02-bad", meta={"depends_on": ["Z99-missing"]})
    bad = CliRunner().invoke(app, ["doctor", "--quick", "--port", str(_free_port())])
    assert bad.exit_code == 1 and "Проблем: 1" in bad.output
