from typer.testing import CliRunner

from toycourse import GOOD, write_task
from llmlab.cli import app
from llmlab.content_loader import load_course
from llmlab.validate import validate


def _checks(report) -> dict[str, bool]:
    return {c.name: c.ok for c in report.checks}


def test_good_implement_and_debug_tasks(toy_root):
    write_task(toy_root, "Z01-add")
    write_task(toy_root, "Z01d-add-debug", type_="debug", depends_on=["Z01-add"])
    write_task(toy_root, "Z-P1-reading", type_="paper")
    reports, global_errors = validate(load_course(toy_root))
    by_id = {r.task_id: r for r in reports}
    assert not global_errors
    assert all(r.ok for r in reports), [(r.task_id, r.checks) for r in reports]
    assert _checks(by_id["Z01-add"]) == {"reference ✓": True, "starter ✗": True}
    assert _checks(by_id["Z01d-add-debug"]) == {"reference ✓": True, "buggy ✗": True}
    assert _checks(by_id["Z-P1-reading"]) == {"no-code": True}


def test_starter_that_passes_a_test_is_rejected(toy_root):
    write_task(toy_root, "Z01-add", files={"starter.py": "def add(a, b):\n    return 5\n"})
    (report,), _ = validate(load_course(toy_root))
    check = next(c for c in report.checks if c.name == "starter ✗")
    assert not check.ok and "test_add" in check.detail


def test_broken_reference_is_rejected(toy_root):
    write_task(toy_root, "Z01-add", files={"solution.py": "def add(a, b):\n    return a * b\n"})
    (report,), _ = validate(load_course(toy_root))
    check = next(c for c in report.checks if c.name == "reference ✓")
    assert not check.ok and "test_add" in check.detail


def test_undetected_bug_is_rejected(toy_root):
    write_task(toy_root, "Z01d-add-debug", type_="debug", files={"buggy.py": GOOD})
    (report,), _ = validate(load_course(toy_root))
    check = next(c for c in report.checks if c.name == "buggy ✗")
    assert not check.ok and "не ловится" in check.detail


def test_content_errors_fail_validation(toy_root):
    write_task(toy_root, "Z01-add", depends_on=["Z99-nope"])
    (report,), _ = validate(load_course(toy_root))
    assert not report.ok
    assert any(c.name == "content" and "Z99-nope" in c.detail for c in report.checks)


def test_cli_validate_and_list(toy_root):
    write_task(toy_root, "Z01-add")
    runner = CliRunner()
    ok = runner.invoke(app, ["validate", "--all"])
    assert ok.exit_code == 0, ok.output
    assert "Все инварианты соблюдены" in ok.output

    listing = runner.invoke(app, ["list"])
    assert listing.exit_code == 0 and "Z01-add" in listing.output

    write_task(toy_root, "Z02-bad", files={"solution.py": "def add(a, b):\n    return 0\n"})
    bad = runner.invoke(app, ["validate", "Z02-bad"])
    assert bad.exit_code == 1


def test_cli_test_creates_workspace(toy_root):
    write_task(toy_root, "Z01-add")
    result = CliRunner().invoke(app, ["test", "Z01-add"])
    assert result.exit_code == 1  # заготовка не проходит
    assert (toy_root / "workspace" / "Z01-add" / "solution.py").exists()
    passed = CliRunner().invoke(app, ["test", "Z01-add", "--impl", "reference"])
    assert passed.exit_code == 0, passed.output
