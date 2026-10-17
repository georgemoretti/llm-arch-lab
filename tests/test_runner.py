from toycourse import GOOD, write_task
from llmlab.content_loader import load_course
from llmlab.runner import db, run_task_tests
from llmlab.workspace import init_workspace, workspace_file


def _task(root, task_id="Z01-add", **kw):
    write_task(root, task_id, **kw)
    return load_course(root).tasks[task_id]


def test_reference_passes_and_is_not_recorded(toy_root):
    task = _task(toy_root)
    result = run_task_tests(task, impl="reference", root=toy_root)
    assert result["status"] == "passed", result
    assert (result["passed"], result["failed"], result["total"]) == (2, 0, 2)
    assert result["tests"][0]["title"] == "add складывает два числа"
    assert db.attempts(task.id) == []


def test_starter_fails_with_readable_message(toy_root):
    task = _task(toy_root)
    result = run_task_tests(task, impl="starter", root=toy_root)
    assert result["status"] == "failed"
    assert result["passed"] == 0 and result["failed"] == 2
    msg = result["tests"][0]["message"]
    assert "не реализована" in msg and "NotImplementedError" in msg


def test_workspace_attempt_is_recorded(toy_root):
    task = _task(toy_root)
    path = init_workspace(task, toy_root)
    first = run_task_tests(task, root=toy_root)
    assert first["status"] == "failed" and "attempt_id" in first

    path.write_text(GOOD, encoding="utf-8")
    second = run_task_tests(task, root=toy_root)
    assert second["status"] == "passed"

    attempts = db.attempts(task.id)
    assert [a["status"] for a in attempts] == ["failed", "passed"]
    assert attempts[0]["code_sha256"] != attempts[1]["code_sha256"]
    statuses = db.task_statuses()
    assert statuses[task.id]["status"] == "solved"
    assert statuses[task.id]["attempts"] == 2


def test_assertion_message_and_user_location(toy_root):
    task = _task(toy_root)
    path = init_workspace(task, toy_root)
    path.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    result = run_task_tests(task, root=toy_root)
    failed = [t for t in result["tests"] if t["outcome"] == "failed"]
    assert failed and failed[0]["message"] == "2 + 3 должно быть 5"

    path.write_text("def add(a, b):\n    x = 1\n    return a + b + undefined_name\n", encoding="utf-8")
    result = run_task_tests(task, root=toy_root)
    loc = result["tests"][0]["location"]
    assert loc == {"file": "solution.py", "line": 3, "function": "add",
                   "source": "return a + b + undefined_name"}
    assert "NameError" in result["tests"][0]["message"]


def test_syntax_error_in_user_code(toy_root):
    task = _task(toy_root)
    init_workspace(task, toy_root).write_text("def add(a, b)\n    return a + b\n", encoding="utf-8")
    result = run_task_tests(task, root=toy_root)
    assert result["status"] == "failed"
    assert all(t["outcome"] == "error" for t in result["tests"])
    assert "SyntaxError" in result["tests"][0]["message"]


def test_missing_workspace_gives_hint(toy_root):
    task = _task(toy_root)
    result = run_task_tests(task, root=toy_root, record=False)
    assert result["status"] == "failed"
    assert "llmlab init-workspace Z01-add" in result["tests"][0]["message"]


def test_timeout(toy_root):
    task = _task(toy_root)
    init_workspace(task, toy_root).write_text("def add(a, b):\n    while True:\n        pass\n", encoding="utf-8")
    result = run_task_tests(task, root=toy_root, timeout_s=3)
    assert result["status"] == "timeout"
    assert "не уложились" in result["message"]
    assert db.attempts(task.id)[-1]["status"] == "timeout"


def test_broken_test_file_is_error(toy_root):
    task = _task(toy_root, files={"test_task.py": "import nonexistent_module_xyz\n"})
    result = run_task_tests(task, impl="reference", root=toy_root)
    assert result["status"] == "error"
    assert "nonexistent_module_xyz" in result["message"]


def test_init_workspace_does_not_overwrite(toy_root):
    task = _task(toy_root)
    path = init_workspace(task, toy_root)
    path.write_text("# мой код\n", encoding="utf-8")
    try:
        init_workspace(task, toy_root)
    except FileExistsError:
        pass
    else:
        raise AssertionError("init_workspace перезаписал код пользователя")
    assert path.read_text(encoding="utf-8") == "# мой код\n"
    init_workspace(task, toy_root, force=True)
    assert "NotImplementedError" in workspace_file(task, toy_root).read_text(encoding="utf-8")


def test_debug_workspace_starts_from_buggy_code(toy_root):
    task = _task(toy_root, "Z02-fix", type_="debug")
    assert "abs(b)" in init_workspace(task, toy_root).read_text(encoding="utf-8")


def test_importing_reference_from_llmlab_is_rejected(toy_root, tmp_path):
    """Ответ нельзя «решить» импортом эталона из llmlab; обычные импорты не мешают."""
    task = _task(toy_root)
    cheat = tmp_path / "cheat.py"
    cheat.write_text("from llmlab.paths import project_root as add\n", encoding="utf-8")
    result = run_task_tests(task, impl_path=cheat, root=toy_root)
    assert result["status"] == "failed" and result["passed"] == 0
    assert "llmlab.paths" in result["tests"][0]["message"], result["tests"][0]["message"]
    honest = tmp_path / "honest.py"
    honest.write_text("from operator import add\n", encoding="utf-8")
    assert run_task_tests(task, impl_path=honest, root=toy_root)["status"] == "passed"
