"""Запуск тестов задачи в отдельном процессе pytest.

Почему subprocess: код пользователя может зависнуть, упасть с segfault или «загрязнить»
состояние torch — изолированный процесс с таймаутом защищает сервер и CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from llmlab.content_loader import Task
from llmlab.paths import project_root
from llmlab.pytest_plugin import resolve_impl_path
from llmlab.runner import db

DEFAULT_TIMEOUT_S = 60


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run_task_tests(
    task: Task,
    impl: str = "workspace",
    impl_path: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    record: bool | None = None,
    root: Path | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Прогоняет test_task.py задачи и возвращает RunResult (dict).

    ``record`` по умолчанию True только для ``impl='workspace'``: прогоны эталона и
    заготовки (validate) не засоряют историю попыток пользователя.
    """
    root = root or project_root()
    if record is None:
        record = impl == "workspace" and impl_path is None
    code_path = resolve_impl_path(task.dir, impl, str(impl_path) if impl_path else None)

    with tempfile.TemporaryDirectory(prefix="llmlab-") as tmp:
        report_path = Path(tmp) / "report.json"
        cmd = [
            sys.executable, "-m", "pytest", str(task.test_file),
            "-p", "llmlab.pytest_plugin",
            f"--impl={impl}",
            f"--llmlab-report={report_path}",
            f"--rootdir={root}",
            "--import-mode=importlib",
            "-p", "no:cacheprovider",
            "-q", "--no-header", "-rN", "--tb=short",
            *(extra_args or []),
        ]
        if impl_path is not None:
            cmd.append(f"--impl-path={impl_path}")
        env = {**os.environ, "LLMLAB_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"}

        started = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout_s)
            output = proc.stdout + proc.stderr
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            output = _decode(exc.stdout) + _decode(exc.stderr)
            exit_code = None
        duration = round(time.perf_counter() - started, 3)

        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None

    tests = report["tests"] if report else []
    passed = sum(t["outcome"] == "passed" for t in tests)
    failed = sum(t["outcome"] in ("failed", "error") for t in tests)
    collect_errors = report["collect_errors"] if report else []

    if timed_out:
        status = "timeout"
    elif collect_errors or not tests or report is None:
        status = "error"
    elif failed == 0 and passed > 0:
        status = "passed"
    else:
        status = "failed"

    result = {
        "task_id": task.id,
        "impl": "path" if impl_path is not None else impl,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "passed": passed,
        "failed": failed,
        "total": len(tests),
        "duration_s": duration,
        "exit_code": exit_code,
        "code_path": str(code_path),
        "code_sha256": _sha256(code_path),
        "tests": tests,
        "collect_errors": collect_errors,
        "message": _summary_message(status, timeout_s, collect_errors, output),
        "output_tail": output[-4000:],
    }
    if record:
        result["attempt_id"] = db.record_attempt(result)
    return result


def _decode(data: bytes | str | None) -> str:
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


def _summary_message(status: str, timeout_s: float, collect_errors: list[dict], output: str) -> str | None:
    if status == "timeout":
        return (f"Тесты не уложились в {timeout_s:.0f} с. Проверь бесконечные циклы и слишком "
                "большие тензоры (в тестах размеры игрушечные).")
    if status == "error":
        if collect_errors:
            return "Не удалось собрать тесты:\n" + "\n".join(e["message"] for e in collect_errors)
        return "pytest не запустил ни одного теста. Вывод:\n" + output[-2000:]
    return None
