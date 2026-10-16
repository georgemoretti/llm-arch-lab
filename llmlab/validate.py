"""Проверка инвариантов курса (``llmlab validate``).

Для каждой задачи с кодом:

- ``reference`` (solution.py) проходит ВСЕ тесты;
- ``implement``/``project``: ``starter.py`` не проходит НИ ОДНОГО теста;
- ``debug``: ``buggy.py`` падает хотя бы на одном тесте (а исправление = solution.py проходит).

Плюс ошибки загрузчика: схема task.yaml, обязательные файлы, граф depends_on.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from llmlab.content_loader import Course, Task
from llmlab.runner import run_task_tests


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TaskReport:
    task_id: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _failing(result: dict) -> list[str]:
    return [t["name"] for t in result["tests"] if t["outcome"] in ("failed", "error")]


def _passing(result: dict) -> list[str]:
    return [t["name"] for t in result["tests"] if t["outcome"] == "passed"]


def _explain(result: dict) -> str:
    if result["status"] in ("timeout", "error"):
        return (result.get("message") or result["status"])[:500]
    return ""


def validate_task(task: Task, course: Course) -> TaskReport:
    report = TaskReport(task.id)
    for issue in course.issues:
        if issue.level == "error" and task.id in Path(issue.where).parts:
            report.checks.append(Check("content", False, issue.message))
    if not task.has_code:
        report.checks.append(Check("no-code", True, f"type: {task.type} — тесты не запускаются"))
        return report
    if not task.test_file.exists() or not task.solution_file.exists():
        return report  # отсутствие файлов уже записано загрузчиком

    ref = run_task_tests(task, impl="reference", root=course.root, record=False)
    ok = ref["status"] == "passed" and ref["total"] > 0
    detail = f"{ref['passed']}/{ref['total']} прошло"
    if not ok:
        failing = _failing(ref)
        detail += (f"; падают: {', '.join(failing)}" if failing else "") + (f"; {_explain(ref)}" if _explain(ref) else "")
        if failing:
            msg = next(t["message"] for t in ref["tests"] if t["name"] == failing[0])
            detail += f"\n  {failing[0]}: {(msg or '').splitlines()[0] if msg else ''}"
    report.checks.append(Check("reference ✓", ok, detail))

    if task.type == "debug":
        if task.starter_file.exists():
            bug = run_task_tests(task, impl="buggy", root=course.root, record=False)
            ok = bug["failed"] >= 1 and bug["status"] != "timeout"
            detail = f"{bug['failed']}/{bug['total']} падает"
            if not ok:
                detail += "; баг не ловится ни одним тестом" + (f"; {_explain(bug)}" if _explain(bug) else "")
            report.checks.append(Check("buggy ✗", ok, detail))
    elif task.starter_file.exists():
        st = run_task_tests(task, impl="starter", root=course.root, record=False)
        passing = _passing(st)
        ok = not passing and st["status"] != "timeout"
        detail = f"{st['passed']}/{st['total']} прошло"
        if passing:
            detail += f"; на заготовке проходят: {', '.join(passing)}"
        report.checks.append(Check("starter ✗", ok, detail))
    return report


def validate(course: Course, task_ids: list[str] | None = None, jobs: int = 4) -> tuple[list[TaskReport], list]:
    """Возвращает (отчёты по задачам, глобальные ошибки загрузчика, не привязанные к задаче)."""
    ids = task_ids if task_ids is not None else course.topo_order()
    tasks = [course.task(tid) for tid in ids]
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        reports = list(pool.map(lambda t: validate_task(t, course), tasks))
    task_dirs = set(course.tasks)
    global_errors = [
        i for i in course.errors
        if not any(part in task_dirs for part in Path(i.where).parts)
    ]
    return reports, global_errors
