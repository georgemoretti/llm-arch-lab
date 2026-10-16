"""CLI курса: ``llmlab list | test | validate | init-workspace | serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from llmlab.content_loader import Course, load_course
from llmlab.runner import DEFAULT_TIMEOUT_S, db, run_task_tests
from llmlab.workspace import init_workspace, workspace_file

app = typer.Typer(help="LLM Arch Lab — интерактивный курс по архитектурам LLM.", no_args_is_help=True,
                  add_completion=False)
console = Console()

STATUS_STYLE = {"solved": "[green]✓ решена[/]", "attempted": "[yellow]… в процессе[/]", "new": "[dim]· новая[/]"}


def _course() -> Course:
    try:
        return load_course()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from None


def _get_task(course: Course, task_id: str):
    try:
        return course.task(task_id)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/]")
        raise typer.Exit(2) from None


@app.command("list")
def list_tasks() -> None:
    """Список задач и их статус."""
    course = _course()
    statuses = db.task_statuses()
    table = Table(show_lines=False)
    for col in ("id", "задача", "тип", "сложность", "мин", "статус"):
        table.add_column(col)
    for tid in course.topo_order():
        t = course.tasks[tid]
        st = statuses.get(tid, {"status": "new"})
        table.add_row(tid, t.title, t.type, t.difficulty, str(t.time_estimate_min), STATUS_STYLE[st["status"]])
    console.print(table)
    for issue in course.issues:
        console.print(f"[yellow]{escape(str(issue))}[/]")


@app.command("init-workspace")
def init_workspace_cmd(
    task_id: str,
    force: bool = typer.Option(False, "--force", help="перезаписать существующую рабочую копию заготовкой"),
) -> None:
    """Скопировать заготовку задачи в workspace/<id>/solution.py."""
    course = _course()
    task = _get_task(course, task_id)
    try:
        path = init_workspace(task, course.root, force=force)
    except (FileExistsError, ValueError) as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(1) from None
    console.print(f"Рабочая копия: [bold]{path.relative_to(course.root)}[/]")


def _print_result(result: dict) -> None:
    for t in result["tests"]:
        icon = {"passed": "[green]✓[/]", "skipped": "[dim]-[/]"}.get(t["outcome"], "[red]✗[/]")
        console.print(f" {icon} {escape(t['title'])} [dim]({t['name']})[/]")
        if t["outcome"] in ("failed", "error") and t["message"]:
            for line in t["message"].splitlines():
                console.print(f"     [red]{escape(line)}[/]")
            loc = t.get("location")
            if loc:
                console.print(f"     [dim]→ {loc['file']}:{loc['line']} в {loc['function']}: {escape(loc['source'])}[/]")
    if result.get("message"):
        console.print(f"[red]{escape(result['message'])}[/]")
    color = {"passed": "green", "failed": "red"}.get(result["status"], "yellow")
    console.print(f"[{color}]{result['status'].upper()}[/] — {result['passed']}/{result['total']} тестов, "
                  f"{result['duration_s']:.1f} с")


@app.command("test")
def test_cmd(
    task_id: str,
    impl: str = typer.Option("workspace", help="workspace | reference | starter | buggy"),
    impl_path: Optional[Path] = typer.Option(None, help="проверить произвольный файл"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT_S, help="таймаут, с"),
) -> None:
    """Прогнать тесты задачи (по умолчанию — на твоей рабочей копии)."""
    course = _course()
    task = _get_task(course, task_id)
    if not task.has_code:
        console.print(f"[yellow]У задачи {task.id} (type: {task.type}) нет автотестов.[/]")
        raise typer.Exit(1)
    if impl == "workspace" and impl_path is None and not workspace_file(task, course.root).exists():
        path = init_workspace(task, course.root)
        console.print(f"[dim]Рабочей копии не было — создал {path.relative_to(course.root)} из заготовки.[/]")
    console.print(f"[bold]{task.id}[/] — {task.title} [dim](impl={impl if impl_path is None else impl_path})[/]")
    result = run_task_tests(task, impl=impl, impl_path=impl_path.resolve() if impl_path else None,
                            timeout_s=timeout, root=course.root)
    _print_result(result)
    raise typer.Exit(0 if result["status"] == "passed" else 1)


@app.command("validate")
def validate_cmd(
    task_id: Optional[str] = typer.Argument(None, help="id задачи; без аргумента — нужен --all"),
    all_: bool = typer.Option(False, "--all", help="проверить все задачи"),
    jobs: int = typer.Option(4, "--jobs", "-j", help="сколько задач проверять параллельно"),
) -> None:
    """Инварианты: эталон проходит всё, заготовка — ничего, баг debug-задачи ловится."""
    from llmlab.validate import validate

    if not task_id and not all_:
        console.print("[yellow]Укажи id задачи или --all[/]")
        raise typer.Exit(2)
    course = _course()
    ids = None if all_ else [_get_task(course, task_id).id]
    with console.status("Прогоняю тесты…"):
        reports, global_errors = validate(course, ids, jobs=jobs)

    table = Table(title="llmlab validate")
    table.add_column("задача")
    table.add_column("проверка")
    table.add_column("")
    table.add_column("детали", overflow="fold")
    for rep in reports:
        for i, check in enumerate(rep.checks):
            table.add_row(rep.task_id if i == 0 else "", check.name,
                          "[green]OK[/]" if check.ok else "[red]FAIL[/]", escape(check.detail))
    console.print(table)
    for issue in global_errors:
        console.print(f"[red]{escape(str(issue))}[/]")
    for issue in course.issues:
        if issue.level == "warning":
            console.print(f"[yellow]{escape(str(issue))}[/]")
    bad = [r.task_id for r in reports if not r.ok]
    if bad or global_errors:
        console.print(f"[red]Нарушены инварианты: {', '.join(bad) or 'контент'}[/]")
        raise typer.Exit(1)
    console.print(f"[green]Все инварианты соблюдены (задач: {len(reports)}).[/]")


@app.command("serve")
def serve(
    port: int = typer.Option(8765, help="порт на 127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="открыть браузер после запуска"),
) -> None:
    """Веб-интерфейс курса на http://127.0.0.1:<port> (только локально)."""
    import threading
    import webbrowser

    import uvicorn

    from llmlab.server import STATIC_DIR, create_app

    course = _course()
    if not (STATIC_DIR / "index.html").exists():
        console.print("[yellow]Фронтенд не собран (нет llmlab/static/index.html). "
                      "Собери его: `cd web && npm install && npm run build`.[/]")
    url = f"http://127.0.0.1:{port}"
    console.print(f"LLM Arch Lab: [bold]{url}[/]  [dim](Ctrl+C — остановить)[/]")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    # Только 127.0.0.1: сервер исполняет код, наружу его открывать нельзя.
    uvicorn.run(create_app(course.root), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()
