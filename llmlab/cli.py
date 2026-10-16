"""CLI курса: ``llmlab list | test | validate | init-workspace | serve | notes | questions``."""

from __future__ import annotations

import sys
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
notes_app = typer.Typer(help="Конспекты: пересборка и экспорт.", no_args_is_help=True)
app.add_typer(notes_app, name="notes")
questions_app = typer.Typer(help="База вопросов с собеседований: импорт, статус разметки, проверки.",
                            no_args_is_help=True)
app.add_typer(questions_app, name="questions")

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
    if result["status"] == "passed" and impl == "workspace" and impl_path is None:
        _after_solve(course, task)
    raise typer.Exit(0 if result["status"] == "passed" else 1)


def _after_solve(course: Course, task) -> None:
    """Зелёный прогон: обновить конспект или (впервые) спросить «Главное, что я понял»."""
    from llmlab import notes

    try:
        info = notes.build_note(course, task, course.root)
        console.print(f"[green]Конспект обновлён:[/] notes/{info['path']}")
        return
    except notes.NoteError as exc:
        if exc.code != "needs_insight":
            console.print(f"[yellow]{escape(str(exc))}[/]")
            return
    if not sys.stdin.isatty():
        console.print(f"[cyan]Задача решена! Создай конспект:[/] llmlab notes build {task.id} --insight \"…\"")
        return
    console.print(f"[cyan]Задача решена! Что главное ты понял? (1–3 строки, пустая строка — закончить; "
                  f"пропустить — сразу Enter)[/]")
    lines: list[str] = []
    while len(lines) < notes.generator.INSIGHT_MAX_LINES:
        line = input("> ").strip()
        if not line:
            break
        lines.append(line)
    if not lines:
        console.print(f"[dim]Позже: llmlab notes build {task.id} --insight \"…\" или в веб-интерфейсе.[/]")
        return
    try:
        info = notes.build_note(course, task, course.root, insight="\n".join(lines))
        console.print(f"[green]Конспект создан:[/] notes/{info['path']}")
    except notes.NoteError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")


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


@notes_app.command("build")
def notes_build(
    task_id: Optional[str] = typer.Argument(None, help="id задачи; без аргумента — все конспекты"),
    insight: Optional[str] = typer.Option(None, "--insight", "-i",
                                          help="«Главное, что я понял» (1–3 строки, перенос — \\n)"),
) -> None:
    """Создать или пересобрать конспект. Вывод «Главное, что я понял» сохраняется."""
    from llmlab import notes

    course = _course()
    if task_id is None:
        if insight:
            console.print("[yellow]--insight задаётся только вместе с id задачи.[/]")
            raise typer.Exit(2)
        built, skipped = notes.build_all(course, course.root)
        for b in built:
            console.print(f"[green]✓[/] notes/{b['path']}")
        for tid, reason in skipped:
            console.print(f"[yellow]· {tid}: {escape(reason)}[/]")
        console.print(f"Пересобрано конспектов: {len(built)}" + (f", пропущено: {len(skipped)}" if skipped else ""))
        return
    task = _get_task(course, task_id)
    try:
        info = notes.build_note(course, task, course.root,
                                insight=insight.replace("\\n", "\n") if insight else None)
    except notes.NoteError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        if exc.code == "needs_insight":
            console.print(f'[dim]Добавь: llmlab notes build {task.id} --insight "…"[/]')
        raise typer.Exit(1) from None
    console.print(f"[green]{'Создан' if info['created'] else 'Пересобран'}:[/] notes/{info['path']}")


@notes_app.command("export")
def notes_export(
    out: Path = typer.Option(Path("llmlab-notes.zip"), "--out", "-o", help="куда положить zip"),
) -> None:
    """Выгрузить все конспекты в zip (распакуй и открой папку в Obsidian как vault)."""
    from llmlab import notes

    course = _course()
    notes.write_index_note(course, course.root)
    out.write_bytes(notes.export_zip(course.root))
    console.print(f"[green]Готово:[/] {out}")


@questions_app.command("import")
def questions_import(
    paths: list[Path] = typer.Argument(None, help="файлы или папки; по умолчанию data/interview_raw/"),
) -> None:
    """Разобрать сырую базу (md/txt/csv/json), убрать дубли, сохранить в .llmlab/questions_import/raw.jsonl."""
    from llmlab.questions import importer

    course = _course()
    paths = paths or [course.root / "data" / "interview_raw"]
    try:
        res = importer.import_paths(paths, course.root)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from None
    console.print(f"Файлы: {', '.join(res.files) or '—'}" + (f"; пропущены: {', '.join(res.skipped)}" if res.skipped else ""))
    console.print(f"Вопросов: {res.parsed}, уникальных: {res.unique}, точных дублей: {res.duplicates}")
    console.print(f"[dim]Сохранено: {res.out} (в git не попадает)[/]")
    st = importer.import_status(course, course.root)
    console.print(f"Размечено: {st.labeled} / {st.total}. Дальше: [bold]llmlab questions batch[/] — "
                  "следующая пачка для разметки (см. docs/QUESTIONS_IMPORT.md).")


@questions_app.command("status")
def questions_status() -> None:
    """Сколько сырых вопросов уже размечено в content/questions (по source_ref)."""
    from llmlab.questions import importer

    course = _course()
    st = importer.import_status(course, course.root)
    if not st.total:
        console.print("[yellow]Импорта ещё не было: llmlab questions import[/]")
        return
    user = [q for q in course.questions.values() if q.source == "user-db"]
    console.print(f"Сырых уникальных: {st.total}; покрыто разметкой: {st.labeled}; осталось: {len(st.remaining)}")
    console.print(f"Вопросов user-db в content/questions: {len(user)} "
                  f"(по темам: {', '.join(f'{t}={n}' for t, n in _count(q.topic for q in user))})")
    if st.unknown_refs:
        console.print(f"[yellow]source_ref без сырого вопроса: {len(st.unknown_refs)} (база изменилась?)[/]")


def _count(items) -> list[tuple[str, int]]:
    from collections import Counter

    return sorted(Counter(items).items(), key=lambda kv: -kv[1])


@questions_app.command("batch")
def questions_batch(size: int = typer.Option(50, "--size", "-n", help="сколько вопросов в пачке")) -> None:
    """Следующая пачка неразмеченных сырых вопросов (ref, раздел, текст) — вход для разметки."""
    from llmlab.questions import importer

    course = _course()
    batch = importer.next_batch(course, size, course.root)
    if not batch:
        console.print("[green]Всё размечено.[/]")
        return
    for q in batch:
        console.print(f"{q.ref}  [dim]{escape(' / '.join(q.section))}[/]\n    {escape(q.text)}")
    console.print(f"[dim]В пачке {len(batch)}. Размеченные вопросы клади в content/questions/interview-*.yaml "
                  "с source: user-db и source_ref: [<ref>, …].[/]")


@questions_app.command("check")
def questions_check() -> None:
    """Проверить, что размеченные вопросы и ответы не копируют сырой текст (репозиторий публичный)."""
    from llmlab.questions import importer

    course = _course()
    hits = importer.verbatim_matches(course, course.root)
    if not importer.load_raw(course.root):
        console.print("[yellow]Нет импорта — сравнивать не с чем (llmlab questions import).[/]")
        return
    for h in hits:
        console.print(f"[red]✗ {h['id']}.{h['field']} ≈ сырой {h['ref']} (сходство {h['ratio']})[/]")
    if hits:
        raise typer.Exit(1)
    console.print("[green]Совпадений с сырым текстом нет.[/]")


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
