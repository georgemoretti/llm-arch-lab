"""``llmlab doctor`` — проверка окружения: всё ли готово, чтобы курс работал.

Каждая проверка возвращает ``Check``: ``ok`` (зелёный), ``warn`` (работает, но не всё) или ``fail`` (курс не
заработает). У warn/fail всегда есть ``fix`` — что сделать. Проверки не меняют состояние пользователя: прогон
тестов идёт на эталоне без записи в историю, а SQLite открывается во временной базе.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import socket
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

Status = Literal["ok", "warn", "fail"]

MIN_PYTHON = (3, 11)
CORE_PACKAGES = ("torch", "numpy", "pytest", "fastapi", "uvicorn", "typer", "yaml", "pydantic", "safetensors", "rich")
WEIGHTS_PACKAGES = ("huggingface_hub", "tokenizers", "transformers")
SMOKE_TASK = "C00-causal-mha"


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""


def check_python() -> Check:
    v = tuple(sys.version_info[:3])
    version = ".".join(map(str, v))
    if v[:2] < MIN_PYTHON:
        return Check("Python", "fail", f"{version} ({sys.executable})",
                     "нужен Python ≥ 3.11: `uv sync` поставит подходящий сам (см. .python-version)")
    return Check("Python", "ok", f"{version} ({sys.executable})")


def check_venv() -> Check:
    """Активированный venv должен быть тем, из которого запущен llmlab (ловит скопированный чужой .venv)."""
    active = os.environ.get("VIRTUAL_ENV")
    prefix = Path(sys.prefix).resolve()
    if active and Path(active).resolve() != prefix:
        return Check("Виртуальное окружение", "warn", f"активирован {active}, а llmlab запущен из {prefix}",
                     "пересоздай окружение: `rm -rf .venv && uv sync`, затем `source .venv/bin/activate`")
    in_venv = sys.prefix != sys.base_prefix
    return Check("Виртуальное окружение", "ok" if in_venv else "warn", str(prefix) if in_venv else "системный Python",
                 "" if in_venv else "лучше ставить в venv: `uv sync` (или `python -m venv .venv && pip install -e .`)")


def check_packages() -> Check:
    missing = [p for p in CORE_PACKAGES if importlib.util.find_spec(p) is None]
    if missing:
        return Check("Зависимости", "fail", "нет: " + ", ".join(missing), "`uv sync` (или `pip install -e .`)")
    return Check("Зависимости", "ok", ", ".join(CORE_PACKAGES))


def check_torch() -> Check:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — сломанная установка torch падает чем угодно
        return Check("PyTorch", "fail", f"не импортируется: {exc}", "`uv sync` (CPU-колёса ставятся автоматически)")
    try:
        g = torch.Generator().manual_seed(0)
        a = torch.randn(32, 32, generator=g)
        ok = torch.allclose(a @ torch.eye(32), a)
    except Exception as exc:  # noqa: BLE001
        return Check("PyTorch", "fail", f"torch {torch.__version__}: matmul упал: {exc}", "переустанови torch: `uv sync --reinstall-package torch`")
    if not ok:
        return Check("PyTorch", "fail", f"torch {torch.__version__}: неверный результат matmul",
                     "переустанови torch: `uv sync --reinstall-package torch`")
    threads = torch.get_num_threads()
    return Check("PyTorch", "ok", f"{torch.__version__}, CPU, потоков: {threads}")


def check_course(root: Path | None) -> tuple[Check, object | None]:
    from llmlab.content_loader import load_course

    try:
        course = load_course(root)
    except RuntimeError as exc:
        return Check("Курс", "fail", str(exc), "запускай llmlab из клона репозитория или задай LLMLAB_ROOT"), None
    errors = course.errors
    if errors:
        first = errors[0]
        return Check("Курс", "fail", f"{len(errors)} ошибок контента, первая: {first.where}: {first.message}",
                     "`llmlab validate --all` покажет все; если ты не менял content/ — `git status`"), course
    return Check("Курс", "ok", f"{course.root}: задач {len(course.tasks)}, вопросов {len(course.questions)}"), course


def check_frontend() -> Check:
    from llmlab.server import STATIC_DIR

    index = STATIC_DIR / "index.html"
    if not index.exists():
        return Check("Веб-интерфейс", "fail", f"нет {index}",
                     "бандл коммитится в репозиторий — `git checkout llmlab/static`; или `cd web && npm ci && npm run build`")
    import re

    html = index.read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="/?(assets/[^"]+)"', html)
    missing = [r for r in refs if not (STATIC_DIR / r).exists()]
    if missing:
        return Check("Веб-интерфейс", "fail", f"index.html ссылается на отсутствующие файлы: {missing[:2]}",
                     "`git checkout llmlab/static` или пересобери: `cd web && npm ci && npm run build`")
    return Check("Веб-интерфейс", "ok", f"бандл на месте (файлов в index.html: {len(refs)})")


def check_state(root: Path | None) -> Check:
    from llmlab.paths import state_dir

    try:
        path = state_dir(root)
        probe = path / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("Хранилище прогресса", "fail", f"нет записи: {exc}",
                     "проверь права на папку .llmlab/ или задай LLMLAB_HOME=<папка с правом записи>")
    with tempfile.TemporaryDirectory() as tmp:
        con = sqlite3.connect(Path(tmp) / "probe.db")
        try:
            con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            fts = True
        except sqlite3.OperationalError:
            fts = False
        finally:
            con.close()
    detail = f"{path}, SQLite {sqlite3.sqlite_version}"
    if not fts:
        return Check("Хранилище прогресса", "warn", detail + ", без FTS5",
                     "поиск по конспектам будет простым сканом — работает, но медленнее; FTS5 есть в Python с python.org и uv")
    return Check("Хранилище прогресса", "ok", detail + ", FTS5")


def check_port(port: int) -> Check:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return Check(f"Порт {port}", "warn", "занят (возможно, llmlab serve уже запущен)",
                         f"открой http://127.0.0.1:{port} или запусти на другом: `llmlab serve --port 8766`")
    return Check(f"Порт {port}", "ok", "свободен")


def check_runner(course) -> Check:
    """Сквозная проверка раннера: pytest в subprocess на эталоне демо-задачи (без записи в историю)."""
    from llmlab.runner import run_task_tests

    task = course.tasks.get(SMOKE_TASK) or next((t for t in course.tasks.values() if t.has_code), None)
    if task is None:
        return Check("Запуск тестов", "warn", "в курсе нет задач с тестами — пропускаю")
    result = run_task_tests(task, impl="reference", record=False, root=course.root)
    if result["status"] != "passed":
        detail = result.get("message") or f"{result['passed']}/{result['total']} тестов"
        return Check("Запуск тестов", "fail", f"эталон {task.id}: {result['status']}: {detail}"[:300],
                     f"запусти вручную и посмотри вывод: `llmlab test {task.id} --impl reference`")
    return Check("Запуск тестов", "ok", f"эталон {task.id}: {result['passed']}/{result['total']} за "
                                        f"{result['duration_s']:.1f} с")


def check_weights() -> Check:
    """Нужно только для финального проекта (C25, `llmlab generate`) — поэтому warn, а не fail."""
    missing = [p for p in WEIGHTS_PACKAGES if importlib.util.find_spec(p) is None]
    if missing:
        return Check("Реальные веса (C25)", "warn", "не установлено: " + ", ".join(missing),
                     "нужно только для финального проекта: `uv sync --extra weights`")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cached = (cache / "models--Qwen--Qwen3-0.6B").is_dir()
    return Check("Реальные веса (C25)", "ok", "пакеты есть; Qwen3-0.6B " +
                 ("в кеше HF" if cached else "ещё не скачана (~1.5 ГБ при первом `llmlab generate`)"))


def run_checks(root: Path | None = None, port: int = 8765, run_tests: bool = True,
               on_check: Callable[[Check], None] | None = None) -> list[Check]:
    """Все проверки по порядку. Если нет зависимостей или torch — дальше не идём: остальное упадёт по той же причине."""
    checks: list[Check] = []

    def add(check: Check) -> Check:
        checks.append(check)
        if on_check:
            on_check(check)
        return check

    add(check_python())
    add(check_venv())
    if add(check_packages()).status == "fail" or add(check_torch()).status == "fail":
        return checks
    course_check, course = check_course(root)
    add(course_check)
    add(check_frontend())
    add(check_state(course.root if course is not None else root))
    add(check_port(port))
    if run_tests and course_check.status != "fail":
        add(check_runner(course))
    add(check_weights())
    return checks
