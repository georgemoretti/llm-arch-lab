"""pytest-плагин курса: выбор реализации (``--impl``), детерминизм и JSON-отчёт.

Подключается из корневого ``conftest.py`` (для ручного ``pytest content/...``) и через
``-p llmlab.pytest_plugin`` из раннера.

Фикстуры для тестов задач:

- ``impl`` — модуль с реализацией, которую проверяем (выбирается флагом ``--impl``). Всё, что эталон
  определяет сам и чего не импортирует заготовка, должно быть определено в этом файле, а не взято из
  ``llmlab.core`` (``check_own_code``) — иначе все тесты падают с объяснением;
- ``ref``  — эталон ``solution.py`` той же задачи (для численного сравнения);
- ``task_info`` — dict из ``task.yaml``.

Флаги:

- ``--impl=workspace|reference|starter|buggy`` (по умолчанию ``workspace``);
- ``--impl-path=<file.py>`` — проверить произвольный файл (перекрывает ``--impl``);
- ``--llmlab-report=<file.json>`` — записать структурированный отчёт (использует раннер);
- ``--weights`` (или ``LLMLAB_WEIGHTS=1``) — запускать тесты ``@pytest.mark.weights``, которые скачивают
  реальные веса с Hugging Face. Без флага они пропускаются.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest
import yaml

IMPL_CHOICES = ("workspace", "reference", "starter", "buggy")
_impl_path_key = pytest.StashKey[dict]()
_module_cache: dict[Path, ModuleType] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llmlab")
    group.addoption("--impl", choices=IMPL_CHOICES, default="workspace",
                    help="какую реализацию тестировать: workspace (по умолчанию), reference, starter, buggy")
    group.addoption("--impl-path", default=None, help="путь к произвольному файлу реализации")
    group.addoption("--llmlab-report", default=None, help="куда записать JSON-отчёт о прогоне")
    group.addoption("--weights", action="store_true", default=False,
                    help="запускать тесты с реальными весами (@pytest.mark.weights)")


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_impl_path_key] = {}
    config.addinivalue_line("markers", "weights: тест скачивает реальные веса с Hugging Face (нужен --weights)")
    report_path = config.getoption("--llmlab-report", default=None)
    if report_path:
        config.pluginmanager.register(_JsonReporter(Path(report_path)), "llmlab-json-reporter")


def weights_enabled(config: pytest.Config) -> bool:
    return bool(config.getoption("--weights", default=False)) or os.environ.get("LLMLAB_WEIGHTS") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if weights_enabled(config):
        return
    skip = pytest.mark.skip(reason="нужны реальные веса: запусти с --weights (llmlab test <id> --weights)")
    for item in items:
        if item.get_closest_marker("weights") is not None:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Поиск реализации
# ---------------------------------------------------------------------------

def _find_task_dir(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "task.yaml").exists():
            return candidate
    raise pytest.UsageError(f"Рядом с {start} нет task.yaml — это не тест задачи курса")


def _project_root(task_dir: Path) -> Path:
    env = os.environ.get("LLMLAB_ROOT")
    if env:
        return Path(env).resolve()
    for candidate in task_dir.parents:
        if (candidate / "content" / "blocks").is_dir():
            return candidate
    raise pytest.UsageError("Не найден корень курса (папка с content/blocks); задай LLMLAB_ROOT")


def _load_task_info(task_dir: Path) -> dict:
    return yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))


def resolve_impl_path(task_dir: Path, impl: str, impl_path: str | None = None) -> Path:
    """Путь к файлу реализации для данного режима ``--impl``."""
    if impl_path:
        return Path(impl_path).resolve()
    info = _load_task_info(task_dir)
    if impl == "reference":
        return task_dir / "solution.py"
    if impl == "starter":
        return task_dir / ("buggy.py" if info.get("type") == "debug" else "starter.py")
    if impl == "buggy":
        return task_dir / "buggy.py"
    return _project_root(task_dir) / "workspace" / info["id"] / "solution.py"


def _import_file(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_module(path: Path, tag: str) -> ModuleType:
    path = path.resolve()
    if path not in _module_cache:
        if not path.exists():
            pytest.fail(f"Файл реализации не найден: {path}", pytrace=False)
        _module_cache[path] = _import_file(path, f"llmlab_impl_{tag}_{abs(hash(path))}")
    return _module_cache[path]


def _top_level_names(path: Path) -> tuple[set[str], set[str]]:
    """(определённые def/class, импортированные) имена верхнего уровня файла."""
    if not path.exists():
        return set(), set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imported = {a.asname or a.name.split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}
    return defined, imported


def check_own_code(module: ModuleType, task_dir: Path) -> None:
    """То, что задача просит написать, должно быть написано в файле решения, а не импортировано из эталонов
    ``llmlab`` (``llmlab.core.*`` содержит ответы для следующих задач). Готовые детали, которые импортирует сама
    заготовка, разрешены."""
    info = _load_task_info(task_dir)
    own, _ = _top_level_names(task_dir / "solution.py")
    _, given = _top_level_names(task_dir / ("buggy.py" if info.get("type") == "debug" else "starter.py"))
    borrowed = []
    for name in sorted(own - given):
        obj = getattr(module, name, None)
        origin = getattr(obj, "__module__", None)
        if obj is not None and origin and origin != module.__name__ and origin.startswith("llmlab"):
            borrowed.append(f"`{name}` из `{origin}`")
    if borrowed:
        pytest.fail("Импортированы готовые эталоны: " + ", ".join(borrowed) + ". Это ответы из llmlab, а задача — "
                    "написать их самому в solution.py (импортировать можно только то, что импортирует заготовка).",
                    pytrace=False)


@pytest.fixture(scope="module")
def task_info(request: pytest.FixtureRequest) -> dict:
    return _load_task_info(_find_task_dir(Path(request.path).parent))


@pytest.fixture(scope="module")
def impl(request: pytest.FixtureRequest) -> ModuleType:
    """Модуль, который тестируем (workspace / reference / starter / buggy / --impl-path)."""
    task_dir = _find_task_dir(Path(request.path).parent)
    mode = request.config.getoption("--impl")
    path = resolve_impl_path(task_dir, mode, request.config.getoption("--impl-path"))
    if mode == "workspace" and not path.exists() and not request.config.getoption("--impl-path"):
        tid = _load_task_info(task_dir)["id"]
        pytest.fail(f"Нет рабочей копии {path}. Создай её: `llmlab init-workspace {tid}`", pytrace=False)
    request.config.stash[_impl_path_key]["path"] = path.resolve()
    module = _load_module(path, mode)
    check_own_code(module, task_dir)
    return module


@pytest.fixture(scope="module")
def ref(request: pytest.FixtureRequest) -> ModuleType:
    """Эталонное решение задачи — для сравнения чисел."""
    task_dir = _find_task_dir(Path(request.path).parent)
    return _load_module(task_dir / "solution.py", "reference")


@pytest.fixture(autouse=True)
def _deterministic():
    """Фиксированный сид и детерминированные алгоритмы перед каждым тестом.

    torch импортируется лениво: тесты платформы на чистом Python не платят за его загрузку.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
    yield


# ---------------------------------------------------------------------------
# JSON-отчёт
# ---------------------------------------------------------------------------

def _clean_message(text: str) -> str:
    """Убирает хвост pytest-интроспекции ('assert ...'), если есть авторское сообщение."""
    lines = text.strip().splitlines()
    if lines and not lines[0].startswith("assert "):
        for i, line in enumerate(lines):
            if line.startswith("assert ") and i > 0:
                lines = lines[:i]
                break
    out = "\n".join(lines).strip()
    return out[:3000] + ("\n…(обрезано)" if len(out) > 3000 else "")


def _format_exception(excinfo: pytest.ExceptionInfo) -> str:
    exc = excinfo.value
    if isinstance(exc, NotImplementedError):
        detail = f": {exc}" if str(exc) else ""
        return f"Функция ещё не реализована (NotImplementedError{detail})."
    if isinstance(exc, AssertionError):
        return _clean_message(str(exc)) or "Проверка не прошла (assert без сообщения)."
    if excinfo.typename == "Failed":  # pytest.fail(...)
        return _clean_message(str(exc))
    return _clean_message(f"{excinfo.typename}: {exc}")


def _user_location(excinfo: pytest.ExceptionInfo, impl_path: Path | None) -> dict | None:
    """Последний кадр трейсбека внутри файла пользователя: где именно упало."""
    if impl_path is None:
        return None
    found = None
    for entry in excinfo.traceback:
        try:
            if Path(str(entry.path)).resolve() == impl_path:
                found = entry
        except (OSError, ValueError):
            continue
    if found is None:
        return None
    try:
        source = str(found.statement).strip()
    except Exception:  # noqa: BLE001 — исходник может быть недоступен
        source = ""
    return {"file": impl_path.name, "line": found.lineno + 1, "function": found.name, "source": source}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is not None and rep.failed:
        impl_path = item.config.stash.get(_impl_path_key, {}).get("path")
        rep.llmlab_message = _format_exception(call.excinfo)
        rep.llmlab_location = _user_location(call.excinfo, impl_path)


class _JsonReporter:
    """Собирает результаты в простой JSON (формат описан в docs/TASK_FORMAT.md)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tests: dict[str, dict] = {}
        self.collect_errors: list[dict] = []
        self.started = time.perf_counter()

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collect_errors.append({"nodeid": report.nodeid, "message": _clean_message(str(report.longrepr))})

    def pytest_itemcollected(self, item: pytest.Item) -> None:
        doc = getattr(getattr(item, "function", None), "__doc__", None) or ""
        self.tests[item.nodeid] = {
            "nodeid": item.nodeid,
            "name": item.name,
            "title": doc.strip().splitlines()[0] if doc.strip() else item.name,
            "outcome": "not-run",
            "message": None,
            "location": None,
            "duration_s": 0.0,
        }

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        entry = self.tests.setdefault(report.nodeid, {
            "nodeid": report.nodeid, "name": report.nodeid.split("::")[-1], "title": report.nodeid,
            "outcome": "not-run", "message": None, "location": None, "duration_s": 0.0,
        })
        entry["duration_s"] = round(entry["duration_s"] + report.duration, 4)
        if report.when == "call":
            entry["outcome"] = "passed" if report.passed else ("skipped" if report.skipped else "failed")
        elif report.when == "setup" and report.skipped:
            entry["outcome"] = "skipped"
        if report.failed:
            if report.when != "call":
                entry["outcome"] = "error"
            entry["message"] = getattr(report, "llmlab_message", None) or _clean_message(str(report.longrepr))
            entry["location"] = getattr(report, "llmlab_location", None)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        payload = {
            "exitstatus": int(exitstatus),
            "duration_s": round(time.perf_counter() - self.started, 3),
            "tests": list(self.tests.values()),
            "collect_errors": self.collect_errors,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
