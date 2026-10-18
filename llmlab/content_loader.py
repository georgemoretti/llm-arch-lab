"""Загрузка и валидация контента курса.

Источник правды — файлы в ``content/``:

- ``content/blocks/<B>/block.yaml``            — метаданные блока, модулей и базовых архитектур;
- ``content/blocks/<B>/<module>/<task>/task.yaml`` — задача;
- ``content/questions/*.yaml``                 — список вопросов;
- ``content/papers/*.yaml``                    — по одной статье в файле;
- ``content/viz/*.yaml``                       — по одной визуализации в файле (компонент — в web/src/viz);
- ``content/architectures.yaml``               — строки и столбцы таблицы архитектур (ячейки — в task.yaml → arch).

``load_course()`` никогда не падает на ошибках контента: он собирает их в ``Course.issues``,
чтобы ``llmlab validate`` мог показать всё сразу. Падает только на битом YAML/схеме,
если ``strict=True``.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llmlab.paths import content_dir

TaskType = Literal["implement", "debug", "paper", "quiz", "project"]
Difficulty = Literal["easy", "medium", "hard"]

# Какие файлы обязательны для каждого типа задачи.
REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "implement": ("task.yaml", "README.md", "starter.py", "solution.py", "test_task.py", "hints.md", "summary.md"),
    "project": ("task.yaml", "README.md", "starter.py", "solution.py", "test_task.py", "hints.md", "summary.md"),
    "debug": ("task.yaml", "README.md", "buggy.py", "solution.py", "test_task.py", "hints.md", "summary.md"),
    "paper": ("task.yaml", "README.md"),
    "quiz": ("task.yaml", "README.md"),
}

# Типы задач, у которых есть код и тесты.
CODE_TASK_TYPES = frozenset({"implement", "debug", "project"})

# Директива вставки визуализации в README: строка ``:::viz rope-rotation:::``.
VIZ_DIRECTIVE_RE = re.compile(r"^:::viz\s+([a-z0-9][a-z0-9-]*)\s*:::\s*$", re.MULTILINE)

# Схема форм тензоров в README: блок ```shapes, рисует web/src/components/ShapeFlow.tsx.
SHAPES_BLOCK_RE = re.compile(r"^```shapes[ \t]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
AXIS_KINDS = {"batch", "seq", "model", "heads", "hdim", "wide", "other", "num"}


def check_shapes(text: str) -> list[str]:
    """Ошибки синтаксиса блоков ```shapes (грамматика — в шапке ShapeFlow.tsx и в TASK_FORMAT.md)."""
    errors: list[str] = []
    for block_no, m in enumerate(SHAPES_BLOCK_RE.finditer(text), 1):
        in_fork = False
        tensors = 0
        for line_no, raw in enumerate(m.group(1).splitlines(), 1):
            line = raw.strip()
            at = f"shapes #{block_no}, строка {line_no} «{line[:40]}»"
            if not line or line.startswith("//"):
                continue
            if line.startswith("axis") and not line.startswith("axis "):
                errors.append(f"{at}: после axis нужен пробел")
            elif line.startswith("axis "):
                am = re.fullmatch(r"axis\s+([^\s:=]+)(?::(\w+))?\s*=\s*(.+)", line)
                if not am:
                    errors.append(f"{at}: ожидается «axis ИМЯ = смысл» или «axis ИМЯ:вид = смысл»")
                elif am.group(2) and am.group(2) not in AXIS_KINDS:
                    errors.append(f"{at}: неизвестный вид оси '{am.group(2)}' (есть: {', '.join(sorted(AXIS_KINDS))})")
            elif line == "fork":
                if in_fork:
                    errors.append(f"{at}: fork внутри fork — сначала закрой предыдущий через join")
                in_fork = True
            elif line == "branch" or line == "join" or line.startswith("join "):
                if not in_fork:
                    errors.append(f"{at}: {line.split()[0]} без fork")
                if line != "branch":
                    in_fork = False
            elif line.startswith(("#", ">", "-")) and not line.startswith(("## ", "> ", "- ")):
                errors.append(f"{at}: после «{line[0] * (2 if line.startswith('##') else 1)}» нужен пробел")
            elif not line.startswith(("## ", "> ", "- ")):
                tensors += 1
                body = line.split(" — ", 1)[0]
                if body.count("[") != body.count("]"):
                    errors.append(f"{at}: несбалансированные квадратные скобки")
                elif not re.search(r"(^|\s)\[", body):
                    errors.append(f"{at}: тензор без формы [..] — если это операция, начни строку с «> », "
                                  "если пояснение — с «- »")
        if in_fork:
            errors.append(f"shapes #{block_no}: fork не закрыт строкой join")
        if tensors == 0:
            errors.append(f"shapes #{block_no}: в схеме нет ни одного тензора")
    return errors

TASK_ID_RE = re.compile(r"^[A-Z]\d{2}[a-z]?(-[a-z0-9]+)+$|^[A-Z]-[A-Z]\d+[a-z]?(-[a-z0-9]+)*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadingLink(_Strict):
    title: str
    url: str


class Task(_Strict):
    id: str
    title: str
    # Короткое имя для навигации (граф, хлебные крошки, «после: …»); по умолчанию — из title.
    short_title: str | None = None
    block: str
    module: str
    type: TaskType
    difficulty: Difficulty
    depends_on: list[str] = Field(default_factory=list)
    diff_from: str | None = None
    tags: list[str] = Field(default_factory=list)
    entrypoint: str | list[str] | None = None
    time_estimate_min: int = Field(ge=5, le=120)
    papers: list[str] = Field(default_factory=list)
    reading: list[ReadingLink] = Field(default_factory=list)
    viz: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    # Ячейки таблицы архитектур, которые открывает решение задачи: {модель: {столбец: значение}}.
    arch: dict[str, dict[str, str]] = Field(default_factory=dict)

    # Заполняется загрузчиком, не берётся из YAML.
    dir: Path = Field(default=Path("."), exclude=True)

    @property
    def entrypoints(self) -> list[str]:
        if self.entrypoint is None:
            return []
        return [self.entrypoint] if isinstance(self.entrypoint, str) else list(self.entrypoint)

    @property
    def has_code(self) -> bool:
        return self.type in CODE_TASK_TYPES

    @property
    def nav_title(self) -> str:
        """short_title или title без служебного префикса «Отладка: » / «Проект: »."""
        if self.short_title:
            return self.short_title
        return re.sub(r"^(Отладка|Проект|Статьи?|Квиз модуля):\s*", "", self.title)

    @property
    def starter_file(self) -> Path:
        """Файл, с которого пользователь начинает: заготовка или багованный код."""
        return self.dir / ("buggy.py" if self.type == "debug" else "starter.py")

    @property
    def solution_file(self) -> Path:
        return self.dir / "solution.py"

    @property
    def test_file(self) -> Path:
        return self.dir / "test_task.py"


class Question(_Strict):
    id: str
    topic: str
    tasks: list[str] = Field(default_factory=list)
    difficulty: Difficulty = "medium"
    kind: Literal["open", "mcq", "code-reading"] = "open"
    question: str
    answer: str
    options: list[str] = Field(default_factory=list)
    correct: int | None = None    # mcq: индекс верного варианта в options (с нуля)
    source: str = "course"
    tags: list[str] = Field(default_factory=list)        # подтема: rag, agents, sampling, … (для фильтров)
    source_ref: list[str] = Field(default_factory=list)  # user-db: хэши сырых вопросов (llmlab questions)

    @model_validator(mode="after")
    def _check_mcq(self) -> "Question":
        if self.kind == "mcq":
            if len(self.options) < 2:
                raise ValueError("mcq: нужно хотя бы 2 варианта в options")
            if self.correct is None or not 0 <= self.correct < len(self.options):
                raise ValueError("mcq: correct — индекс верного варианта в options (с нуля)")
        elif self.correct is not None:
            raise ValueError("correct задаётся только для kind: mcq")
        return self


class Paper(_Strict):
    id: str
    title: str
    authors: str
    year: int
    url: str
    venue: str | None = None
    tags: list[str] = Field(default_factory=list)


class Viz(_Strict):
    """Описание визуализации. Сам интерактивный компонент — ``web/src/viz/<id>``."""

    id: str
    title: str
    description: str = ""


class ArchColumn(_Strict):
    id: str
    title: str


class ArchModel(_Strict):
    id: str
    title: str
    year: int | None = None
    module: str | None = None   # в каком модуле курса строка начнёт заполняться
    source: str | None = None   # откуда брать факты: config.json на HF или статья


class Architectures(_Strict):
    """Таблица «модель × компонент». Значения ячеек задают задачи (поле ``arch``)."""

    columns: list[ArchColumn]
    models: list[ArchModel]


class Module(_Strict):
    id: str
    title: str
    description: str = ""

    def number(self, position: int) -> int:
        """Номер модуля в навигации: из id «module-<N>-…», иначе позиция в block.yaml."""
        m = re.match(r"module-(\d+)(?:-|$)", self.id)
        return int(m.group(1)) if m else position


class Baseline(_Strict):
    title: str
    path: str | None = None
    description: str = ""


class Block(_Strict):
    id: str
    title: str
    description: str = ""
    # Необязательный блок (фундамент A): карта его помечает и не предлагает «следующей» тем, кто уже идёт по
    # обязательным блокам.
    optional: bool = False
    modules: list[Module] = Field(default_factory=list)
    baselines: dict[str, Baseline] = Field(default_factory=dict)


@dataclass
class Issue:
    level: Literal["error", "warning"]
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


@dataclass
class Course:
    root: Path
    blocks: dict[str, Block] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    questions: dict[str, Question] = field(default_factory=dict)
    papers: dict[str, Paper] = field(default_factory=dict)
    viz: dict[str, Viz] = field(default_factory=dict)
    architectures: Architectures | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    def task(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError:
            close = [t for t in self.tasks if t.startswith(task_id.split("-")[0])]
            hint = f" Возможно, ты имел в виду: {', '.join(close)}." if close else ""
            raise KeyError(f"Задача '{task_id}' не найдена.{hint}") from None

    def questions_for(self, task: Task) -> list[Question]:
        """Вопросы задачи: по глобам из task.yaml и по явной привязке question.tasks."""
        out = []
        for q in self.questions.values():
            if task.id in q.tasks or any(fnmatch.fnmatchcase(q.id, pat) for pat in task.questions):
                out.append(q)
        return out

    def labels(self) -> dict[str, dict]:
        """Номера для навигации вместо id: «C1.3» = блок C, модуль 1, шаг 3.

        Нумеруются implement/project в порядке topo_order внутри модуля (номер модуля — из id
        «module-<N>-…», см. `Module.number`). Буква блока нужна, потому что «1.3» есть в каждом блоке.
        debug получает номер и ссылку `parent` на задачу, которую чинит (первая implement-зависимость);
        paper/quiz — без номера.
        """
        module_no = {(b.id, m.id): f"{b.id}{m.number(i)}" for b in self.blocks.values() for i, m in enumerate(b.modules)}
        counters: dict[tuple[str, str], int] = {}
        out: dict[str, dict] = {}
        order = self.topo_order()
        for tid in order:
            t = self.tasks[tid]
            key = (t.block, t.module)
            if t.type in ("implement", "project") and key in module_no:
                counters[key] = counters.get(key, 0) + 1
                out[tid] = {"number": f"{module_no[key]}.{counters[key]}", "parent": None}
            else:
                out[tid] = {"number": None, "parent": None}
        for tid in order:
            t = self.tasks[tid]
            if t.type == "debug":
                parent = next((d for d in t.depends_on if d in self.tasks and self.tasks[d].type == "implement"), None)
                out[tid] = {"number": out[parent]["number"] if parent else None, "parent": parent}
        return out

    def topo_order(self) -> list[str]:
        """Задачи в порядке, где зависимости идут раньше (стабильно по id)."""
        order: list[str] = []
        seen: set[str] = set()

        def visit(tid: str) -> None:
            if tid in seen or tid not in self.tasks:
                return
            seen.add(tid)
            for dep in sorted(self.tasks[tid].depends_on, key=lambda t: (t in self.tasks and self.tasks[t].type in ("paper", "quiz"), t)):
                visit(dep)
            order.append(tid)

        # paper/quiz — в конце модуля; иначе «C-P1» (дефис < цифры) встал бы перед «C01».
        for tid in sorted(self.tasks, key=lambda t: (self.tasks[t].type in ("paper", "quiz"), t)):
            visit(tid)
        return order


def _read_yaml(path: Path, course: Course):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        course.issues.append(Issue("error", str(path), f"битый YAML: {exc}"))
        return None


def _schema_error(path: Path, exc: ValidationError, course: Course) -> None:
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        course.issues.append(Issue("error", str(path), f"{loc}: {err['msg']}"))


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_course(root: Path | None = None) -> Course:
    cdir = content_dir(root)
    course = Course(root=cdir.parent)
    blocks_dir = cdir / "blocks"

    # --- блоки и задачи ---
    for block_dir in sorted(p for p in blocks_dir.glob("*") if p.is_dir()):
        block_yaml = block_dir / "block.yaml"
        if block_yaml.exists():
            data = _read_yaml(block_yaml, course)
            if data is not None:
                try:
                    block = Block.model_validate(data)
                    if block.id != block_dir.name:
                        course.issues.append(Issue("error", _rel(block_yaml, course.root),
                                                   f"id '{block.id}' не совпадает с папкой '{block_dir.name}'"))
                    course.blocks[block.id] = block
                except ValidationError as exc:
                    _schema_error(block_yaml, exc, course)
        else:
            course.issues.append(Issue("warning", _rel(block_dir, course.root), "нет block.yaml"))

        for task_yaml in sorted(block_dir.glob("*/*/task.yaml")):
            data = _read_yaml(task_yaml, course)
            if data is None:
                continue
            try:
                task = Task.model_validate({**data, "dir": task_yaml.parent})
            except ValidationError as exc:
                _schema_error(task_yaml, exc, course)
                continue
            where = _rel(task_yaml, course.root)
            if task.id in course.tasks:
                course.issues.append(Issue("error", where, f"дублирующийся id '{task.id}'"))
                continue
            if task.id != task_yaml.parent.name:
                course.issues.append(Issue("error", where, f"id '{task.id}' не совпадает с папкой '{task_yaml.parent.name}'"))
            if task.block != block_dir.name:
                course.issues.append(Issue("error", where, f"block '{task.block}' не совпадает с папкой '{block_dir.name}'"))
            if task.module != task_yaml.parent.parent.name:
                course.issues.append(Issue("error", where, f"module '{task.module}' не совпадает с папкой '{task_yaml.parent.parent.name}'"))
            if not TASK_ID_RE.match(task.id):
                course.issues.append(Issue("warning", where, f"id '{task.id}' не похож на формат 'C03-rope' / 'C01d-rmsnorm-debug'"))
            for fname in REQUIRED_FILES[task.type]:
                if not (task.dir / fname).exists():
                    course.issues.append(Issue("error", where, f"нет обязательного файла {fname} (type: {task.type})"))
            if task.has_code and not task.entrypoints:
                course.issues.append(Issue("error", where, "у задачи с кодом должен быть entrypoint"))
            course.tasks[task.id] = task

    # --- вопросы ---
    qdir = cdir / "questions"
    for qfile in sorted(qdir.glob("*.yaml")) if qdir.is_dir() else []:
        data = _read_yaml(qfile, course)
        if data is None:
            continue
        if not isinstance(data, list):
            course.issues.append(Issue("error", _rel(qfile, course.root), "файл вопросов должен быть YAML-списком"))
            continue
        for item in data:
            try:
                q = Question.model_validate(item)
            except ValidationError as exc:
                _schema_error(qfile, exc, course)
                continue
            if q.id in course.questions:
                course.issues.append(Issue("error", _rel(qfile, course.root), f"дублирующийся id вопроса '{q.id}'"))
            course.questions[q.id] = q

    # --- статьи ---
    pdir = cdir / "papers"
    for pfile in sorted(pdir.glob("*.yaml")) if pdir.is_dir() else []:
        data = _read_yaml(pfile, course)
        if data is None:
            continue
        try:
            paper = Paper.model_validate(data)
        except ValidationError as exc:
            _schema_error(pfile, exc, course)
            continue
        course.papers[paper.id] = paper

    # --- визуализации ---
    vdir = cdir / "viz"
    for vfile in sorted(vdir.glob("*.yaml")) if vdir.is_dir() else []:
        data = _read_yaml(vfile, course)
        if data is None:
            continue
        try:
            viz = Viz.model_validate(data)
        except ValidationError as exc:
            _schema_error(vfile, exc, course)
            continue
        if viz.id != vfile.stem:
            course.issues.append(Issue("error", _rel(vfile, course.root), f"id '{viz.id}' не совпадает с именем файла"))
        course.viz[viz.id] = viz

    # --- таблица архитектур ---
    afile = cdir / "architectures.yaml"
    if afile.exists():
        data = _read_yaml(afile, course)
        if data is not None:
            try:
                course.architectures = Architectures.model_validate(data)
            except ValidationError as exc:
                _schema_error(afile, exc, course)

    _check_references(course)
    return course


def _check_references(course: Course) -> None:
    baselines = {name for b in course.blocks.values() for name in b.baselines}
    # Interview и таблица архитектур ищут модуль по id без блока — id должны быть уникальны во всём курсе.
    seen_modules: dict[str, str] = {}
    for b in course.blocks.values():
        for m in b.modules:
            if m.id in seen_modules:
                course.issues.append(Issue("error", f"content/blocks/{b.id}/block.yaml",
                                           f"модуль '{m.id}' уже есть в блоке {seen_modules[m.id]}: id модулей уникальны во всём курсе"))
            seen_modules.setdefault(m.id, b.id)
    for task in course.tasks.values():
        where = _rel(task.dir / "task.yaml", course.root)
        block = course.blocks.get(task.block)
        if block is not None and task.module not in {m.id for m in block.modules}:
            course.issues.append(Issue("error", where, f"модуль '{task.module}' не описан в block.yaml"))
        for dep in task.depends_on:
            if dep not in course.tasks:
                course.issues.append(Issue("error", where, f"depends_on: неизвестная задача '{dep}'"))
        if task.diff_from and task.diff_from not in course.tasks and task.diff_from not in baselines:
            course.issues.append(Issue("error", where,
                                       f"diff_from: '{task.diff_from}' — ни задача, ни baseline из block.yaml"))
        for pid in task.papers:
            if pid not in course.papers:
                course.issues.append(Issue("error", where, f"papers: нет content/papers/{pid}.yaml"))
        readme = task.dir / "README.md"
        readme_text = readme.read_text(encoding="utf-8") if readme.exists() else ""
        directives = VIZ_DIRECTIVE_RE.findall(readme_text)
        for err in check_shapes(readme_text):
            course.issues.append(Issue("error", _rel(readme, course.root), err))
        for vid in [*task.viz, *directives]:
            if vid not in course.viz:
                course.issues.append(Issue("error", where, f"viz: нет content/viz/{vid}.yaml"))
        for pat in task.questions:
            if not any(fnmatch.fnmatchcase(qid, pat) for qid in course.questions):
                course.issues.append(Issue("warning", where, f"questions: глоб '{pat}' не нашёл ни одного вопроса"))

    _check_arch(course)

    for q in course.questions.values():
        for tid in q.tasks:
            if tid not in course.tasks:
                course.issues.append(Issue("warning", f"question {q.id}", f"tasks: неизвестная задача '{tid}'"))

    for cycle in find_cycles({t.id: t.depends_on for t in course.tasks.values()}):
        course.issues.append(Issue("error", "depends_on", "цикл зависимостей: " + " → ".join(cycle)))


def _check_arch(course: Course) -> None:
    arch = course.architectures
    models = {m.id for m in arch.models} if arch else set()
    columns = {c.id for c in arch.columns} if arch else set()
    owner: dict[tuple[str, str], str] = {}
    for task in course.tasks.values():
        where = _rel(task.dir / "task.yaml", course.root)
        for model, cells in task.arch.items():
            if model not in models:
                course.issues.append(Issue("error", where, f"arch: модели '{model}' нет в content/architectures.yaml"))
                continue
            for col in cells:
                if col not in columns:
                    course.issues.append(Issue("error", where, f"arch: столбца '{col}' нет в content/architectures.yaml"))
                elif (model, col) in owner:
                    course.issues.append(Issue("error", where,
                                               f"arch: ячейку {model}.{col} уже открывает {owner[model, col]}"))
                else:
                    owner[model, col] = task.id
    if arch:
        block_modules = {m.id for b in course.blocks.values() for m in b.modules}
        for m in arch.models:
            if m.module and m.module not in block_modules:
                course.issues.append(Issue("error", "content/architectures.yaml",
                                           f"модель {m.id}: неизвестный модуль '{m.module}'"))


def find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """Находит циклы в графе зависимостей (DFS с цветами). Каждый цикл — один раз."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    stack: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        color[node] = GRAY
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt not in color:
                continue
            if color[nxt] == GRAY:
                cycles.append(stack[stack.index(nxt):] + [nxt])
            elif color[nxt] == WHITE:
                dfs(nxt)
        stack.pop()
        color[node] = BLACK

    for node in sorted(graph):
        if color[node] == WHITE:
            dfs(node)
    return cycles
