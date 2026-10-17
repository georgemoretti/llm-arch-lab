"""Генерация конспекта задачи: ``notes/<block>/<task_id>.md``.

Конспект собирается из четырёх источников:

- ``summary.md`` задачи — авторские TL;DR, формула, формы тензоров, «что поменялось»;
- рабочая копия пользователя — финальный код и его diff с эталоном;
- история попыток в SQLite — раздел «Мои ошибки по пути»;
- ввод пользователя — «Главное, что я понял» (обязательное поле, 1–3 строки).

Файл совместим с Obsidian: YAML-frontmatter, ``[[wiki-ссылки]]`` на родителя (``diff_from``) и
зависимости, сворачиваемые callout-блоки (``> [!example]-``).

Пересборка (``llmlab notes build``) перезаписывает всё, кроме «Главное, что я понял»: если
пользователь поправил этот раздел прямо в файле (например, в Obsidian), побеждает версия из
файла, и она же сохраняется в БД.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from llmlab.content_loader import Course, Task
from llmlab.paths import db_path, notes_dir
from llmlab.runner import db
from llmlab.workspace import workspace_file

INSIGHT_HEADING = "Главное, что я понял"
INSIGHT_MAX_LINES = 3
INSIGHT_MAX_CHARS = 600
INDEX_NOTE = "index.md"  # оглавление vault'а, не конспект задачи

# Сообщения, которые не несут информации об ошибке (заготовка ещё не тронута).
_NOISE_PREFIXES = ("Функция ещё не реализована",)
# Сколько разных ошибок показывать в конспекте — дальше это уже шум.
MAX_MISTAKES = 12


class NoteError(Exception):
    """Конспект нельзя собрать. ``code`` — машинный код причины для API/CLI."""

    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- пути и парсинг


def note_relpath(task: Task) -> Path:
    return Path(task.block) / f"{task.id}.md"


def note_file(task: Task, root: Path | None = None) -> Path:
    return notes_dir(root) / note_relpath(task)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """``---\\nyaml\\n---\\nbody`` → (meta, body). Без frontmatter → ({}, text)."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            try:
                meta = yaml.safe_load(text[4:end]) or {}
            except yaml.YAMLError:
                meta = {}
            return (meta if isinstance(meta, dict) else {}), text[end + 5:]
    return {}, text


def parse_sections(markdown: str) -> dict[str, str]:
    """Разделы по заголовкам ``## …`` → {заголовок: текст}. Порядок сохраняется."""
    parts = re.split(r"^##[ \t]+(.+?)[ \t]*$", markdown, flags=re.MULTILINE)
    return {parts[i].strip(): parts[i + 1].strip("\n") for i in range(1, len(parts) - 1, 2)}


def validate_insight(text: str) -> str:
    """Нормализует «Главное, что я понял»: 1–3 непустые строки, не длиннее 600 символов."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        raise NoteError("Поле «Главное, что я понял» обязательно: напиши 1–3 строки своими словами.",
                        "insight_empty")
    if len(lines) > INSIGHT_MAX_LINES:
        raise NoteError(f"«Главное, что я понял» — не больше {INSIGHT_MAX_LINES} строк (сейчас {len(lines)}). "
                        "Оставь самое важное.", "insight_too_long")
    joined = "\n".join(lines)
    if len(joined) > INSIGHT_MAX_CHARS:
        raise NoteError(f"«Главное, что я понял» — не больше {INSIGHT_MAX_CHARS} символов.", "insight_too_long")
    return joined


def insight_from_file(path: Path) -> str | None:
    """Пользовательский раздел из уже существующего конспекта (мог быть отредактирован руками)."""
    if not path.exists():
        return None
    _, body = split_frontmatter(path.read_text(encoding="utf-8"))
    text = parse_sections(body).get(INSIGHT_HEADING, "")
    # В файле вывод записан списком («- строка»), но руками его могли переписать как угодно.
    lines = [re.sub(r"^[-*+]\s+", "", ln.strip()) for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines) or None


# ---------------------------------------------------------------- история попыток


@dataclass
class Mistake:
    """Одна и та же ошибка (с точностью до чисел), сгруппированная по попыткам и тестам."""

    message: str
    tests: list[str] = field(default_factory=list)
    attempts: set[int] = field(default_factory=set)
    location: dict | None = None


def _norm(message: str) -> str:
    # Числа в сообщениях меняются от попытки к попытке (max |Δ| = 2.02 → 1.7) — это та же ошибка.
    return re.sub(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?", "#", message.strip().lower())


def collect_mistakes(attempts: list[dict]) -> list[Mistake]:
    """Падения тестов из попыток → уникальные ошибки в порядке первого появления."""
    groups: dict[str, Mistake] = {}
    for a in attempts:
        if a["status"] == "passed":
            continue
        result = a["result"]
        entries: list[tuple[str, str | None, dict | None]] = []
        if result.get("message") and a["status"] in ("error", "timeout"):
            entries.append((result["message"], None, None))
        for t in result.get("tests", []):
            if t["outcome"] in ("failed", "error") and t.get("message"):
                entries.append((t["message"], t.get("title") or t["name"], t.get("location")))
        for message, test, loc in entries:
            if message.startswith(_NOISE_PREFIXES):
                continue
            m = groups.setdefault(_norm(message), Mistake(message=message.strip()))
            m.attempts.add(a["id"])
            if test and test not in m.tests:
                m.tests.append(test)
            if m.location is None and loc:
                m.location = loc
    return list(groups.values())


def _workspace_attempts(task: Task, dbfile: Path) -> list[dict]:
    import json

    out = []
    for row in db.attempts(task.id, dbfile):
        if row["impl"] != "workspace":
            continue
        row["result"] = json.loads(row.pop("result_json"))
        out.append(row)
    return out


def _local_date(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().date().isoformat()


def _minutes_between(a: str, b: str) -> int:
    return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60)


# ---------------------------------------------------------------- рендер


def _quote(text: str, prefix: str = "> ") -> str:
    return "\n".join(prefix + ln if ln else prefix.rstrip() for ln in text.splitlines())


def _wiki(task_id: str) -> str:
    return f"[[{task_id}]]"


def _render_mistakes(attempts: list[dict], first_pass: int) -> str:
    n_before = first_pass  # индекс первой зелёной попытки = число неудачных до неё
    lines = []
    if n_before == 0:
        lines.append("Решено с первой попытки.")
    else:
        mins = _minutes_between(attempts[0]["created_at"], attempts[first_pass]["created_at"])
        took = f", от первого запуска до решения — {mins} мин" if mins > 0 else ""
        failed = _plural(n_before, "неудачная", "неудачные", "неудачных")
        lines.append(f"Решено с {first_pass + 1}-й попытки ({n_before} {failed}{took}).")

    mistakes = collect_mistakes(attempts)
    if not mistakes:
        if n_before:
            lines.append("")
            lines.append("Тесты падали только потому, что код ещё не был написан — содержательных ошибок нет.")
        return "\n".join(lines)

    lines.append("")
    for m in mistakes[:MAX_MISTAKES]:
        if not m.tests:
            label = "Прогон целиком"
        elif len(m.tests) <= 3:
            label = "; ".join(m.tests)
        else:
            label = f"{len(m.tests)} тестов: " + "; ".join(m.tests[:2]) + "; …"
        times = len(m.attempts)
        lines.append(f"- **{label}** — {times} {_plural(times, 'попытка', 'попытки', 'попыток')}")
        lines.append(_quote(m.message, "  > "))
        if m.location:
            loc = m.location
            src = f"`{loc['source']}` — " if loc.get("source") else ""
            lines.append(f"  >\n  > {src}{loc['file']}:{loc['line']}, `{loc['function']}`")
    if len(mistakes) > MAX_MISTAKES:
        lines.append(f"- …и ещё {len(mistakes) - MAX_MISTAKES} разных ошибок.")
    return "\n".join(lines)


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _render_diff(code: str, reference: str) -> str:
    diff = list(difflib.unified_diff(reference.splitlines(), code.splitlines(),
                                     "эталон", "мой код", lineterm="", n=2))
    if not diff:
        return "Код совпадает с эталоном."
    added = sum(ln.startswith("+") and not ln.startswith("+++") for ln in diff)
    removed = sum(ln.startswith("-") and not ln.startswith("---") for ln in diff)
    body = "```diff\n" + "\n".join(diff) + "\n```"
    return f"> [!example]- Diff: эталон → мой код (+{added} −{removed})\n" + _quote(body)


# Ссылка на карточку вопроса в локальном UI (порт по умолчанию `llmlab serve`).
INTERVIEW_URL = "http://127.0.0.1:8765/#/interview/q/"

def _render_questions(course: Course, task: Task) -> str:
    qs = course.questions_for(task)
    if not qs:
        return "Вопросов к этой задаче пока нет."
    out = []
    for q in qs:
        title = " ".join(q.question.split())
        body = q.answer.strip()
        if q.kind == "mcq":
            opts = "\n".join(f"- {'✅ ' if i == q.correct else ''}{o}" for i, o in enumerate(q.options))
            body = f"{opts}\n\n{body}"
        src = " · с собеседования" if q.source == "user-db" else ""
        out.append(f"> [!question]- {title}\n" + _quote(body)
                   + f"\n>\n> `{q.id}` · {q.difficulty}{src} · [повторить в Interview]({INTERVIEW_URL}{q.id})")
    return "\n\n".join(out)


def _render_reading(course: Course, task: Task) -> str:
    out = []
    for pid in task.papers:
        p = course.papers.get(pid)
        if p:
            out.append(f"- [{p.title}]({p.url}) — {p.authors}, {p.year}")
    out += [f"- [{r.title}]({r.url})" for r in task.reading]
    return "\n".join(out) or "—"


def render_note(course: Course, task: Task, *, insight: str, code: str, attempts: list[dict],
                now: datetime | None = None) -> str:
    """Текст конспекта. ``attempts`` — попытки workspace до последней зелёной включительно."""
    now = now or datetime.now(timezone.utc)
    first_pass = next(i for i, a in enumerate(attempts) if a["status"] == "passed")
    summary = parse_sections(task.dir.joinpath("summary.md").read_text(encoding="utf-8")) \
        if (task.dir / "summary.md").exists() else {}

    parent = None
    if task.diff_from:
        parent = _wiki(task.diff_from) if task.diff_from in course.tasks else task.diff_from
    meta = {
        "id": task.id,
        "title": task.title,
        "aliases": [task.title],
        "block": task.block,
        "module": task.module,
        "type": task.type,
        "difficulty": task.difficulty,
        "tags": list(task.tags),
        "solved_at": _local_date(attempts[first_pass]["created_at"]),
        "attempts": first_pass + 1,
        "parent": parent,
        "depends_on": [_wiki(d) for d in task.depends_on],
        "updated": now.astimezone().isoformat(timespec="seconds"),
        "source": "llmlab",
    }
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000).strip()

    links = []
    if task.diff_from in course.tasks:
        links.append(f"Родитель: {_wiki(task.diff_from)}")
    elif task.diff_from:
        baselines = {k: v for b in course.blocks.values() for k, v in b.baselines.items()}
        title = baselines[task.diff_from].title if task.diff_from in baselines else task.diff_from
        links.append(f"Отправная точка: {title}")
    if task.depends_on:
        links.append("Опирается на: " + ", ".join(_wiki(d) for d in task.depends_on))
    dependents = sorted(t.id for t in course.tasks.values() if task.id in t.depends_on)
    if dependents:
        links.append("Дальше: " + ", ".join(_wiki(d) for d in dependents))

    parts = [f"---\n{front}\n---", f"# {task.title}"]
    if links:
        parts.append(" · ".join(links))

    # Авторские разделы из summary.md: четыре обязательных в фиксированном порядке, заголовки как у автора;
    # дополнительные (например, карта «деталь → задача» у C25) — следом, в порядке автора.
    changed_key = next((k for k in summary if k.startswith("Что поменялось")), None)
    standard = ("TL;DR", "Формула", "Формы тензоров", changed_key)
    for key in [*standard, *(k for k in summary if k not in standard)]:
        if key and summary.get(key, "").strip():
            parts.append(f"## {key}\n\n{summary[key].strip()}")

    reference = task.solution_file.read_text(encoding="utf-8")
    parts.append(f"## Мой код\n\n```python\n{code.rstrip()}\n```")
    parts.append(f"## Отличия от эталона\n\n{_render_diff(code, reference)}")
    parts.append(f"## Мои ошибки по пути\n\n{_render_mistakes(attempts, first_pass)}")
    # Списком: так каждая строка вывода остаётся отдельной в любом Markdown-рендерере.
    insight_md = "\n".join(f"- {ln}" for ln in insight.strip().splitlines())
    parts.append(f"## {INSIGHT_HEADING}\n\n{insight_md}")
    parts.append(f"## Вопросы для повторения\n\n{_render_questions(course, task)}")
    parts.append(f"## Читать\n\n{_render_reading(course, task)}")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------- сборка


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _resolve_code(task: Task, root: Path, attempts: list[dict], stored: dict | None) -> str:
    """Финальный код: рабочая копия, если именно она проходила тесты, иначе сохранённый снимок."""
    passed_shas = {a["code_sha256"] for a in attempts if a["status"] == "passed"}
    ws = workspace_file(task, root)
    if ws.exists():
        text = ws.read_text(encoding="utf-8")
        if hashlib.sha256(ws.read_bytes()).hexdigest() in passed_shas:
            return text
    if stored and stored.get("code"):
        return stored["code"]
    if ws.exists():
        return ws.read_text(encoding="utf-8")
    raise NoteError(f"Нет кода решения {task.id}: workspace/{task.id}/solution.py не найден.", "no_code")


def build_note(course: Course, task: Task, root: Path | None = None, insight: str | None = None,
               now: datetime | None = None) -> dict:
    """Создаёт или пересобирает конспект. Возвращает {task_id, path, created}.

    ``insight=None`` — взять «Главное, что я понял» из файла или БД (пересборка); если его
    нигде нет, ``NoteError(code="needs_insight")``.
    """
    root = root or course.root
    dbfile = db_path(root)
    if not task.has_code:
        raise NoteError(f"У задачи {task.id} (type: {task.type}) нет тестов — конспект не создаётся.", "no_tests")

    attempts = _workspace_attempts(task, dbfile)
    last_pass = max((i for i, a in enumerate(attempts) if a["status"] == "passed"), default=None)
    if last_pass is None:
        raise NoteError(f"Задача {task.id} ещё не решена: конспект появится после первого зелёного прогона.",
                        "not_solved")
    attempts = attempts[: last_pass + 1]

    path = note_file(task, root)
    stored = db.get_note(task.id, dbfile)
    if insight is not None:
        final_insight = validate_insight(insight)
    else:
        final_insight = insight_from_file(path) or (stored or {}).get("insight")
        if not final_insight:
            raise NoteError(f"Для конспекта {task.id} нужен вывод «Главное, что я понял» (1–3 строки).",
                            "needs_insight")

    code = _resolve_code(task, root, attempts, stored)
    text = render_note(course, task, insight=final_insight, code=code, attempts=attempts, now=now)
    _write_atomic(path, text)
    db.upsert_note(task.id, insight=final_insight, code=code, code_sha256=_sha(code),
                   note_path=note_relpath(task).as_posix(),
                   now=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), path=dbfile)
    write_index_note(course, root)
    return {"task_id": task.id, "path": note_relpath(task).as_posix(), "created": stored is None}


def build_all(course: Course, root: Path | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """Пересобирает все конспекты, для которых есть вывод пользователя.

    Возвращает (собранные, [(task_id, причина пропуска)]). Нерешённые задачи молча пропускаются.
    """
    root = root or course.root
    built, skipped = [], []
    for tid in course.topo_order():
        task = course.tasks[tid]
        if not task.has_code:
            continue
        try:
            built.append(build_note(course, task, root))
        except NoteError as exc:
            if exc.code != "not_solved":
                skipped.append((tid, str(exc)))
    write_index_note(course, root)
    return built, skipped


def write_index_note(course: Course, root: Path | None = None) -> Path:
    """``notes/index.md`` — оглавление vault'а и открытая часть таблицы архитектур."""
    from llmlab.notes.arch import arch_table

    root = root or course.root
    ndir = notes_dir(root)
    existing = {p.stem for p in ndir.glob("*/*.md")} if ndir.is_dir() else set()
    lines = ["---", "title: LLM Arch Lab — конспекты", "source: llmlab", "---",
             "# LLM Arch Lab — конспекты", ""]
    code_tasks = [course.tasks[t] for t in course.topo_order() if course.tasks[t].has_code]
    lines.append(f"Конспектов: {sum(t.id in existing for t in code_tasks)} из {len(code_tasks)} задач с кодом. "
                 "Файл пересобирается автоматически.")
    for block in course.blocks.values():
        lines += ["", f"## {block.id} — {block.title}"]
        for m in block.modules:
            notes = [t for t in code_tasks if t.block == block.id and t.module == m.id and t.id in existing]
            if notes:
                lines += ["", f"### {m.title}", ""]
                lines += [f"- {_wiki(t.id)} — {t.title}" for t in notes]

    statuses = db.task_statuses(db_path(root))
    table = arch_table(course, solved={t for t, st in statuses.items() if st["status"] == "solved"})
    if table["rows"]:
        lines += ["", "## Таблица архитектур", "",
                  "| Модель | " + " | ".join(c["title"] for c in table["columns"]) + " |",
                  "|---" * (len(table["columns"]) + 1) + "|"]
        for row in table["rows"]:
            cells = []
            for c in table["columns"]:
                cell = row["cells"][c["id"]]
                cells.append(cell["value"].replace("|", "\\|") if cell["status"] == "open"
                             else ("🔒 " + cell["task_id"] if cell["status"] == "locked" else "·"))
            lines.append(f"| {row['title']} | " + " | ".join(cells) + " |")
        lines += ["", "🔒 — откроется после решения задачи; · — появится в следующих модулях курса."]
    path = ndir / INDEX_NOTE
    _write_atomic(path, "\n".join(lines) + "\n")
    return path
