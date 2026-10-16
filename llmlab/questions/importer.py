"""Импорт сырой базы вопросов с собеседований (``data/interview_raw/``) — детерминированная часть.

Конвейер (подробно — ``docs/QUESTIONS_IMPORT.md``):

1. ``parse_file`` — разбор md / txt / csv / json в ``RawQuestion`` (текст, заголовки-разделы, строка);
2. ``import_paths`` — нормализация, дедупликация по ключу, сохранение в ``.llmlab/questions_import/raw.jsonl``
   (папка состояния, в git не попадает: сырой текст — личные данные пользователя);
3. разметка (тема, сложность, задачи, вид) и **переформулировка своими словами** делается пачками вне этого
   модуля (агентом или вручную) прямо в ``content/questions/interview-*.yaml``; каждый размеченный вопрос
   хранит ``source_ref`` — хэши сырых вопросов, которые он покрывает;
4. ``import_status`` / ``next_batch`` — что уже размечено, что осталось (продолжение в новой сессии);
   ``verbatim_matches`` — страховка: размеченный текст не должен совпадать с сырым.

Ответы из сырой базы (если они есть) импортируются как подсказка разметчику и тоже не коммитятся.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llmlab.content_loader import Course
from llmlab.paths import state_dir

SUPPORTED = {".md", ".markdown", ".txt", ".csv", ".json"}
QUESTION_KEYS = ("question", "вопрос", "q", "text")
ANSWER_KEYS = ("answer", "ответ", "a")

_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class RawQuestion:
    ref: str                       # sha1 нормализованного ключа, 10 символов
    text: str                      # вопрос как есть (без маркера списка)
    section: list[str]             # заголовки над вопросом: ["Блок 1: …", "Токенизация"]
    file: str
    line: int
    answer: str | None = None
    duplicates: list[dict] = field(default_factory=list)   # другие места, где встретился тот же вопрос


def clean(text: str) -> str:
    return " ".join(text.replace(" ", " ").split()).strip()


def dedupe_key(text: str) -> str:
    """Ключ для поиска дублей: регистр, ё/е, пунктуация и пробелы не важны."""
    t = clean(text).lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def make_ref(text: str) -> str:
    return hashlib.sha1(dedupe_key(text).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------- парсеры


def _parse_markdown(lines: list[str], name: str) -> list[RawQuestion]:
    out: list[RawQuestion] = []
    stack: list[tuple[int, str]] = []
    for no, line in enumerate(lines, 1):
        h = _HEADING.match(line)
        if h:
            level = len(h.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, clean(h.group(2)))]
            continue
        is_item = bool(_BULLET.match(line))
        text = clean(_BULLET.sub("", line))
        if not text or set(text) <= set("-—=_*"):
            continue
        if is_item or text.endswith("?"):
            # заголовок документа (уровень 1) — это название файла, а не раздел
            section = [t for lv, t in stack if lv > 1] or [t for _, t in stack]
            out.append(RawQuestion(make_ref(text), text, section, name, no))
    return out


def _parse_txt(lines: list[str], name: str) -> list[RawQuestion]:
    out = []
    for no, line in enumerate(lines, 1):
        text = clean(_BULLET.sub("", line))
        if text and not text.startswith("#"):
            out.append(RawQuestion(make_ref(text), text, [], name, no))
    return out


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    low = {str(k).strip().lower(): v for k, v in row.items()}
    for k in keys:
        if low.get(k):
            return clean(str(low[k]))
    return None


def _parse_csv(path: Path, name: str) -> list[RawQuestion]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if sample else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        out = []
        for no, row in enumerate(reader, 2):
            text = _pick(row, QUESTION_KEYS) or clean(str(next(iter(row.values()), "") or ""))
            if text:
                section = [clean(str(row[k])) for k in row if str(k).strip().lower() in ("topic", "тема", "section")]
                out.append(RawQuestion(make_ref(text), text, [s for s in section if s], name, no,
                                       answer=_pick(row, ANSWER_KEYS)))
    return out


def _parse_json(path: Path, name: str) -> list[RawQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out = []
    for i, item in enumerate(data, 1):
        if isinstance(item, str):
            text, answer = clean(item), None
        elif isinstance(item, dict):
            text, answer = _pick(item, QUESTION_KEYS), _pick(item, ANSWER_KEYS)
        else:
            continue
        if text:
            out.append(RawQuestion(make_ref(text), text, [], name, i, answer=answer))
    return out


def parse_file(path: Path) -> list[RawQuestion]:
    suffix = path.suffix.lower()
    name = path.name
    if suffix in (".md", ".markdown"):
        return _parse_markdown(path.read_text(encoding="utf-8").splitlines(), name)
    if suffix == ".txt":
        return _parse_txt(path.read_text(encoding="utf-8").splitlines(), name)
    if suffix == ".csv":
        return _parse_csv(path, name)
    if suffix == ".json":
        return _parse_json(path, name)
    raise ValueError(f"{name}: формат {suffix} не поддерживается (md, txt, csv, json; xlsx — сохрани как csv)")


# ---------------------------------------------------------------------------- импорт и состояние


@dataclass
class ImportResult:
    files: list[str]
    parsed: int
    unique: int
    duplicates: int
    skipped: list[str]
    out: Path


def raw_file(root: Path | None = None) -> Path:
    path = state_dir(root) / "questions_import"
    path.mkdir(parents=True, exist_ok=True)
    return path / "raw.jsonl"


def _collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and f.name.lower() != "readme.md")
        else:
            files.append(p)
    return files


def import_paths(paths: list[Path], root: Path | None = None) -> ImportResult:
    """Разобрать файлы, убрать точные дубли (по ``dedupe_key``), сохранить в raw.jsonl."""
    by_ref: dict[str, RawQuestion] = {}
    parsed, skipped, used = 0, [], []
    for f in _collect(paths):
        if f.suffix.lower() not in SUPPORTED:
            skipped.append(f.name)
            continue
        used.append(f.name)
        for q in parse_file(f):
            parsed += 1
            if q.ref in by_ref:
                by_ref[q.ref].duplicates.append({"file": q.file, "line": q.line, "text": q.text})
                if q.answer and not by_ref[q.ref].answer:
                    by_ref[q.ref].answer = q.answer
            else:
                by_ref[q.ref] = q
    out = raw_file(root)
    with out.open("w", encoding="utf-8") as fh:
        for q in by_ref.values():
            fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    return ImportResult(used, parsed, len(by_ref), parsed - len(by_ref), skipped, out)


def load_raw(root: Path | None = None) -> list[RawQuestion]:
    path = raw_file(root)
    if not path.exists():
        return []
    return [RawQuestion(**json.loads(ln)) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@dataclass
class ImportStatus:
    total: int
    labeled: int
    remaining: list[RawQuestion]
    unknown_refs: list[str]        # source_ref в content/, которых нет в текущем импорте


def import_status(course: Course, root: Path | None = None) -> ImportStatus:
    raw = load_raw(root)
    covered = {r for q in course.questions.values() for r in q.source_ref}
    refs = {q.ref for q in raw}
    return ImportStatus(
        total=len(raw),
        labeled=sum(q.ref in covered for q in raw),
        remaining=[q for q in raw if q.ref not in covered],
        unknown_refs=sorted(covered - refs) if raw else [],
    )


def next_batch(course: Course, size: int = 50, root: Path | None = None) -> list[RawQuestion]:
    return import_status(course, root).remaining[:size]


def verbatim_matches(course: Course, root: Path | None = None, threshold: float = 0.8) -> list[dict]:
    """Вопросы user-db, текст которых почти совпадает с сырым (нарушение правила «своими словами»)."""
    raw = {q.ref: q for q in load_raw(root)}
    hits = []
    for q in course.questions.values():
        for ref in q.source_ref:
            r = raw.get(ref)
            if r is None:
                continue
            for field_name, mine in (("question", q.question), ("answer", q.answer)):
                theirs = [r.text] + ([r.answer] if r.answer and field_name == "answer" else [])
                for t in theirs:
                    ratio = difflib.SequenceMatcher(None, dedupe_key(mine), dedupe_key(t)).ratio()
                    if ratio >= threshold:
                        hits.append({"id": q.id, "field": field_name, "ref": ref, "ratio": round(ratio, 3)})
    return hits
