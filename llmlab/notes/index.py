"""Список, полнотекстовый поиск и экспорт конспектов.

Источник правды — сами файлы в ``notes/`` (их можно править в Obsidian). Индекс FTS5 в SQLite —
это кеш: перед каждым поиском ``sync()`` переиндексирует файлы, у которых поменялся mtime, и
удаляет записи об удалённых файлах. Если SQLite собран без FTS5, поиск идёт простым сканом
файлов — медленнее, но для десятков конспектов незаметно.

Совпадения в сниппетах обрамляются маркерами ``HL_START``/``HL_END`` (управляющие символы,
которых не бывает в Markdown), фронтенд превращает их в подсветку.
"""

from __future__ import annotations

import io
import re
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

from llmlab.notes.generator import INDEX_NOTE, split_frontmatter
from llmlab.paths import db_path, notes_dir
from llmlab.runner import db

HL_START, HL_END = "\u0002", "\u0003"
_FTS_SQL = """CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    path UNINDEXED, title, tags, body, tokenize = "unicode61 remove_diacritics 2")"""


def _note_files(root: Path) -> list[Path]:
    ndir = notes_dir(root)
    if not ndir.is_dir():
        return []
    return sorted(p for p in ndir.rglob("*.md") if p.name != INDEX_NOTE or p.parent != ndir)


def read_note(path: Path, root: Path) -> dict:
    """Файл конспекта → метаданные + тело без frontmatter."""
    text = path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(text)
    rel = path.relative_to(notes_dir(root)).as_posix()
    tags = meta.get("tags") or []
    return {
        "task_id": str(meta.get("id") or path.stem),
        "path": rel,
        "title": str(meta.get("title") or path.stem),
        "block": meta.get("block"),
        "module": meta.get("module"),
        "tags": [str(t) for t in tags] if isinstance(tags, list) else [str(tags)],
        "solved_at": str(meta["solved_at"]) if meta.get("solved_at") else None,
        "attempts": meta.get("attempts"),
        "parent": meta.get("parent"),
        "depends_on": meta.get("depends_on") or [],
        "mtime_ns": path.stat().st_mtime_ns,
        "meta": meta,
        "body": body,
    }


def list_notes(root: Path) -> list[dict]:
    out = []
    for p in _note_files(root):
        n = read_note(p, root)
        n.pop("body")
        n.pop("meta")
        out.append(n)
    return out


def _has_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute(_FTS_SQL)
        return True
    except sqlite3.OperationalError:
        return False


def sync(root: Path) -> bool:
    """Приводит индекс в соответствие с файлами. Возвращает True, если FTS5 доступен."""
    files = {p.relative_to(notes_dir(root)).as_posix(): p for p in _note_files(root)}
    with closing(db.connect(db_path(root))) as conn:
        if not _has_fts(conn):
            return False
        known = {r["path"]: r["mtime_ns"] for r in conn.execute("SELECT path, mtime_ns FROM notes_index")}
        for rel in known.keys() - files.keys():
            conn.execute("DELETE FROM notes_fts WHERE path = ?", (rel,))
            conn.execute("DELETE FROM notes_index WHERE path = ?", (rel,))
        for rel, p in files.items():
            mtime = p.stat().st_mtime_ns
            if known.get(rel) == mtime:
                continue
            n = read_note(p, root)
            conn.execute("DELETE FROM notes_fts WHERE path = ?", (rel,))
            conn.execute("INSERT INTO notes_fts (path, title, tags, body) VALUES (?, ?, ?, ?)",
                         (rel, n["title"], " ".join(n["tags"]), n["body"]))
            conn.execute("INSERT OR REPLACE INTO notes_index (path, task_id, mtime_ns) VALUES (?, ?, ?)",
                         (rel, n["task_id"], mtime))
        conn.commit()
    return True


def _terms(query: str) -> list[str]:
    return re.findall(r"\w+", query.lower())


def search(root: Path, query: str, limit: int = 50) -> list[dict]:
    """Поиск по заголовку, тегам и тексту. Каждое слово запроса — префикс («норм» найдёт «нормализация»)."""
    terms = _terms(query)
    if not terms:
        return []
    if sync(root):
        fts_query = " ".join(f'"{t}"*' for t in terms)
        with closing(db.connect(db_path(root))) as conn:
            try:
                rows = conn.execute(
                    f"""SELECT notes_fts.path AS path, i.task_id AS task_id, title,
                               snippet(notes_fts, 3, '{HL_START}', '{HL_END}', '…', 14) AS snippet
                        FROM notes_fts JOIN notes_index i ON i.path = notes_fts.path
                        WHERE notes_fts MATCH ? ORDER BY bm25(notes_fts, 0, 10.0, 5.0, 1.0) LIMIT ?""",
                    (fts_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        return [dict(r) for r in rows]
    return _scan_search(root, terms, limit)


def _scan_search(root: Path, terms: list[str], limit: int) -> list[dict]:
    """Запасной поиск без FTS5: все слова должны встретиться как подстроки."""
    out = []
    for p in _note_files(root):
        n = read_note(p, root)
        hay = f"{n['title']}\n{' '.join(n['tags'])}\n{n['body']}"
        low = hay.lower()
        if not all(t in low for t in terms):
            continue
        pos = low.find(terms[0])
        start = max(0, pos - 60)
        frag = hay[start: pos + 80].replace("\n", " ")
        frag = re.sub(re.escape(terms[0]), lambda m: HL_START + m.group(0) + HL_END, frag, flags=re.IGNORECASE)
        out.append({"path": n["path"], "task_id": n["task_id"], "title": n["title"],
                    "snippet": ("…" if start else "") + frag + "…"})
        if len(out) >= limit:
            break
    return out


def export_zip(root: Path) -> bytes:
    """Весь ``notes/`` как zip: распакуй и открой папку в Obsidian как vault."""
    ndir = notes_dir(root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if ndir.is_dir():
            for p in sorted(ndir.rglob("*.md")):
                zf.write(p, "llmlab-notes/" + p.relative_to(ndir).as_posix())
    return buf.getvalue()
