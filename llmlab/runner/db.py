"""SQLite-хранилище прогресса: ``.llmlab/progress.db``.

Схема (версия 1):

    attempts(
        id           INTEGER PRIMARY KEY,
        task_id      TEXT    NOT NULL,
        created_at   TEXT    NOT NULL,   -- ISO-8601 UTC
        impl         TEXT    NOT NULL,   -- workspace | reference | starter | buggy | path
        status       TEXT    NOT NULL,   -- passed | failed | error | timeout
        passed       INTEGER NOT NULL,
        failed       INTEGER NOT NULL,   -- failed + error
        total        INTEGER NOT NULL,
        duration_s   REAL    NOT NULL,
        code_sha256  TEXT,
        result_json  TEXT    NOT NULL    -- полный RunResult (см. TASK_FORMAT.md)
    )

Версия 2 (Фаза 2, конспекты):

    notes(
        task_id      TEXT PRIMARY KEY,
        insight      TEXT NOT NULL,      -- «Главное, что я понял» (ввод пользователя)
        code         TEXT,               -- снимок кода последнего зелёного прогона
        code_sha256  TEXT,
        path         TEXT NOT NULL,      -- путь конспекта относительно notes/
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    notes_index(path TEXT PRIMARY KEY, task_id TEXT, mtime_ns INTEGER NOT NULL)

Полнотекстовый индекс ``notes_fts`` (FTS5) создаётся отдельно в ``llmlab/notes/index.py``:
если SQLite собран без FTS5, поиск деградирует до LIKE, а не ломает всю базу.

Таблицы SRS-карточек добавятся в Фазе 7 следующей миграцией (``PRAGMA user_version``).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from llmlab.paths import db_path

SCHEMA_VERSION = 2

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS attempts (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id      TEXT    NOT NULL,
        created_at   TEXT    NOT NULL,
        impl         TEXT    NOT NULL,
        status       TEXT    NOT NULL,
        passed       INTEGER NOT NULL,
        failed       INTEGER NOT NULL,
        total        INTEGER NOT NULL,
        duration_s   REAL    NOT NULL,
        code_sha256  TEXT,
        result_json  TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_attempts_task ON attempts(task_id, created_at);
    """,
    2: """
    CREATE TABLE IF NOT EXISTS notes (
        task_id      TEXT PRIMARY KEY,
        insight      TEXT NOT NULL,
        code         TEXT,
        code_sha256  TEXT,
        path         TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS notes_index (
        path         TEXT PRIMARY KEY,
        task_id      TEXT,
        mtime_ns     INTEGER NOT NULL
    );
    """,
}


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or db_path())
    conn.row_factory = sqlite3.Row
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.executescript(_MIGRATIONS[version])
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def record_attempt(result: dict, path: Path | None = None) -> int:
    """Сохраняет RunResult (dict) как попытку. Возвращает id строки."""
    with closing(connect(path)) as conn:
        cur = conn.execute(
            """INSERT INTO attempts (task_id, created_at, impl, status, passed, failed, total,
                                     duration_s, code_sha256, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result["task_id"], result["created_at"], result["impl"], result["status"],
             result["passed"], result["failed"], result["total"], result["duration_s"],
             result.get("code_sha256"), json.dumps(result, ensure_ascii=False)),
        )
        conn.commit()
        return int(cur.lastrowid)


def attempts(task_id: str, path: Path | None = None) -> list[dict]:
    """Все попытки задачи, от старых к новым."""
    with closing(connect(path)) as conn:
        rows = conn.execute(
            "SELECT * FROM attempts WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def task_statuses(path: Path | None = None) -> dict[str, dict]:
    """Сводка по задачам: status (solved | attempted), число попыток, дата решения."""
    with closing(connect(path)) as conn:
        rows = conn.execute(
            """SELECT task_id,
                      COUNT(*) AS attempts,
                      MIN(CASE WHEN status = 'passed' THEN created_at END) AS solved_at,
                      MAX(created_at) AS last_at
               FROM attempts WHERE impl = 'workspace' GROUP BY task_id"""
        ).fetchall()
    return {
        r["task_id"]: {
            "status": "solved" if r["solved_at"] else "attempted",
            "attempts": r["attempts"],
            "solved_at": r["solved_at"],
            "last_at": r["last_at"],
        }
        for r in rows
    }


# ---------- конспекты ----------


def get_note(task_id: str, path: Path | None = None) -> dict | None:
    with closing(connect(path)) as conn:
        row = conn.execute("SELECT * FROM notes WHERE task_id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def all_notes(path: Path | None = None) -> dict[str, dict]:
    with closing(connect(path)) as conn:
        rows = conn.execute("SELECT * FROM notes").fetchall()
    return {r["task_id"]: dict(r) for r in rows}


def upsert_note(task_id: str, *, insight: str, code: str | None, code_sha256: str | None,
                note_path: str, now: str, path: Path | None = None) -> None:
    with closing(connect(path)) as conn:
        conn.execute(
            """INSERT INTO notes (task_id, insight, code, code_sha256, path, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                   insight = excluded.insight, code = excluded.code,
                   code_sha256 = excluded.code_sha256, path = excluded.path,
                   updated_at = excluded.updated_at""",
            (task_id, insight, code, code_sha256, note_path, now, now),
        )
        conn.commit()
