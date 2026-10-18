"""Локальный веб-сервер курса: JSON API + собранный фронтенд из ``llmlab/static/``.

Сервер — тонкий слой поверх ``content_loader`` (что за задачи), ``workspace`` (код
пользователя) и ``runner`` (прогон тестов, история попыток). Своей логики почти нет.

Безопасность (сервер исполняет код пользователя, поэтому чужие сайты не должны до него
дотянуться):

- слушаем только 127.0.0.1 (см. ``llmlab serve``);
- ``Host`` должен быть localhost/127.0.0.1 — защита от DNS rebinding;
- изменяющие запросы (PUT/POST/DELETE) требуют заголовок ``X-LLMLab: 1``: страница с чужого
  origin не может отправить его без CORS-preflight, а CORS мы не включаем;
- файлы адресуются только через id задачи из загруженного курса, итоговый путь проверяется
  на нахождение внутри ``workspace/`` (защита от path traversal).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llmlab import __version__
from llmlab import notes, srs
from llmlab.content_loader import Course, Task, load_course
from llmlab.paths import PACKAGE_DIR, db_path, notes_dir, project_root, workspace_dir
from llmlab.runner import db, run_task_tests
from llmlab.workspace import init_workspace, workspace_file

STATIC_DIR = PACKAGE_DIR / "static"

# Сколько неудачных попыток открывает «Diff с эталоном» (или задача решена).
DIFF_UNLOCK_FAILED_ATTEMPTS = 3

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}
CSRF_HEADER = "x-llmlab"


class CodeIn(BaseModel):
    code: str
    # sha256 версии, с которой начинал редактор. Если файл на диске уже другой (его поменяли
    # во внешнем IDE), сервер вернёт 409, чтобы не затереть чужие правки молча.
    base_sha256: str | None = None
    force: bool = False


class ReviewIn(BaseModel):
    qid: str
    grade: str                     # again | hard | good | easy
    mode: str = "review"           # review | quiz | task — откуда пришёл ответ (для истории)


class NoteIn(BaseModel):
    # None — пересобрать с уже сохранённым выводом; строка — задать/заменить вывод.
    insight: str | None = None


# Код причины NoteError → HTTP-статус.
NOTE_ERROR_STATUS = {"needs_insight": 409, "not_solved": 409, "insight_empty": 422,
                     "insight_too_long": 422, "no_tests": 400, "no_code": 409}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_hints(text: str) -> list[dict]:
    """hints.md → [{title, body}] по заголовкам второго уровня."""
    parts = re.split(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    hints = [{"title": parts[i].strip(), "body": parts[i + 1].strip()} for i in range(1, len(parts) - 1, 2)]
    if not hints and text.strip():
        hints = [{"title": "Подсказка", "body": text.strip()}]
    return hints


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def create_app(root: Path | None = None) -> FastAPI:
    root = (root or project_root()).resolve()
    ws_root = workspace_dir(root).resolve()
    def dbfile() -> Path:
        return db_path(root)  # каждый раз: state_dir() пересоздаст папку, если её удалили
    run_locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    app = FastAPI(title="LLM Arch Lab", version=__version__, docs_url="/api/docs", redoc_url=None)

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ALLOWED_HOSTS and host != "::1":
            return JSONResponse({"detail": f"Недопустимый Host: {host}"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(CSRF_HEADER) != "1":
            return JSONResponse({"detail": f"Нет заголовка {CSRF_HEADER}: 1"}, status_code=403)
        return await call_next(request)

    # ---------- помощники ----------

    def course() -> Course:
        # Загрузка — это чтение нескольких YAML; делаем её на каждый запрос, чтобы правки
        # контента были видны без перезапуска сервера.
        return load_course(root)

    def get_task(c: Course, task_id: str) -> Task:
        task = c.tasks.get(task_id)
        if task is None:
            raise HTTPException(404, f"Задача '{task_id}' не найдена")
        return task

    def code_task(c: Course, task_id: str) -> Task:
        task = get_task(c, task_id)
        if not task.has_code:
            raise HTTPException(400, f"У задачи {task.id} (type: {task.type}) нет кода")
        return task

    def ws_path(task: Task) -> Path:
        path = workspace_file(task, root).resolve()
        if not path.is_relative_to(ws_root):
            raise HTTPException(400, "Путь вне workspace/")
        return path

    def file_meta(path: Path) -> dict:
        data = path.read_bytes()
        return {"sha256": _sha256(data), "mtime_ns": path.stat().st_mtime_ns, "path": str(path.relative_to(root))}

    def ensure_workspace(task: Task) -> Path:
        path = ws_path(task)
        if not path.exists():
            init_workspace(task, root)
        return path

    def task_progress(task_id: str, statuses: dict[str, dict]) -> dict:
        st = statuses.get(task_id)
        return st or {"status": "new", "attempts": 0, "solved_at": None, "last_at": None}

    def diff_unlocked(task: Task) -> tuple[bool, int]:
        attempts = db.attempts(task.id, dbfile())
        failed = sum(a["status"] != "passed" and a["impl"] == "workspace" for a in attempts)
        solved = any(a["status"] == "passed" and a["impl"] == "workspace" for a in attempts)
        return solved or failed >= DIFF_UNLOCK_FAILED_ATTEMPTS, failed

    def note_info(task: Task) -> dict | None:
        """Есть ли конспект: {path} или None."""
        path = notes.note_file(task, root)
        return {"path": path.relative_to(notes_dir(root)).as_posix()} if path.exists() else None

    def build_note(task: Task, insight: str | None = None) -> dict:
        try:
            return notes.build_note(course(), task, root, insight=insight)
        except notes.NoteError as exc:
            raise HTTPException(NOTE_ERROR_STATUS.get(exc.code, 400), {"message": str(exc), "code": exc.code}) from None

    def task_summary(task: Task, statuses: dict[str, dict], labels: dict[str, dict]) -> dict:
        label = labels.get(task.id, {"number": None, "parent": None})
        return {
            "id": task.id, "title": task.title, "short_title": task.nav_title, **label,
            "block": task.block, "module": task.module,
            "type": task.type, "difficulty": task.difficulty, "depends_on": task.depends_on,
            "recommended_after": task.recommended_after,
            "diff_from": task.diff_from, "tags": task.tags, "time_estimate_min": task.time_estimate_min,
            "has_code": task.has_code, "progress": task_progress(task.id, statuses),
        }

    # ---------- API ----------

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__, "root": str(root)}

    @app.get("/api/course")
    def get_course() -> dict:
        """Карта курса: блоки → модули → задачи со статусами."""
        c = course()
        statuses = db.task_statuses(dbfile())
        order = c.topo_order()
        labels = c.labels()
        blocks = []
        for block in c.blocks.values():
            modules = []
            for i, m in enumerate(block.modules):
                tasks = [task_summary(c.tasks[t], statuses, labels) for t in order
                         if c.tasks[t].block == block.id and c.tasks[t].module == m.id]
                modules.append({**m.model_dump(), "number": m.number(i), "tasks": tasks})
            blocks.append({
                "id": block.id, "title": block.title, "description": block.description, "optional": block.optional,
                "baselines": {k: v.model_dump() for k, v in block.baselines.items()},
                "modules": modules,
            })
        return {"blocks": blocks, "issues": [vars(i) for i in c.issues]}

    @app.get("/api/tasks")
    def list_tasks() -> list[dict]:
        c = course()
        statuses = db.task_statuses(dbfile())
        labels = c.labels()
        return [task_summary(c.tasks[t], statuses, labels) for t in c.topo_order()]

    @app.get("/api/tasks/{task_id}")
    def get_task_detail(task_id: str) -> dict:
        c = course()
        task = get_task(c, task_id)
        statuses = db.task_statuses(dbfile())
        unlocked, failed = diff_unlocked(task) if task.has_code else (False, 0)
        baselines = {k: v for b in c.blocks.values() for k, v in b.baselines.items()}
        diff_from = None
        if task.diff_from:
            if task.diff_from in c.tasks:
                diff_from = {"id": task.diff_from, "title": c.tasks[task.diff_from].title, "kind": "task"}
            elif task.diff_from in baselines:
                b = baselines[task.diff_from]
                diff_from = {"id": task.diff_from, "title": b.title, "kind": "baseline", "description": b.description}
        return {
            **task_summary(task, statuses, c.labels()),
            "entrypoints": task.entrypoints,
            "readme": _read(task.dir / "README.md") or "",
            "hints": _split_hints(_read(task.dir / "hints.md") or ""),
            "reading": [r.model_dump() for r in task.reading],
            "papers": [c.papers[p].model_dump() for p in task.papers if p in c.papers],
            "viz": [c.viz[v].model_dump() for v in task.viz if v in c.viz],
            "questions": [q.model_dump() for q in c.questions_for(task)],
            "diff_from_info": diff_from,
            "dependents": [t.id for t in c.tasks.values() if task.id in t.depends_on],
            "diff_unlocked": unlocked,
            "failed_attempts": failed,
            "diff_unlock_after": DIFF_UNLOCK_FAILED_ATTEMPTS,
            "note": note_info(task),
        }

    @app.get("/api/tasks/{task_id}/code")
    def get_code(task_id: str) -> dict:
        """Код пользователя. Если рабочей копии нет — создаётся из заготовки."""
        task = code_task(course(), task_id)
        path = ensure_workspace(task)
        return {"code": path.read_text(encoding="utf-8"), **file_meta(path)}

    @app.get("/api/tasks/{task_id}/code/meta")
    def get_code_meta(task_id: str) -> dict:
        """Лёгкий запрос для отслеживания правок из внешнего IDE (фронтенд опрашивает его)."""
        task = code_task(course(), task_id)
        path = ws_path(task)
        if not path.exists():
            return {"sha256": None, "mtime_ns": None, "path": str(path.relative_to(root))}
        return file_meta(path)

    @app.put("/api/tasks/{task_id}/code")
    def put_code(task_id: str, body: CodeIn) -> dict:
        task = code_task(course(), task_id)
        path = ws_path(task)
        if path.exists() and body.base_sha256 and not body.force:
            current = _sha256(path.read_bytes())
            if current != body.base_sha256:
                raise HTTPException(409, {"message": "Файл изменён вне редактора", "sha256": current})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".py.tmp")
        tmp.write_text(body.code, encoding="utf-8")
        os.replace(tmp, path)  # атомарно: IDE не увидит полузаписанный файл
        return file_meta(path)

    @app.post("/api/tasks/{task_id}/reset")
    def reset_code(task_id: str) -> dict:
        """Сбросить рабочую копию к заготовке (для debug-задачи — к багованному коду)."""
        task = code_task(course(), task_id)
        ws_path(task)
        path = init_workspace(task, root, force=True)
        return {"code": path.read_text(encoding="utf-8"), **file_meta(path)}

    @app.post("/api/tasks/{task_id}/run")
    def run(task_id: str) -> dict:
        """Прогнать тесты на рабочей копии. Sync-эндпоинт → выполняется в пуле потоков."""
        task = code_task(course(), task_id)
        ensure_workspace(task)
        with locks_guard:
            lock = run_locks.setdefault(task.id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Тесты этой задачи уже запущены")
        try:
            result = run_task_tests(task, impl="workspace", root=root)
        finally:
            lock.release()
        if result["status"] == "passed":
            # Первый зелёный прогон → фронтенд спрашивает «Главное, что я понял» и создаёт
            # конспект. Если конспект уже есть — молча обновляем его свежим кодом и историей.
            if note_info(task) is None and db.get_note(task.id, dbfile()) is None:
                result["note"] = {"status": "needs_insight"}
            else:
                try:
                    result["note"] = {"status": "updated", **notes.build_note(course(), task, root)}
                except notes.NoteError as exc:
                    result["note"] = {"status": exc.code, "message": str(exc)}
        return result

    @app.get("/api/tasks/{task_id}/attempts")
    def get_attempts(task_id: str) -> list[dict]:
        """История попыток (новые сверху). Полный RunResult — только у последней."""
        task = get_task(course(), task_id)
        rows = db.attempts(task.id, dbfile())
        out = []
        for i, row in enumerate(reversed(rows)):
            result_json = row.pop("result_json")
            if i == 0:
                row["result"] = json.loads(result_json)
            out.append(row)
        return out

    @app.get("/api/tasks/{task_id}/reference")
    def get_reference(task_id: str) -> dict:
        """Эталон — только после решения или нескольких неудачных попыток."""
        task = code_task(course(), task_id)
        unlocked, failed = diff_unlocked(task)
        if not unlocked:
            left = DIFF_UNLOCK_FAILED_ATTEMPTS - failed
            raise HTTPException(403, f"Эталон откроется после решения или ещё {left} неудачных попыток")
        return {"code": task.solution_file.read_text(encoding="utf-8")}

    @app.get("/api/progress")
    def get_progress() -> dict:
        c = course()
        statuses = db.task_statuses(dbfile())
        tasks = {tid: task_progress(tid, statuses) for tid in c.tasks}
        return {
            "total": len(tasks),
            "solved": sum(p["status"] == "solved" for p in tasks.values()),
            "attempted": sum(p["status"] == "attempted" for p in tasks.values()),
            "tasks": tasks,
        }

    @app.post("/api/tasks/{task_id}/complete")
    def complete_task(task_id: str) -> dict:
        """Отметить paper/quiz-задачу пройденной: у них нет тестов, статус ставит сам пользователь."""
        task = get_task(course(), task_id)
        if task.has_code:
            raise HTTPException(400, "Задачи с кодом решаются прогоном тестов")
        db.mark_completed(task.id, srs._iso(datetime.now(timezone.utc)), dbfile())
        return task_progress(task.id, db.task_statuses(dbfile()))

    # ---------- Interview: вопросы и интервальное повторение ----------

    def card_of(qid: str, cards: dict[str, dict]) -> srs.Card:
        row = cards.get(qid)
        return srs.Card(**row) if row else srs.Card(qid)

    def card_view(card: srs.Card, now: datetime) -> dict:
        return {**card.to_dict(), "is_new": card.due_at is None, "due": srs.is_due(card, now),
                "preview": srs.preview(card, now)}

    @app.get("/api/interview")
    def interview() -> dict:
        """Все вопросы курса с модулями, состоянием карточек SM-2 и сводкой."""
        c = course()
        now = datetime.now(timezone.utc)
        cards = db.srs_cards(dbfile())
        module_titles = {m.id: m.title for b in c.blocks.values() for m in b.modules}
        by_question: dict[str, set[str]] = {}
        for t in c.tasks.values():
            for q in c.questions_for(t):
                by_question.setdefault(q.id, set()).add(t.id)
        items = []
        for q in c.questions.values():
            linked = by_question.get(q.id, set())
            modules = sorted({c.tasks[t].module for t in linked}, key=list(module_titles).index)
            items.append({**q.model_dump(exclude={"source_ref"}), "tasks": sorted(linked), "modules": modules,
                          "card": card_view(card_of(q.id, cards), now)})
        day_ago = srs._iso(now - timedelta(days=1))
        return {
            "questions": items,
            "modules": [{"id": m, "title": t} for m, t in module_titles.items()],
            "stats": {
                "total": len(items),
                "new": sum(i["card"]["is_new"] for i in items),
                "due": sum(i["card"]["due"] and not i["card"]["is_new"] for i in items),
                "learned": sum(i["card"]["reps"] > 0 for i in items),
                "reviews_24h": len(db.srs_reviews(since=day_ago, path=dbfile())),
            },
        }

    @app.post("/api/interview/review")
    def review(body: ReviewIn) -> dict:
        """Самооценка ответа → новое состояние карточки (SM-2) + строка истории."""
        c = course()
        if body.qid not in c.questions:
            raise HTTPException(404, f"Вопрос '{body.qid}' не найден")
        if body.grade not in srs.GRADES:
            raise HTTPException(422, f"grade: одно из {', '.join(srs.GRADES)}")
        if body.mode not in ("review", "quiz", "task"):
            raise HTTPException(422, "mode: review | quiz | task")
        now = datetime.now(timezone.utc)
        card = srs.schedule(card_of(body.qid, db.srs_cards(dbfile())), body.grade, now)
        db.save_review(card.to_dict(), body.grade, body.mode, srs._iso(now), dbfile())
        return card_view(card, now)

    @app.get("/api/interview/{qid}/history")
    def review_history(qid: str) -> list[dict]:
        return db.srs_reviews(qid=qid, path=dbfile())

    @app.get("/api/notes")
    def list_notes() -> dict:
        """Все конспекты (по файлам в notes/) и облако тегов."""
        items = notes.list_notes(root)
        tags: dict[str, int] = {}
        for n in items:
            for t in n["tags"]:
                tags[t] = tags.get(t, 0) + 1
        return {
            "notes": items,
            "tags": [{"tag": t, "count": c} for t, c in sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))],
            "dir": str(notes_dir(root)),
        }

    @app.get("/api/notes/search")
    def search_notes(q: str = "") -> dict:
        """Полнотекстовый поиск (SQLite FTS5, префиксы слов)."""
        return {"query": q, "results": notes.search(root, q), "highlight": [notes.index.HL_START, notes.index.HL_END]}

    @app.get("/api/notes/export.zip")
    def export_notes() -> Response:
        notes.write_index_note(course(), root)
        return Response(notes.export_zip(root), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="llmlab-notes.zip"'})

    @app.get("/api/notes/{task_id}")
    def get_note(task_id: str) -> dict:
        task = get_task(course(), task_id)
        path = notes.note_file(task, root)
        if not path.exists():
            raise HTTPException(404, f"Конспекта {task.id} ещё нет")
        n = notes.read_note(path, root)
        stored = db.get_note(task.id, dbfile())
        return {**n, "abs_path": str(path), "insight": notes.generator.insight_from_file(path)
                or (stored or {}).get("insight")}

    @app.post("/api/notes/{task_id}")
    def post_note(task_id: str, body: NoteIn) -> dict:
        """Создать/пересобрать конспект. Для первого создания нужен ``insight``."""
        return build_note(get_task(course(), task_id), body.insight)

    @app.post("/api/notes")
    def rebuild_all_notes() -> dict:
        built, skipped = notes.build_all(course(), root)
        return {"built": [b["task_id"] for b in built], "skipped": [{"task_id": t, "reason": r} for t, r in skipped]}

    @app.get("/api/architectures")
    def get_architectures() -> dict:
        statuses = db.task_statuses(dbfile())
        return notes.arch_table(course(), {t for t, st in statuses.items() if st["status"] == "solved"})

    @app.get("/api/viz")
    def list_viz() -> list[dict]:
        return [v.model_dump() for v in course().viz.values()]

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def api_not_found(rest: str) -> None:
        raise HTTPException(404, f"Нет такого API: /api/{rest}")

    # ---------- фронтенд ----------

    index = STATIC_DIR / "index.html"
    if (STATIC_DIR / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if not index.exists():
            return JSONResponse(
                {"detail": "Фронтенд не собран: выполни `cd web && npm install && npm run build`."},
                status_code=503,
            )
        candidate = (STATIC_DIR / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(STATIC_DIR.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)

    return app
