"""Генератор конспектов, поиск, экспорт и таблица архитектур на игрушечном курсе.

Попытки пишутся в БД напрямую (без subprocess), чтобы тесты были быстрыми.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
import yaml

from toycourse import GOOD, write_task
from llmlab import notes
from llmlab.content_loader import load_course
from llmlab.notes.generator import collect_mistakes, split_frontmatter
from llmlab.paths import db_path
from llmlab.runner import db

T0 = "2026-10-16T10:00:00+00:00"


def _attempt(root: Path, task_id: str, status: str, tests: list[dict], *, code: str = GOOD,
             created_at: str = T0, message: str | None = None) -> None:
    passed = sum(t["outcome"] == "passed" for t in tests)
    db.record_attempt({
        "task_id": task_id, "impl": "workspace", "created_at": created_at, "status": status,
        "passed": passed, "failed": len(tests) - passed, "total": len(tests), "duration_s": 0.1,
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "tests": tests, "message": message,
    }, db_path(root))


def _t(name: str, outcome: str, message: str | None = None, loc: dict | None = None) -> dict:
    return {"name": name, "title": f"тест {name}", "outcome": outcome, "message": message, "location": loc}


def _solve(root: Path, task_id: str = "Z01-add", *, failures: int = 1) -> None:
    ws = root / "workspace" / task_id / "solution.py"
    ws.parent.mkdir(parents=True, exist_ok=True)
    ws.write_text(GOOD, encoding="utf-8")
    _attempt(root, task_id, "failed", [_t("a", "failed", "Функция ещё не реализована (NotImplementedError).")],
             code="x", created_at="2026-10-16T09:40:00+00:00")
    for i in range(failures):
        _attempt(root, task_id, "failed",
                 [_t("a", "failed", f"2 + 3 должно быть 5, получилось {i}", {"file": "solution.py", "line": 2,
                                                                          "function": "add", "source": "return a - b"}),
                  _t("b", "passed")], code="bad", created_at=f"2026-10-16T09:5{i}:00+00:00")
    _attempt(root, task_id, "passed", [_t("a", "passed"), _t("b", "passed")])


@pytest.fixture
def course_root(toy_root: Path) -> Path:
    write_task(toy_root, "Z01-add", files={
        "summary.md": "## TL;DR\n\nСложение.\n\n## Формула\n\n$a+b$\n\n## Формы тензоров\n\nскаляры\n\n"
                      "## Что поменялось относительно базы\n\nничего\n",
    }, meta={"diff_from": "toy-baseline", "tags": ["toy", "math"],
             "reading": [{"title": "Ссылка", "url": "https://example.com"}], "questions": ["toy-*"]})
    write_task(toy_root, "Z02-more", depends_on=["Z01-add"], meta={"diff_from": "Z01-add"})
    (toy_root / "content" / "questions" / "q.yaml").write_text(yaml.safe_dump([
        {"id": "toy-01", "topic": "toy", "question": "Сколько будет 2+3?", "answer": "Пять."},
    ], allow_unicode=True), encoding="utf-8")
    return toy_root


def test_note_has_all_sections_and_obsidian_frontmatter(course_root):
    _solve(course_root)
    course = load_course(course_root)
    info = notes.build_note(course, course.tasks["Z01-add"], course_root, insight="Сложение коммутативно")
    assert info == {"task_id": "Z01-add", "path": "Z/Z01-add.md", "created": True}

    text = (course_root / "notes" / "Z" / "Z01-add.md").read_text(encoding="utf-8")
    meta, body = split_frontmatter(text)
    assert meta["id"] == "Z01-add" and meta["attempts"] == 3 and meta["solved_at"] == "2026-10-16"
    assert meta["tags"] == ["toy", "math"] and meta["parent"] == "toy-baseline"
    headings = [ln[3:] for ln in body.splitlines() if ln.startswith("## ")]
    assert headings == ["TL;DR", "Формула", "Формы тензоров", "Что поменялось относительно базы", "Мой код",
                        "Отличия от эталона", "Мои ошибки по пути", "Главное, что я понял",
                        "Вопросы для повторения", "Читать"]
    assert "```python\n" + GOOD.rstrip() in body
    assert "Код совпадает с эталоном" in body
    assert "> [!question]- Сколько будет 2+3?" in body and "Пять." in body
    assert "[Ссылка](https://example.com)" in body
    assert "Дальше: [[Z02-more]]" in body, "должна быть wiki-ссылка на задачи, которые опираются на эту"


def test_child_note_links_parent_by_wikilink(course_root):
    _solve(course_root, "Z02-more", failures=0)
    course = load_course(course_root)
    notes.build_note(course, course.tasks["Z02-more"], course_root, insight="ок")
    meta, body = split_frontmatter((course_root / "notes" / "Z" / "Z02-more.md").read_text(encoding="utf-8"))
    assert meta["parent"] == "[[Z01-add]]" and meta["depends_on"] == ["[[Z01-add]]"]
    assert "Родитель: [[Z01-add]]" in body
    assert "Решено с 2-й попытки" in body


def test_mistakes_are_grouped_without_duplicates_and_noise():
    attempts = [
        {"id": 1, "status": "failed", "result": {"tests": [
            _t("a", "failed", "Функция ещё не реализована (NotImplementedError)."),
        ]}},
        {"id": 2, "status": "failed", "result": {"tests": [
            _t("a", "failed", "max |Δ| = 2.02: забыто деление"), _t("b", "failed", "утечка будущего"),
        ]}},
        {"id": 3, "status": "failed", "result": {"tests": [
            _t("a", "failed", "max |Δ| = 1.7: забыто деление"), _t("c", "failed", "max |Δ| = 0.3: забыто деление"),
        ]}},
        {"id": 4, "status": "error", "result": {"tests": [], "message": "Не удалось собрать тесты"}},
    ]
    ms = collect_mistakes(attempts)
    assert [m.message for m in ms] == ["max |Δ| = 2.02: забыто деление", "утечка будущего", "Не удалось собрать тесты"]
    assert ms[0].tests == ["тест a", "тест c"] and ms[0].attempts == {2, 3}
    assert ms[2].tests == []


def test_rebuild_keeps_insight_and_file_edits_win(course_root):
    _solve(course_root)
    course = load_course(course_root)
    task = course.tasks["Z01-add"]
    with pytest.raises(notes.NoteError) as exc:
        notes.build_note(course, task, course_root)
    assert exc.value.code == "needs_insight"

    notes.build_note(course, task, course_root, insight="первая мысль\nвторая мысль")
    path = course_root / "notes" / "Z" / "Z01-add.md"
    path.write_text(path.read_text(encoding="utf-8").replace("первая мысль", "правка в Obsidian"), encoding="utf-8")

    built, skipped = notes.build_all(course, course_root)
    assert [b["task_id"] for b in built] == ["Z01-add"] and skipped == []
    text = path.read_text(encoding="utf-8")
    assert "правка в Obsidian\nвторая мысль" in text
    assert db.get_note("Z01-add", db_path(course_root))["insight"] == "правка в Obsidian\nвторая мысль"

    path.unlink()  # файл удалили — вывод восстанавливается из БД
    notes.build_note(course, task, course_root)
    assert "правка в Obsidian" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("bad, code", [("", "insight_empty"), ("  \n \n", "insight_empty"),
                                       ("1\n2\n3\n4", "insight_too_long"), ("x" * 601, "insight_too_long")])
def test_insight_validation(bad, code):
    with pytest.raises(notes.NoteError) as exc:
        notes.validate_insight(bad)
    assert exc.value.code == code


def test_unsolved_task_has_no_note(course_root):
    course = load_course(course_root)
    with pytest.raises(notes.NoteError) as exc:
        notes.build_note(course, course.tasks["Z01-add"], course_root, insight="x")
    assert exc.value.code == "not_solved"


def test_note_code_uses_passing_snapshot_when_workspace_changed(course_root):
    _solve(course_root)
    course = load_course(course_root)
    notes.build_note(course, course.tasks["Z01-add"], course_root, insight="x")
    (course_root / "workspace" / "Z01-add" / "solution.py").write_text("# сломал после решения\n", encoding="utf-8")
    notes.build_note(course, course.tasks["Z01-add"], course_root)
    text = (course_root / "notes" / "Z" / "Z01-add.md").read_text(encoding="utf-8")
    assert "сломал после решения" not in text and GOOD.rstrip() in text


def test_search_tags_and_export(course_root):
    _solve(course_root)
    course = load_course(course_root)
    notes.build_note(course, course.tasks["Z01-add"], course_root, insight="Коммутативность сложения важна")

    hits = notes.search(course_root, "коммутатив")
    assert [h["task_id"] for h in hits] == ["Z01-add"]
    assert "\u0002" in hits[0]["snippet"], "совпадение должно быть подсвечено маркерами"
    assert notes.search(course_root, "несуществующееслово") == []
    assert notes.search(course_root, "   ") == []

    # Правка файла снаружи попадает в индекс без пересборки.
    path = course_root / "notes" / "Z" / "Z01-add.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nабракадабра\n", encoding="utf-8")
    assert [h["task_id"] for h in notes.search(course_root, "абракадабра")] == ["Z01-add"]
    path.unlink()
    assert notes.search(course_root, "абракадабра") == []

    notes.build_note(course, course.tasks["Z01-add"], course_root)
    listed = notes.list_notes(course_root)
    assert [n["task_id"] for n in listed] == ["Z01-add"], "index.md — оглавление, не конспект"
    with zipfile.ZipFile(io.BytesIO(notes.export_zip(course_root))) as zf:
        assert set(zf.namelist()) == {"llmlab-notes/index.md", "llmlab-notes/Z/Z01-add.md"}
        assert "[[Z01-add]]" in zf.read("llmlab-notes/index.md").decode()


def test_architecture_table_opens_cells_on_solve(course_root):
    (course_root / "content" / "architectures.yaml").write_text(yaml.safe_dump({
        "columns": [{"id": "norm", "title": "Норма"}, {"id": "ffn", "title": "FFN"}],
        "models": [{"id": "toy", "title": "Toy", "module": "module-1"}],
    }, allow_unicode=True), encoding="utf-8")
    write_task(course_root, "Z01-add", meta={"arch": {"toy": {"norm": "RMSNorm"}}})
    course = load_course(course_root)
    assert course.issues == []

    table = notes.arch_table(course, solved=set())
    cell = table["rows"][0]["cells"]["norm"]
    assert cell == {"status": "locked", "task_id": "Z01-add", "task_title": "Задача Z01-add"}, \
        "закрытая ячейка не должна раскрывать значение"
    assert table["rows"][0]["cells"]["ffn"] == {"status": "empty"}
    opened = notes.arch_table(course, solved={"Z01-add"})
    assert opened["rows"][0]["cells"]["norm"]["value"] == "RMSNorm" and opened["opened"] == 1


def test_architecture_cells_are_validated(course_root):
    (course_root / "content" / "architectures.yaml").write_text(yaml.safe_dump({
        "columns": [{"id": "norm", "title": "Норма"}], "models": [{"id": "toy", "title": "Toy"}],
    }), encoding="utf-8")
    write_task(course_root, "Z01-add", meta={"arch": {"toy": {"norm": "a", "nope": "b"}, "ghost": {"norm": "c"}}})
    write_task(course_root, "Z02-more", meta={"arch": {"toy": {"norm": "d"}}})
    msgs = [i.message for i in load_course(course_root).errors]
    assert any("столбца 'nope'" in m for m in msgs)
    assert any("модели 'ghost'" in m for m in msgs)
    assert any("уже открывает" in m for m in msgs)
