"""Импорт сырой базы вопросов: парсеры, дедупликация, статус разметки, проверка «своими словами».

Все данные — синтетические (настоящая база пользователя в репозиторий не попадает).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llmlab.content_loader import load_course
from llmlab.questions import importer

MD = """# Мои вопросы

## Раздел А

### Подраздел 1
- Что такое внимание?
- Зачем  нужна маска?
1. Нумерованный вопрос?
Просто строка без вопроса
Вопрос без маркера?

---

## Раздел Б
* что такое ВНИМАНИЕ
"""


def _write(root: Path, name: str, text: str) -> Path:
    raw = root / "data" / "interview_raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / name).write_text(text, encoding="utf-8")
    return raw / name


def test_markdown_sections_and_items(tmp_path):
    qs = importer.parse_file(_write(tmp_path, "q.md", MD))
    texts = [q.text for q in qs]
    assert texts == ["Что такое внимание?", "Зачем нужна маска?", "Нумерованный вопрос?",
                     "Вопрос без маркера?", "что такое ВНИМАНИЕ"]
    assert qs[0].section == ["Раздел А", "Подраздел 1"], "заголовок уровня 1 — название файла, не раздел"
    assert qs[-1].section == ["Раздел Б"], "новый раздел того же уровня сбрасывает подразделы"
    assert qs[0].line == 6


def test_dedupe_key_ignores_case_punctuation_yo():
    assert importer.dedupe_key("Что такое ВНИМАНИЕ?") == importer.dedupe_key("что  такое внимание")
    assert importer.make_ref("Ещё вопрос!") == importer.make_ref("еще вопрос")


def test_import_dedupes_and_skips(toy_root):
    _write(toy_root, "a.md", MD)
    _write(toy_root, "b.txt", "1) Что такое внимание\n\nНовый вопрос из txt\n")
    _write(toy_root, "c.csv", "question;answer;topic\nВопрос из csv?;Ответ из csv;RAG\n")
    _write(toy_root, "d.json", json.dumps([{"question": "Вопрос из json?", "answer": "Да"}, "Строкой?"]))
    _write(toy_root, "e.xlsx", "not really")
    res = importer.import_paths([toy_root / "data" / "interview_raw"], toy_root)
    assert res.skipped == ["e.xlsx"]
    assert res.parsed == 5 + 2 + 1 + 2
    assert res.unique == res.parsed - 2, "«что такое внимание» встречается трижды"
    raw = importer.load_raw(toy_root)
    first = next(q for q in raw if q.text == "Что такое внимание?")
    assert len(first.duplicates) == 2
    csvq = next(q for q in raw if q.file == "c.csv")
    assert csvq.answer == "Ответ из csv" and csvq.section == ["RAG"]
    assert res.out.is_relative_to(Path(toy_root).parent / "state"), "сырой текст — только в папке состояния"


def _write_questions(root: Path, items: list[dict]) -> None:
    (root / "content" / "questions" / "interview-test.yaml").write_text(
        yaml.safe_dump(items, allow_unicode=True), encoding="utf-8")


def test_status_and_next_batch(toy_root):
    _write(toy_root, "a.md", MD)
    importer.import_paths([toy_root / "data" / "interview_raw"], toy_root)
    raw = importer.load_raw(toy_root)
    _write_questions(toy_root, [{
        "id": "iv-attn-01", "topic": "attention", "question": "Объясни механизм внимания своими словами.",
        "answer": "Каждый токен взвешивает остальные по сходству запроса и ключа.", "source": "user-db",
        "source_ref": [raw[0].ref, raw[1].ref, "deadbeef00"],
    }])
    course = load_course(toy_root)
    assert not course.errors, course.issues
    st = importer.import_status(course, toy_root)
    assert (st.total, st.labeled) == (4, 2)
    assert st.unknown_refs == ["deadbeef00"]
    assert [q.ref for q in importer.next_batch(course, 1, toy_root)] == [raw[2].ref]


def test_verbatim_check(toy_root):
    _write(toy_root, "a.md", MD)
    importer.import_paths([toy_root / "data" / "interview_raw"], toy_root)
    ref = importer.load_raw(toy_root)[0].ref
    _write_questions(toy_root, [
        {"id": "iv-copy-01", "topic": "attention", "question": "Что такое внимание?", "answer": "…",
         "source": "user-db", "source_ref": [ref]},
        {"id": "iv-own-01", "topic": "attention", "question": "Как устроен механизм attention и что он считает?",
         "answer": "…", "source": "user-db", "source_ref": [ref]},
    ])
    hits = importer.verbatim_matches(load_course(toy_root), toy_root)
    assert [h["id"] for h in hits] == ["iv-copy-01"]


def test_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="xlsx"):
        importer.parse_file(_write(tmp_path, "x.xlsx", ""))
