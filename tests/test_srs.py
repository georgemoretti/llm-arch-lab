"""Интервальное повторение: алгоритм SM-2 (llmlab/srs.py) и API вкладки Interview."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import yaml
from fastapi.testclient import TestClient

from llmlab import srs
from llmlab.server import create_app
from tests.toycourse import write_task

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
H = {"X-LLMLab": "1"}


def _days(card: srs.Card) -> float:
    return (datetime.fromisoformat(card.due_at) - NOW).total_seconds() / 86400


def test_good_sequence_intervals():
    c = srs.Card("q")
    assert srs.is_due(c), "новая карточка всегда к показу"
    intervals = []
    for _ in range(4):
        c = srs.schedule(c, "good", NOW)
        intervals.append(c.interval_days)
    assert intervals == [1.0, 6.0, 15.0, 37.5], "SM-2: 1, 6, затем × EF (у «good» EF не меняется: 2.5)"
    assert c.ease == 2.5 and c.reps == 4 and c.lapses == 0


def test_again_resets_and_lowers_ease():
    c = srs.schedule(srs.schedule(srs.Card("q"), "good", NOW), "good", NOW)
    c = srs.schedule(c, "again", NOW)
    assert c.reps == 0 and c.lapses == 1 and c.interval_days == 0
    assert _days(c) == pytest.approx(10 / 1440), "провал — повтор через 10 минут"
    assert c.ease == pytest.approx(2.5 - 0.32)
    assert not srs.is_due(c, NOW) and srs.is_due(c, NOW + timedelta(minutes=11))


def test_hard_easy_and_floor():
    assert srs.schedule(srs.Card("q"), "easy", NOW).interval_days == 1.3
    assert srs.schedule(srs.Card("q"), "hard", NOW).ease == pytest.approx(2.36)
    c = srs.Card("q", ease=1.35)
    assert srs.schedule(c, "again", NOW).ease == srs.MIN_EASE
    p = srs.preview(srs.Card("q", ease=2.5, interval_days=6, reps=2), NOW)
    assert p["again"] < p["hard"] < p["good"] < p["easy"]
    with pytest.raises(ValueError):
        srs.schedule(srs.Card("q"), "perfect", NOW)


@pytest.fixture
def client(toy_root):
    write_task(toy_root, "Z01-add")
    write_task(toy_root, "Z-Q1-quiz", type_="quiz", meta={"questions": ["toy-*"]})
    (toy_root / "content" / "questions" / "toy.yaml").write_text(yaml.safe_dump([
        {"id": "toy-a", "topic": "t", "tasks": ["Z01-add"], "question": "A?", "answer": "a"},
        {"id": "toy-b", "topic": "t", "question": "B?", "answer": "b"},
        {"id": "iv-x", "topic": "other", "tags": ["rag"], "question": "X?", "answer": "x", "source": "user-db",
         "source_ref": ["0123456789"]},
    ], allow_unicode=True), encoding="utf-8")
    return TestClient(create_app(toy_root))


def test_interview_lists_questions_with_modules(client):
    data = client.get("/api/interview").json()
    qs = {q["id"]: q for q in data["questions"]}
    assert set(qs) == {"toy-a", "toy-b", "iv-x"}
    assert qs["toy-a"]["tasks"] == ["Z-Q1-quiz", "Z01-add"] and qs["toy-a"]["modules"] == ["module-1"]
    assert qs["iv-x"]["tasks"] == [] and qs["iv-x"]["tags"] == ["rag"]
    assert "source_ref" not in qs["iv-x"], "хэши сырых вопросов фронтенду не нужны"
    assert data["stats"] == {"total": 3, "new": 3, "due": 0, "learned": 0, "reviews_24h": 0}
    assert qs["toy-a"]["card"]["is_new"] and set(qs["toy-a"]["card"]["preview"]) == set(srs.GRADES)


def test_review_updates_card_and_history(client):
    r = client.post("/api/interview/review", json={"qid": "toy-a", "grade": "good"}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["reps"] == 1 and r.json()["interval_days"] == 1.0 and not r.json()["due"]
    client.post("/api/interview/review", json={"qid": "toy-b", "grade": "again", "mode": "quiz"}, headers=H)
    stats = client.get("/api/interview").json()["stats"]
    assert stats == {"total": 3, "new": 1, "due": 0, "learned": 1, "reviews_24h": 2}
    hist = client.get("/api/interview/toy-b/history").json()
    assert [(h["grade"], h["mode"]) for h in hist] == [("again", "quiz")]
    assert client.post("/api/interview/review", json={"qid": "nope", "grade": "good"}, headers=H).status_code == 404
    assert client.post("/api/interview/review", json={"qid": "toy-a", "grade": "ok"}, headers=H).status_code == 422
    assert client.post("/api/interview/review", json={"qid": "toy-a", "grade": "good"}).status_code == 403


def test_complete_quiz_marks_solved(client):
    r = client.post("/api/tasks/Z-Q1-quiz/complete", headers=H)
    assert r.status_code == 200 and r.json()["status"] == "solved"
    assert client.post("/api/tasks/Z01-add/complete", headers=H).status_code == 400
    course = client.get("/api/course").json()
    st = {t["id"]: t["progress"]["status"] for m in course["blocks"][0]["modules"] for t in m["tasks"]}
    assert st["Z-Q1-quiz"] == "solved" and st["Z01-add"] == "new"
