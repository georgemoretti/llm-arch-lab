"""API веб-сервера на игрушечном курсе."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from toycourse import GOOD, write_task
from llmlab.server import create_app

H = {"X-LLMLab": "1"}


@pytest.fixture
def client(toy_root):
    write_task(toy_root, "Z01-add", files={"hints.md": "## Подсказка 1\nраз\n\n## Подсказка 2\nдва\n"})
    write_task(toy_root, "Z02-more", depends_on=["Z01-add"], meta={"diff_from": "Z01-add"})
    return TestClient(create_app(toy_root))


def test_course_map_has_modules_tasks_and_statuses(client):
    data = client.get("/api/course").json()
    (block,) = data["blocks"]
    m1 = next(m for m in block["modules"] if m["id"] == "module-1")
    assert [t["id"] for t in m1["tasks"]] == ["Z01-add", "Z02-more"]
    assert m1["tasks"][0]["progress"]["status"] == "new"


def test_task_detail(client):
    d = client.get("/api/tasks/Z02-more").json()
    assert d["diff_from_info"] == {"id": "Z01-add", "title": "Задача Z01-add", "kind": "task"}
    assert d["diff_unlocked"] is False
    hints = client.get("/api/tasks/Z01-add").json()["hints"]
    assert [h["body"] for h in hints] == ["раз", "два"]
    assert client.get("/api/tasks/nope").status_code == 404


def test_code_roundtrip_conflict_and_reset(client, toy_root):
    first = client.get("/api/tasks/Z01-add/code").json()
    assert "NotImplementedError" in first["code"]
    path = toy_root / "workspace" / "Z01-add" / "solution.py"
    assert path.exists()

    saved = client.put("/api/tasks/Z01-add/code", json={"code": GOOD, "base_sha256": first["sha256"]}, headers=H)
    assert saved.status_code == 200 and path.read_text() == GOOD

    path.write_text(GOOD + "# правка из IDE\n")  # внешний редактор
    meta = client.get("/api/tasks/Z01-add/code/meta").json()
    assert meta["sha256"] != saved.json()["sha256"]
    stale = client.put("/api/tasks/Z01-add/code", json={"code": "x = 1\n", "base_sha256": saved.json()["sha256"]},
                       headers=H)
    assert stale.status_code == 409, "сохранение поверх внешней правки должно требовать подтверждения"
    forced = client.put("/api/tasks/Z01-add/code", json={"code": "x = 1\n", "force": True}, headers=H)
    assert forced.status_code == 200 and path.read_text() == "x = 1\n"

    reset = client.post("/api/tasks/Z01-add/reset", headers=H).json()
    assert "NotImplementedError" in reset["code"]


def test_mutations_require_header_and_local_host(client):
    assert client.put("/api/tasks/Z01-add/code", json={"code": ""}).status_code == 403
    assert client.post("/api/tasks/Z01-add/run").status_code == 403
    assert client.get("/api/course", headers={"Host": "evil.example.com"}).status_code == 403


def test_path_traversal_is_rejected(client):
    for bad in ("..%2F..%2Fetc", "..%2F..%2Fpyproject.toml", "Z01-add%2F..%2F..%2Fsecret", "%2E%2E"):
        r = client.get(f"/api/tasks/{bad}/code")
        assert r.status_code == 404, bad
        r = client.put(f"/api/tasks/{bad}/code", json={"code": "pwn", "force": True}, headers=H)
        assert r.status_code == 404, bad


def test_run_records_attempt_and_unlocks_reference(client, toy_root):
    assert client.get("/api/tasks/Z01-add/reference").status_code == 403
    client.get("/api/tasks/Z01-add/code")
    fail = client.post("/api/tasks/Z01-add/run", headers=H).json()
    assert fail["status"] == "failed" and fail["total"] == 2
    client.put("/api/tasks/Z01-add/code", json={"code": GOOD, "force": True}, headers=H)
    ok = client.post("/api/tasks/Z01-add/run", headers=H).json()
    assert ok["status"] == "passed"

    attempts = client.get("/api/tasks/Z01-add/attempts").json()
    assert [a["status"] for a in attempts] == ["passed", "failed"]
    assert attempts[0]["result"]["passed"] == 2 and "result" not in attempts[1]
    assert client.get("/api/tasks/Z01-add").json()["progress"]["status"] == "solved"
    assert client.get("/api/progress").json()["solved"] == 1
    assert "def add" in client.get("/api/tasks/Z01-add/reference").json()["code"]


def test_reference_unlocks_after_three_failures(client):
    for _ in range(3):
        assert client.post("/api/tasks/Z01-add/run", headers=H).json()["status"] == "failed"
    assert client.get("/api/tasks/Z01-add").json()["diff_unlocked"] is True


def test_unknown_api_is_json_404(client):
    assert client.get("/api/whatever").status_code == 404
