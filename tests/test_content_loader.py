from pathlib import Path

import yaml

from toycourse import write_task
from llmlab.content_loader import check_shapes, find_cycles, load_course

REPO_ROOT = Path(__file__).resolve().parents[1]


def _errors(course) -> list[str]:
    return [i.message for i in course.errors]


def test_real_course_loads_without_errors():
    course = load_course(REPO_ROOT)
    assert not course.errors, "\n".join(map(str, course.errors))
    assert "C00-causal-mha" in course.tasks
    task = course.tasks["C00-causal-mha"]
    assert task.entrypoints == ["causal_attention", "CausalSelfAttention"]
    assert course.questions_for(task), "у C00 должны быть вопросы по глобу attention-*"


def test_toy_course_loads(toy_root):
    write_task(toy_root, "Z01-add")
    write_task(toy_root, "Z02-add-more", depends_on=["Z01-add"], meta={"diff_from": "Z01-add"})
    course = load_course(toy_root)
    assert not course.errors, course.errors
    assert course.topo_order() == ["Z01-add", "Z02-add-more"]
    assert course.tasks["Z02-add-more"].dir.name == "Z02-add-more"


def test_unknown_dependency_and_diff_from(toy_root):
    write_task(toy_root, "Z01-add", depends_on=["Z99-missing"], meta={"diff_from": "nope"})
    errors = _errors(load_course(toy_root))
    assert any("Z99-missing" in e for e in errors)
    assert any("diff_from" in e for e in errors)


def test_recommended_after_is_checked_but_not_a_dependency(toy_root):
    write_task(toy_root, "Z01-add")
    write_task(toy_root, "Z02-add-more", meta={"recommended_after": ["Z01-add"]})
    course = load_course(toy_root)
    assert not course.errors, course.errors
    assert course.tasks["Z02-add-more"].depends_on == []
    write_task(toy_root, "Z03-x", meta={"recommended_after": ["Z99-missing"]})
    assert any("recommended_after" in e and "Z99-missing" in e for e in _errors(load_course(toy_root)))


def test_baseline_is_valid_diff_from(toy_root):
    write_task(toy_root, "Z01-add", meta={"diff_from": "toy-baseline"})
    assert not load_course(toy_root).errors


def test_dependency_cycle_detected(toy_root):
    write_task(toy_root, "Z01-a", depends_on=["Z03-c"])
    write_task(toy_root, "Z02-b", depends_on=["Z01-a"])
    write_task(toy_root, "Z03-c", depends_on=["Z02-b"])
    errors = _errors(load_course(toy_root))
    assert any("цикл" in e for e in errors), errors


def test_id_must_match_folder_and_module(toy_root):
    task_dir = write_task(toy_root, "Z01-add")
    data = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    data.update(id="Z01-other", module="module-2")
    (task_dir / "task.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    errors = _errors(load_course(toy_root))
    assert any("не совпадает с папкой 'Z01-add'" in e for e in errors)
    assert any("module" in e for e in errors)


def test_missing_required_files(toy_root):
    write_task(toy_root, "Z01-add", files={"hints.md": None})
    write_task(toy_root, "Z02-fix", type_="debug", files={"buggy.py": None})
    errors = _errors(load_course(toy_root))
    assert any("hints.md" in e for e in errors)
    assert any("buggy.py" in e for e in errors)


def test_schema_errors_are_reported_not_raised(toy_root):
    task_dir = write_task(toy_root, "Z01-add")
    data = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    data.update(difficulty="impossible", typo_field=1)
    (task_dir / "task.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    errors = _errors(load_course(toy_root))
    assert any("difficulty" in e for e in errors)
    assert any("typo_field" in e for e in errors)


def test_questions_and_papers(toy_root):
    write_task(toy_root, "Z01-add", meta={"questions": ["add-*"], "papers": ["missing-paper"]})
    (toy_root / "content" / "questions" / "q.yaml").write_text(yaml.safe_dump([
        {"id": "add-01", "topic": "math", "question": "2+2?", "answer": "4"},
        {"id": "other-01", "topic": "math", "tasks": ["Z01-add"], "question": "1+1?", "answer": "2"},
    ], allow_unicode=True), encoding="utf-8")
    course = load_course(toy_root)
    assert any("missing-paper" in e for e in _errors(course))
    assert {q.id for q in course.questions_for(course.tasks["Z01-add"])} == {"add-01", "other-01"}


def test_mcq_requires_correct_option(toy_root):
    (toy_root / "content" / "questions" / "q.yaml").write_text(yaml.safe_dump([
        {"id": "ok-01", "topic": "t", "kind": "mcq", "question": "?", "answer": "a",
         "options": ["a", "b"], "correct": 0},
        {"id": "bad-01", "topic": "t", "kind": "mcq", "question": "?", "answer": "a", "options": ["a", "b"]},
        {"id": "bad-02", "topic": "t", "kind": "mcq", "question": "?", "answer": "a",
         "options": ["a", "b"], "correct": 2},
        {"id": "bad-03", "topic": "t", "question": "?", "answer": "a", "correct": 0},
    ], allow_unicode=True), encoding="utf-8")
    course = load_course(toy_root)
    assert set(course.questions) == {"ok-01"}
    assert sum("correct" in e for e in _errors(course)) == 3


def test_find_cycles_unit():
    assert find_cycles({"a": ["b"], "b": []}) == []
    assert find_cycles({"a": ["a"]}) == [["a", "a"]]
    cycles = find_cycles({"a": ["b"], "b": ["c"], "c": ["a"]})
    assert len(cycles) == 1 and set(cycles[0]) == {"a", "b", "c"}


GOOD_SHAPES = """
x [B, T, C] — вход
fork
> gate_proj
g [B, T, F]
branch
> up_proj
u [B, T, F]
join ⊙
h [B, T, F]
## дальше
- пояснение
axis r:other = ранг
scalar []
"""


def test_check_shapes_accepts_valid_block():
    assert check_shapes(f"текст\n```shapes\n{GOOD_SHAPES}```\n") == []
    assert check_shapes("```\nx [B] без языка — не схема\n```") == []


def test_check_shapes_catches_typos():
    bad = "```shapes\nx [B, T, C\n>c_attn\nq_proj(x)\n##секция\naxis N:huge = ?\nbranch\nfork\n```\n"
    errors = "\n".join(check_shapes(bad))
    for expected in ["несбалансированные", "после «>» нужен пробел", "тензор без формы", "после «##» нужен пробел",
                     "неизвестный вид оси 'huge'", "branch без fork", "fork не закрыт"]:
        assert expected in errors, expected


def test_shapes_error_is_attached_to_task(toy_root):
    write_task(toy_root, "Z01-add", files={"README.md": "# Задача\n\n```shapes\nx [B, T\n```\n"})
    course = load_course(toy_root)
    (issue,) = [i for i in course.errors if "shapes" in i.message]
    assert "Z01-add" in Path(issue.where).parts and "README.md" in issue.where
