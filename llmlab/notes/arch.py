"""Таблица архитектур «модель × компонент», которая открывается по мере прохождения курса.

Строки и столбцы — ``content/architectures.yaml``, значения ячеек — поле ``arch`` в task.yaml.
Ячейка открыта, если задача, которая её задаёт, решена; закрытая ячейка не раскрывает значение
(только какая задача её откроет), ячейка без задачи — «появится позже».
"""

from __future__ import annotations

from llmlab.content_loader import Course


def arch_table(course: Course, solved: set[str]) -> dict:
    arch = course.architectures
    if arch is None:
        return {"columns": [], "rows": [], "opened": 0, "total": 0}
    owners: dict[tuple[str, str], tuple[str, str]] = {}
    for task in course.tasks.values():
        for model, cells in task.arch.items():
            for col, value in cells.items():
                owners[model, col] = (task.id, value)
    modules = {m.id: m.title for b in course.blocks.values() for m in b.modules}

    rows, opened = [], 0
    for m in arch.models:
        cells = {}
        for c in arch.columns:
            owner = owners.get((m.id, c.id))
            if owner is None:
                cells[c.id] = {"status": "empty"}
            elif owner[0] in solved:
                cells[c.id] = {"status": "open", "value": owner[1], "task_id": owner[0],
                               "task_title": course.tasks[owner[0]].title}
                opened += 1
            else:
                cells[c.id] = {"status": "locked", "task_id": owner[0], "task_title": course.tasks[owner[0]].title}
        rows.append({**m.model_dump(), "module_title": modules.get(m.module or ""), "cells": cells})
    return {
        "columns": [c.model_dump() for c in arch.columns],
        "rows": rows,
        "opened": opened,
        "total": len(owners),
    }
