"""Автоконспекты задач: генерация Markdown, поиск, экспорт, таблица архитектур."""

from llmlab.notes.arch import arch_table
from llmlab.notes.generator import (
    INSIGHT_HEADING,
    NoteError,
    build_all,
    build_note,
    note_file,
    validate_insight,
    write_index_note,
)
from llmlab.notes.index import export_zip, list_notes, read_note, search, sync

__all__ = [
    "INSIGHT_HEADING", "NoteError", "arch_table", "build_all", "build_note", "export_zip", "list_notes",
    "note_file", "read_note", "search", "sync", "validate_insight", "write_index_note",
]
