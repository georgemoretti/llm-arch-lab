// Человеческие имена задач вместо id («C01d-rmsnorm-debug» → «1.1 · Найди баг: RMSNorm»).
// Карта курса грузится один раз на вкладку: имена и номера не зависят от прогресса.
import { useEffect, useState } from "react";
import { api, type TaskSummary } from "./api";

export type CourseIndex = Map<string, TaskSummary>;

let pending: Promise<CourseIndex> | null = null;

export function loadCourseIndex(): Promise<CourseIndex> {
  pending ??= api.course().then(
    (c) => new Map(c.blocks.flatMap((b) => b.modules.flatMap((m) => m.tasks)).map((t) => [t.id, t])),
    (e) => {
      pending = null; // следующая попытка — заново
      throw e;
    },
  );
  return pending;
}

export function useCourseIndex(): CourseIndex | null {
  const [index, setIndex] = useState<CourseIndex | null>(null);
  useEffect(() => {
    loadCourseIndex().then(setIndex, () => setIndex(new Map()));
  }, []);
  return index;
}

type Named = Pick<TaskSummary, "short_title" | "number">;

/** «1.3 · RoPE» или просто «Статьи: RoFormer…» у задач без номера. */
export const taskLabel = (t: Named) => (t.number ? `${t.number} · ${t.short_title}` : t.short_title);

/** Подпись по id; пока индекс не загружен (или id чужой) — сам id. */
export const labelById = (index: CourseIndex | null, id: string) => {
  const t = index?.get(id);
  return t ? taskLabel(t) : id;
};

/** «C05» / «C05d» / «C-P1» / полный id → id задачи: так задачи упоминаются в тексте README. */
export function codeLookup(index: CourseIndex): Map<string, string> {
  const out = new Map<string, string>();
  for (const id of index.keys()) {
    out.set(id, id);
    const code = id.match(/^C-[A-Z]\d+[a-z]?|^C\d+[a-z]?/)?.[0];
    if (code && !out.has(code)) out.set(code, id);
  }
  return out;
}

export const TYPE_ICON: Record<TaskSummary["type"], string> = {
  implement: "",
  project: "",
  debug: "🐞",
  paper: "📄",
  quiz: "❓",
};
