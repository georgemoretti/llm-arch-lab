// Типы и вызовы JSON API (llmlab/server.py). Формат RunResult — docs/TASK_FORMAT.md §4.3.

export type Status = "new" | "attempted" | "solved";

export interface Progress {
  status: Status;
  attempts: number;
  solved_at: string | null;
  last_at: string | null;
}

export interface TaskSummary {
  id: string;
  title: string;
  short_title: string; // для навигации: граф, крошки, «после: …»
  number: string | null; // «C1.3» = блок C, модуль 1, шаг 3; у debug — номер задачи, которую чинит
  parent: string | null; // debug → implement-задача, к которой относится
  block: string;
  module: string;
  type: "implement" | "debug" | "paper" | "quiz" | "project";
  difficulty: "easy" | "medium" | "hard";
  depends_on: string[];
  diff_from: string | null;
  tags: string[];
  time_estimate_min: number;
  has_code: boolean;
  progress: Progress;
}

export interface ModuleInfo {
  id: string;
  title: string;
  description: string;
  number: number; // из id «module-<N>-…»
  tasks: TaskSummary[];
}

export interface BlockInfo {
  id: string;
  title: string;
  description: string;
  optional: boolean; // необязательный блок (фундамент): не предлагается «следующим» тем, кто уже идёт дальше
  baselines: Record<string, { title: string; path: string | null; description: string }>;
  modules: ModuleInfo[];
}

export interface Issue {
  level: "error" | "warning";
  where: string;
  message: string;
}

export interface CourseData {
  blocks: BlockInfo[];
  issues: Issue[];
}

export interface VizInfo {
  id: string;
  title: string;
  description: string;
}

export interface Question {
  id: string;
  topic: string;
  difficulty: "easy" | "medium" | "hard";
  kind: "open" | "mcq" | "code-reading";
  question: string;
  answer: string;
  options: string[];
  correct: number | null;
}

export type Grade = "again" | "hard" | "good" | "easy";

export interface Card {
  qid: string;
  ease: number;
  interval_days: number;
  reps: number;
  lapses: number;
  due_at: string | null;
  last_grade: Grade | null;
  last_review_at: string | null;
  is_new: boolean;
  due: boolean;
  preview: Record<Grade, number>; // через сколько дней вернётся карточка при каждой оценке
}

export interface InterviewQuestion extends Question {
  source: string;
  tags: string[];
  tasks: string[];
  modules: string[];
  card: Card;
}

export interface InterviewData {
  questions: InterviewQuestion[];
  modules: { id: string; title: string }[];
  stats: { total: number; new: number; due: number; learned: number; reviews_24h: number };
}

export interface TaskDetail extends TaskSummary {
  entrypoints: string[];
  readme: string;
  hints: { title: string; body: string }[];
  reading: { title: string; url: string }[];
  papers: { id: string; title: string; authors: string; year: number; url: string; venue: string | null }[];
  viz: VizInfo[];
  questions: Question[];
  diff_from_info: { id: string; title: string; kind: "task" | "baseline"; description?: string } | null;
  dependents: string[];
  diff_unlocked: boolean;
  failed_attempts: number;
  diff_unlock_after: number;
  note: { path: string } | null;
}

export interface CodeFile {
  code: string;
  sha256: string;
  mtime_ns: number;
  path: string;
}

export interface FileMeta {
  sha256: string | null;
  mtime_ns: number | null;
  path: string;
}

export interface TestEntry {
  nodeid: string;
  name: string;
  title: string;
  outcome: "passed" | "failed" | "error" | "skipped" | "not-run";
  message: string | null;
  location: { file: string; line: number; function: string; source: string } | null;
  duration_s: number;
}

export interface RunResult {
  task_id: string;
  impl: string;
  created_at: string;
  status: "passed" | "failed" | "error" | "timeout";
  passed: number;
  failed: number;
  total: number;
  duration_s: number;
  tests: TestEntry[];
  collect_errors: { nodeid: string; message: string }[];
  message: string | null;
  output_tail: string;
  attempt_id?: number;
  // Только у POST /run при status=passed: нужно ли спросить вывод или конспект уже обновлён.
  note?: { status: "needs_insight" | "updated" | string; path?: string; message?: string };
}

export interface Attempt {
  id: number;
  task_id: string;
  created_at: string;
  status: RunResult["status"];
  passed: number;
  failed: number;
  total: number;
  duration_s: number;
  result?: RunResult;
}

export interface NoteSummary {
  task_id: string;
  path: string;
  title: string;
  block: string | null;
  module: string | null;
  tags: string[];
  solved_at: string | null;
  attempts: number | null;
  parent: string | null;
  depends_on: string[];
  mtime_ns: number;
}

export interface NoteDetail extends NoteSummary {
  meta: Record<string, unknown>;
  body: string;
  abs_path: string;
  insight: string | null;
}

export interface NotesList {
  notes: NoteSummary[];
  tags: { tag: string; count: number }[];
  dir: string;
}

export interface SearchHit {
  task_id: string;
  path: string;
  title: string;
  snippet: string;
}

export type ArchCell =
  | { status: "open"; value: string; task_id: string; task_title: string }
  | { status: "locked"; task_id: string; task_title: string }
  | { status: "empty" };

export interface ArchTable {
  columns: { id: string; title: string }[];
  rows: {
    id: string;
    title: string;
    year: number | null;
    module: string | null;
    module_title: string | null;
    source: string | null;
    cells: Record<string, ArchCell>;
  }[];
  opened: number;
  total: number;
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : (detail as { message?: string })?.message ?? `HTTP ${status}`);
  }
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (method !== "GET") headers["X-LLMLab"] = "1"; // см. защиту в server.py
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(url, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? res.statusText);
  return data as T;
}

const t = (id: string) => `/api/tasks/${encodeURIComponent(id)}`;

export const api = {
  course: () => request<CourseData>("GET", "/api/course"),
  task: (id: string) => request<TaskDetail>("GET", t(id)),
  code: (id: string) => request<CodeFile>("GET", `${t(id)}/code`),
  codeMeta: (id: string) => request<FileMeta>("GET", `${t(id)}/code/meta`),
  saveCode: (id: string, code: string, base_sha256: string | null, force = false) =>
    request<FileMeta>("PUT", `${t(id)}/code`, { code, base_sha256, force }),
  reset: (id: string) => request<CodeFile>("POST", `${t(id)}/reset`),
  run: (id: string) => request<RunResult>("POST", `${t(id)}/run`),
  attempts: (id: string) => request<Attempt[]>("GET", `${t(id)}/attempts`),
  reference: (id: string) => request<{ code: string }>("GET", `${t(id)}/reference`),
  notes: () => request<NotesList>("GET", "/api/notes"),
  note: (id: string) => request<NoteDetail>("GET", `/api/notes/${encodeURIComponent(id)}`),
  saveNote: (id: string, insight?: string) =>
    request<{ task_id: string; path: string; created: boolean }>("POST", `/api/notes/${encodeURIComponent(id)}`, { insight }),
  rebuildNotes: () => request<{ built: string[]; skipped: { task_id: string; reason: string }[] }>("POST", "/api/notes"),
  searchNotes: (q: string) =>
    request<{ results: SearchHit[] }>("GET", `/api/notes/search?q=${encodeURIComponent(q)}`),
  architectures: () => request<ArchTable>("GET", "/api/architectures"),
  completeTask: (id: string) => request<Progress>("POST", `${t(id)}/complete`),
  interview: () => request<InterviewData>("GET", "/api/interview"),
  review: (qid: string, grade: Grade, mode: "review" | "quiz" | "task") =>
    request<Card>("POST", "/api/interview/review", { qid, grade, mode }),
};

export const NOTES_EXPORT_URL = "/api/notes/export.zip";
