import type { Status } from "../api";

const LABEL: Record<Status, string> = { new: "новая", attempted: "в процессе", solved: "решена" };

export function StatusIcon({ status }: { status: Status }) {
  return (
    <span className={`status-icon ${status}`} title={LABEL[status]} aria-label={LABEL[status]}>
      {status === "solved" ? "✓" : status === "attempted" ? "◐" : "○"}
    </span>
  );
}

export const TYPE_LABEL: Record<string, string> = {
  implement: "реализация",
  debug: "отладка",
  paper: "статья",
  quiz: "квиз",
  project: "проект",
};

export const DIFFICULTY_LABEL: Record<string, string> = { easy: "легко", medium: "средне", hard: "сложно" };
