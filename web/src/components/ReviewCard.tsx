// Карточка вопроса с самопроверкой и самооценкой SM-2 (Again / Hard / Good / Easy).
// Используется во вкладке Interview (повторение, квиз модуля, один вопрос) и во вкладке «Вопросы» задачи.
import { useState } from "react";
import { api, type Card, type Grade, type Question } from "../api";
import { questionHref, taskHref } from "../router";
import { Markdown } from "./Markdown";
import { DIFFICULTY_LABEL } from "./StatusIcon";

export const GRADE_LABEL: Record<Grade, string> = { again: "Не помню", hard: "С трудом", good: "Помню", easy: "Легко" };
const GRADES: Grade[] = ["again", "hard", "good", "easy"];

export function fmtInterval(days: number): string {
  if (days < 1 / 24) return `${Math.max(1, Math.round(days * 1440))} мин`;
  if (days < 1) return `${Math.round(days * 24)} ч`;
  if (days < 30) return `${+days.toFixed(1)} дн`;
  if (days < 365) return `${+(days / 30).toFixed(1)} мес`;
  return `${+(days / 365).toFixed(1)} г`;
}

export interface ReviewQuestion extends Question {
  source?: string;
  tags?: string[];
  tasks?: string[];
}

interface Props {
  q: ReviewQuestion;
  n?: number;
  card?: Card | null; // нет — карточка без оценок (сервер недоступен)
  mode: "review" | "quiz" | "task";
  onGraded?(card: Card, grade: Grade): void;
  showMeta?: boolean; // ссылки на задачи и id
}

export function ReviewCard({ q, n, card, mode, onGraded, showMeta = true }: Props) {
  const [picked, setPicked] = useState<number | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [graded, setGraded] = useState<Grade | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const mcq = q.kind === "mcq" && q.correct !== null;
  const revealed = mcq ? picked !== null : open;
  const suggested: Grade | null = mcq && picked !== null ? (picked === q.correct ? "good" : "again") : null;

  const grade = async (g: Grade) => {
    setBusy(true);
    setErr(null);
    try {
      const next = await api.review(q.id, g, mode);
      setGraded(g);
      onGraded?.(next, g);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="q-card">
      <div className="q-head">
        {n !== undefined && <span className="q-num">{n}</span>}
        <span className={`chip diff-${q.difficulty}`}>{DIFFICULTY_LABEL[q.difficulty]}</span>
        {mcq && <span className="chip">выбор ответа</span>}
        {q.source === "user-db" && <span className="chip src-user">с собеседования</span>}
        {q.tags?.slice(0, 3).map((t) => (
          <span key={t} className="chip muted">
            {t}
          </span>
        ))}
        {showMeta && (
          <a className="muted mono q-id" href={questionHref(q.id)} title="Открыть карточку">
            {q.id}
          </a>
        )}
      </div>
      <Markdown text={q.question} linkTasks />
      {mcq ? (
        <ol className="q-options">
          {q.options.map((opt, i) => {
            let cls = "q-option";
            if (picked !== null && i === q.correct) cls += " right";
            else if (picked === i) cls += " wrong";
            return (
              <li key={i}>
                <button className={cls} disabled={picked !== null} onClick={() => setPicked(i)}>
                  {opt}
                </button>
              </li>
            );
          })}
        </ol>
      ) : (
        !open && (
          <button className="btn small" onClick={() => setOpen(true)}>
            Показать эталонный ответ
          </button>
        )
      )}
      {revealed && (
        <div className="q-answer">
          {mcq && (
            <div className={picked === q.correct ? "q-verdict ok" : "q-verdict bad"}>
              {picked === q.correct ? "Верно." : "Неверно."}
            </div>
          )}
          <Markdown text={q.answer} linkTasks />
          {showMeta && q.tasks && q.tasks.length > 0 && (
            <div className="muted q-tasks">
              Задачи:{" "}
              {q.tasks.map((t, i) => (
                <span key={t}>
                  {i > 0 && ", "}
                  <a href={taskHref(t)}>{t}</a>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
      {revealed && card && (
        <div className="q-grade">
          {graded ? (
            <span className="muted">
              Оценка «{GRADE_LABEL[graded]}» сохранена.{" "}
              <button className="link-btn" onClick={() => setGraded(null)}>
                изменить
              </button>
            </span>
          ) : (
            <>
              <span className="muted">{mcq ? "Насколько уверенно?" : "Как ответил?"}</span>
              {GRADES.map((g) => (
                <button
                  key={g}
                  className={`btn small grade-${g} ${suggested === g ? "suggested" : ""}`}
                  disabled={busy}
                  onClick={() => grade(g)}
                  title={`Следующий показ через ${fmtInterval(card.preview[g])}`}
                >
                  {GRADE_LABEL[g]} <span className="q-grade-int">{fmtInterval(card.preview[g])}</span>
                </button>
              ))}
            </>
          )}
          {err && <span className="error">{err}</span>}
        </div>
      )}
      {revealed && !mcq && !card && (
        <button className="link-btn" onClick={() => setOpen(false)}>
          Скрыть ответ
        </button>
      )}
    </div>
  );
}
