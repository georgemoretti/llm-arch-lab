// Вкладка «Вопросы» на странице задачи: вопросы из content/questions, привязанные к задаче.
// Самопроверка без сохранения прогресса; карточки и SM-2 — в Фазе 7 (вкладка Interview).
import { useState } from "react";
import type { Question } from "../api";
import { Markdown } from "./Markdown";
import { DIFFICULTY_LABEL } from "./StatusIcon";

export function Questions({ questions }: { questions: Question[] }) {
  if (!questions.length) return <p className="muted">Вопросов к этой задаче пока нет.</p>;
  return (
    <div className="questions">
      <p className="muted">
        Сначала ответь сам — вслух или письменно, — потом открой эталон. В вопросах с вариантами ответ
        проверяется сразу.
      </p>
      {questions.map((q, i) => (
        <QuestionCard key={q.id} q={q} n={i + 1} />
      ))}
    </div>
  );
}

function QuestionCard({ q, n }: { q: Question; n: number }) {
  const [picked, setPicked] = useState<number | null>(null);
  const [open, setOpen] = useState(false);
  const mcq = q.kind === "mcq" && q.correct !== null;
  const revealed = mcq ? picked !== null : open;

  return (
    <div className="q-card">
      <div className="q-head">
        <span className="q-num">{n}</span>
        <span className={`chip diff-${q.difficulty}`}>{DIFFICULTY_LABEL[q.difficulty]}</span>
        {mcq && <span className="chip">выбор ответа</span>}
        <span className="muted mono q-id">{q.id}</span>
      </div>
      <Markdown text={q.question} />
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
          <Markdown text={q.answer} />
          <button className="link-btn" onClick={() => (mcq ? setPicked(null) : setOpen(false))}>
            {mcq ? "Попробовать ещё раз" : "Скрыть ответ"}
          </button>
        </div>
      )}
    </div>
  );
}
