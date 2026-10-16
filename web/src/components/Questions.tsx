// Вкладка «Вопросы» на странице задачи: вопросы из content/questions, привязанные к задаче.
// После ответа — самооценка, она идёт в интервальное повторение (вкладка Interview, SM-2).
import { useEffect, useState } from "react";
import { api, type Card, type Question } from "../api";
import { INTERVIEW_HREF } from "../router";
import { ReviewCard } from "./ReviewCard";

export function Questions({ questions }: { questions: Question[] }) {
  const [cards, setCards] = useState<Record<string, Card> | null>(null);

  useEffect(() => {
    api
      .interview()
      .then((d) => setCards(Object.fromEntries(d.questions.map((q) => [q.id, q.card]))))
      .catch(() => setCards({}));
  }, []);

  if (!questions.length) return <p className="muted">Вопросов к этой задаче пока нет.</p>;
  return (
    <div className="questions">
      <p className="muted">
        Сначала ответь сам — вслух или письменно, — потом открой эталон. В вопросах с вариантами ответ проверяется сразу.
        Оценка ответа попадает в <a href={INTERVIEW_HREF}>интервальное повторение</a>: вопрос вернётся, когда начнёшь
        его забывать.
      </p>
      {questions.map((q, i) => (
        <ReviewCard
          key={q.id}
          q={q}
          n={i + 1}
          card={cards?.[q.id] ?? null}
          mode="task"
          onGraded={(c) => setCards((prev) => ({ ...(prev ?? {}), [q.id]: c }))}
        />
      ))}
    </div>
  );
}
