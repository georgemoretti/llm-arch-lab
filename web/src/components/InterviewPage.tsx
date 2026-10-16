// Вкладка Interview: интервальное повторение (SM-2), квиз по модулю, список всех вопросов и карточка вопроса.
// Состояние карточек хранит сервер (SQLite, llmlab/srs.py); здесь — только выбор очереди и показ.
import { useEffect, useMemo, useState } from "react";
import { api, type Card, type Grade, type InterviewData, type InterviewQuestion } from "../api";
import { INTERVIEW_HREF, questionHref } from "../router";
import { fmtInterval, GRADE_LABEL, ReviewCard } from "./ReviewCard";
import { DIFFICULTY_LABEL } from "./StatusIcon";

type Mode = "review" | "quiz" | "all";
type Source = "all" | "course" | "user-db";
const OFF_COURSE = "~off"; // псевдо-модуль: вопросы без привязки к задачам курса
const NEW_LIMITS = [10, 20, 50];

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(`llmlab.interview.${key}`);
    return v === null ? fallback : (JSON.parse(v) as T);
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try {
    localStorage.setItem(`llmlab.interview.${key}`, JSON.stringify(value));
  } catch {
    /* приватный режим — не страшно */
  }
}

export function InterviewPage({ qid }: { qid: string | null }) {
  const [data, setData] = useState<InterviewData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setModeState] = useState<Mode>(() => load("mode", "review"));
  const setMode = (m: Mode) => {
    setModeState(m);
    save("mode", m);
  };

  useEffect(() => {
    api.interview().then(setData).catch((e) => setError(e.message));
  }, []);

  const onGraded = (card: Card) =>
    setData((d) => {
      if (!d) return d;
      const questions = d.questions.map((q) => (q.id === card.qid ? { ...q, card } : q));
      return { ...d, questions, stats: computeStats(questions, d.stats.reviews_24h + 1) };
    });

  if (error) return <div className="page-msg error">{error}</div>;
  if (!data) return <div className="page-msg">Загрузка…</div>;

  if (qid) {
    const q = data.questions.find((x) => x.id === qid);
    return (
      <div className="interview">
        <a href={INTERVIEW_HREF} className="muted">
          ← Все вопросы
        </a>
        {q ? (
          <>
            <ReviewCard key={q.id} q={q} card={q.card} mode="review" onGraded={onGraded} />
            <CardState card={q.card} />
          </>
        ) : (
          <p className="error">Вопрос «{qid}» не найден.</p>
        )}
      </div>
    );
  }

  return (
    <div className="interview">
      <div className="iv-head">
        <h1>Interview</h1>
        <Stats data={data} />
      </div>
      <div className="seg" role="tablist">
        {(
          [
            ["review", "Повторение"],
            ["quiz", "Квиз по модулю"],
            ["all", "Все вопросы"],
          ] as [Mode, string][]
        ).map(([id, label]) => (
          <button key={id} role="tab" aria-selected={mode === id} className={mode === id ? "active" : ""} onClick={() => setMode(id)}>
            {label}
          </button>
        ))}
      </div>
      {mode === "review" && <ReviewMode data={data} onGraded={onGraded} />}
      {mode === "quiz" && <QuizMode data={data} onGraded={onGraded} />}
      {mode === "all" && <AllMode data={data} />}
    </div>
  );
}

function computeStats(questions: InterviewQuestion[], reviews: number): InterviewData["stats"] {
  return {
    total: questions.length,
    new: questions.filter((q) => q.card.is_new).length,
    due: questions.filter((q) => q.card.due && !q.card.is_new).length,
    learned: questions.filter((q) => q.card.reps > 0).length,
    reviews_24h: reviews,
  };
}

function Stats({ data }: { data: InterviewData }) {
  const s = data.stats;
  return (
    <div className="iv-stats">
      <span>
        <b>{s.total}</b> вопросов
      </span>
      <span>
        <b>{s.due}</b> к повторению
      </span>
      <span>
        <b>{s.new}</b> новых
      </span>
      <span>
        <b>{s.learned}</b> изучено
      </span>
      <span className="muted">{s.reviews_24h} ответов за сутки</span>
    </div>
  );
}

function bySource(q: InterviewQuestion, src: Source) {
  return src === "all" || (src === "user-db" ? q.source === "user-db" : q.source !== "user-db");
}

function SourceSelect({ value, onChange }: { value: Source; onChange(s: Source): void }) {
  return (
    <label className="check">
      источник
      <select value={value} onChange={(e) => onChange(e.target.value as Source)}>
        <option value="all">все</option>
        <option value="course">вопросы курса</option>
        <option value="user-db">с собеседований</option>
      </select>
    </label>
  );
}

// ---------------------------------------------------------------- повторение (SM-2)

function ReviewMode({ data, onGraded }: { data: InterviewData; onGraded(c: Card): void }) {
  const [src, setSrcState] = useState<Source>(() => load("src", "all"));
  const [newLimit, setNewLimitState] = useState<number>(() => load("newLimit", 20));
  const [offCourse, setOffCourseState] = useState<boolean>(() => load("offCourse", true));
  const setSrc = (v: Source) => (setSrcState(v), save("src", v));
  const setNewLimit = (v: number) => (setNewLimitState(v), save("newLimit", v));
  const setOffCourse = (v: boolean) => (setOffCourseState(v), save("offCourse", v));

  // Очередь фиксируется при старте сессии, чтобы карточка не исчезала сразу после оценки.
  const [queue, setQueue] = useState<string[] | null>(null);
  const [pos, setPos] = useState(0);
  const [tally, setTally] = useState<Record<Grade, number>>({ again: 0, hard: 0, good: 0, easy: 0 });

  const candidates = useMemo(() => {
    const pool = data.questions.filter((q) => bySource(q, src) && (offCourse || q.topic !== "other"));
    const due = pool
      .filter((q) => q.card.due && !q.card.is_new)
      .sort((a, b) => (a.card.due_at ?? "").localeCompare(b.card.due_at ?? ""));
    const fresh = pool.filter((q) => q.card.is_new).slice(0, newLimit);
    return { due, fresh };
  }, [data, src, newLimit, offCourse]);

  const byId = useMemo(() => Object.fromEntries(data.questions.map((q) => [q.id, q])), [data]);

  if (queue === null) {
    const n = candidates.due.length + candidates.fresh.length;
    return (
      <div className="iv-panel">
        <div className="viz-controls">
          <SourceSelect value={src} onChange={setSrc} />
          <label className="check">
            новых за сессию
            <select value={newLimit} onChange={(e) => setNewLimit(+e.target.value)}>
              {NEW_LIMITS.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={offCourse} onChange={(e) => setOffCourse(e.target.checked)} /> вопросы вне тем
            курса (RAG, агенты, …)
          </label>
        </div>
        <p>
          К повторению: <b>{candidates.due.length}</b>, новых в сессии: <b>{candidates.fresh.length}</b>.
        </p>
        <button
          className="btn primary"
          disabled={n === 0}
          onClick={() => {
            setQueue([...candidates.due, ...candidates.fresh].map((q) => q.id));
            setPos(0);
            setTally({ again: 0, hard: 0, good: 0, easy: 0 });
          }}
        >
          {n ? `Начать (${n})` : "На сегодня всё повторено"}
        </button>
        <p className="muted iv-help">
          Ответь на вопрос сам, открой эталон и честно оцени себя. «Не помню» вернёт карточку через 10 минут, остальные
          оценки — через растущие интервалы (алгоритм SM-2: 1 день, 6 дней, дальше × лёгкость карточки).
        </p>
      </div>
    );
  }

  if (pos >= queue.length) {
    return (
      <div className="iv-panel">
        <h3>Сессия завершена</h3>
        <p>
          {(Object.keys(tally) as Grade[]).map((g) => `${GRADE_LABEL[g]}: ${tally[g]}`).join(" · ")}
        </p>
        <button className="btn" onClick={() => setQueue(null)}>
          К настройкам
        </button>
      </div>
    );
  }

  const q = byId[queue[pos]];
  return (
    <div className="iv-panel">
      <div className="iv-progress">
        <span>
          {pos + 1} / {queue.length}
        </span>
        <div className="iv-bar">
          <div style={{ width: `${(pos / queue.length) * 100}%` }} />
        </div>
        <button className="link-btn" onClick={() => setQueue(null)}>
          закончить
        </button>
      </div>
      <ReviewCard
        key={`${q.id}-${pos}`}
        q={q}
        card={q.card}
        mode="review"
        onGraded={(card, g) => {
          onGraded(card);
          setTally((t) => ({ ...t, [g]: t[g] + 1 }));
          if (g === "again") setQueue((qq) => (qq ? [...qq, q.id] : qq)); // ещё раз в конце сессии
          setPos((p) => p + 1);
        }}
      />
    </div>
  );
}

// ---------------------------------------------------------------- квиз по модулю

function QuizMode({ data, onGraded }: { data: InterviewData; onGraded(c: Card): void }) {
  const [module, setModuleState] = useState<string>(() => load("quizModule", data.modules[0]?.id ?? OFF_COURSE));
  const setModule = (m: string) => (setModuleState(m), save("quizModule", m));
  const [src, setSrc] = useState<Source>("all");
  const [started, setStarted] = useState(false);
  const [pos, setPos] = useState(0);
  const [tally, setTally] = useState<Record<Grade, number>>({ again: 0, hard: 0, good: 0, easy: 0 });

  const list = useMemo(
    () =>
      data.questions.filter(
        (q) => (module === OFF_COURSE ? q.modules.length === 0 : q.modules.includes(module)) && bySource(q, src),
      ),
    [data, module, src],
  );

  if (!started || pos >= list.length) {
    const finished = started && pos >= list.length;
    return (
      <div className="iv-panel">
        {finished && (
          <>
            <h3>Квиз пройден</h3>
            <p>{(Object.keys(tally) as Grade[]).map((g) => `${GRADE_LABEL[g]}: ${tally[g]}`).join(" · ")}</p>
          </>
        )}
        <div className="viz-controls">
          <label className="check">
            модуль
            <select value={module} onChange={(e) => setModule(e.target.value)}>
              {data.modules.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.title}
                </option>
              ))}
              <option value={OFF_COURSE}>Без привязки к задачам (RAG, агенты, …)</option>
            </select>
          </label>
          <SourceSelect value={src} onChange={setSrc} />
        </div>
        <p>
          Вопросов: <b>{list.length}</b> (из них с собеседований: {list.filter((q) => q.source === "user-db").length}).
        </p>
        <button
          className="btn primary"
          disabled={!list.length}
          onClick={() => {
            setStarted(true);
            setPos(0);
            setTally({ again: 0, hard: 0, good: 0, easy: 0 });
          }}
        >
          {finished ? "Пройти ещё раз" : "Начать квиз"}
        </button>
        <p className="muted iv-help">Все вопросы модуля по порядку. Оценки тоже идут в интервальное повторение.</p>
      </div>
    );
  }

  const q = list[pos];
  return (
    <div className="iv-panel">
      <div className="iv-progress">
        <span>
          {pos + 1} / {list.length}
        </span>
        <div className="iv-bar">
          <div style={{ width: `${(pos / list.length) * 100}%` }} />
        </div>
        <button className="link-btn" onClick={() => setPos(list.length)}>
          закончить
        </button>
      </div>
      <ReviewCard
        key={q.id}
        q={q}
        n={pos + 1}
        card={q.card}
        mode="quiz"
        onGraded={(card, g) => {
          onGraded(card);
          setTally((t) => ({ ...t, [g]: t[g] + 1 }));
          setPos((p) => p + 1);
        }}
      />
    </div>
  );
}

// ---------------------------------------------------------------- все вопросы

function AllMode({ data }: { data: InterviewData }) {
  const [text, setText] = useState("");
  const [src, setSrc] = useState<Source>("all");
  const [topic, setTopic] = useState("");
  const topics = useMemo(() => [...new Set(data.questions.map((q) => q.topic))].sort(), [data]);
  const shown = data.questions.filter(
    (q) =>
      bySource(q, src) &&
      (!topic || q.topic === topic) &&
      (!text || `${q.id} ${q.question} ${q.tags.join(" ")}`.toLowerCase().includes(text.toLowerCase())),
  );
  return (
    <div className="iv-panel">
      <div className="viz-controls">
        <input className="iv-search" placeholder="поиск по тексту, id, тегам" value={text} onChange={(e) => setText(e.target.value)} />
        <SourceSelect value={src} onChange={setSrc} />
        <label className="check">
          тема
          <select value={topic} onChange={(e) => setTopic(e.target.value)}>
            <option value="">все</option>
            {topics.map((t) => (
              <option key={t} value={t}>
                {t === "other" ? "other (вне курса)" : t}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">{shown.length}</span>
      </div>
      <ul className="iv-list">
        {shown.map((q) => (
          <li key={q.id}>
            <a href={questionHref(q.id)}>
              <span className="iv-list-q">{q.question}</span>
              <span className="iv-list-meta">
                <span className={`chip diff-${q.difficulty}`}>{DIFFICULTY_LABEL[q.difficulty]}</span>
                {q.source === "user-db" && <span className="chip src-user">собес</span>}
                <span className="muted">{q.topic === "other" ? q.tags.join(", ") || "other" : q.topic}</span>
                <span className="iv-due">{cardStatus(q.card)}</span>
              </span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

function cardStatus(c: Card): string {
  if (c.is_new) return "новый";
  if (c.due) return "к повторению";
  const days = (new Date(c.due_at!).getTime() - Date.now()) / 86400000;
  return `через ${fmtInterval(days)}`;
}

function CardState({ card }: { card: Card }) {
  if (card.is_new) return <p className="muted">Ещё не повторялся.</p>;
  return (
    <p className="muted">
      Повторений подряд: {card.reps}, забываний: {card.lapses}, лёгкость {card.ease.toFixed(2)}; следующий показ —{" "}
      {cardStatus(card)}.
    </p>
  );
}
