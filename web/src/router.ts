// Минимальный hash-роутер: #/ — карта курса, #/task/<id> — задача,
// #/notes — конспекты (#/notes/<id> — конспект, #/notes/~graph, #/notes/~arch — граф и таблица),
// #/interview — вопросы и повторение (#/interview/q/<qid> — один вопрос).
import { useEffect, useState } from "react";

export type NotesView = { kind: "note"; id: string } | { kind: "graph" } | { kind: "arch" } | { kind: "home" };
export type Route =
  | { name: "map" }
  | { name: "task"; id: string }
  | { name: "notes"; view: NotesView }
  | { name: "interview"; qid: string | null };

function parse(hash: string): Route {
  const task = hash.match(/^#\/task\/([^/?]+)/);
  if (task) return { name: "task", id: decodeURIComponent(task[1]) };
  const iv = hash.match(/^#\/interview(?:\/q\/([^/?]+))?/);
  if (iv) return { name: "interview", qid: iv[1] ? decodeURIComponent(iv[1]) : null };
  const notes = hash.match(/^#\/notes(?:\/([^/?]+))?/);
  if (notes) {
    const seg = notes[1] ? decodeURIComponent(notes[1]) : "";
    if (seg === "~graph") return { name: "notes", view: { kind: "graph" } };
    if (seg === "~arch") return { name: "notes", view: { kind: "arch" } };
    return { name: "notes", view: seg ? { kind: "note", id: seg } : { kind: "home" } };
  }
  return { name: "map" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const taskHref = (id: string) => `#/task/${encodeURIComponent(id)}`;
export const noteHref = (id: string) => `#/notes/${encodeURIComponent(id)}`;
export const NOTES_GRAPH_HREF = "#/notes/~graph";
export const NOTES_ARCH_HREF = "#/notes/~arch";
export const INTERVIEW_HREF = "#/interview";
export const questionHref = (qid: string) => `#/interview/q/${encodeURIComponent(qid)}`;
