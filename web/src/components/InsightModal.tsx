// Модалка с обязательным полем «Главное, что я понял» (1–3 строки). Без него конспект не создаётся:
// своими словами сформулированный вывод — самая ценная часть конспекта.
import { useEffect, useRef, useState } from "react";
import { api } from "../api";

const MAX_LINES = 3;
const MAX_CHARS = 600;

export function InsightModal(props: {
  taskId: string;
  title: string;
  initial?: string;
  mode: "create" | "edit";
  onDone(path: string): void;
  onClose(): void;
}) {
  const [text, setText] = useState(props.initial ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);

  const onClose = useRef(props.onClose);
  onClose.current = props.onClose;
  useEffect(() => {
    area.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose.current();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const lines = text.split("\n").filter((l) => l.trim()).length;
  const chars = text.trim().length;
  const problem =
    lines === 0 ? "Напиши хотя бы одну строку" : lines > MAX_LINES ? `Не больше ${MAX_LINES} строк` : chars > MAX_CHARS ? `Не больше ${MAX_CHARS} символов` : null;

  const submit = async () => {
    if (problem || busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.saveNote(props.taskId, text);
      props.onDone(r.path);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="insight-title">
        <div className="modal-head">
          {props.mode === "create" && <div className="modal-kicker">✓ Все тесты прошли</div>}
          <h2 id="insight-title">{props.mode === "create" ? "Главное, что я понял" : "Изменить вывод"}</h2>
          <div className="muted">
            {props.taskId} — {props.title}
          </div>
        </div>
        <p className="modal-text">
          {props.mode === "create"
            ? "Сформулируй своими словами 1–3 строки: что за механизм, зачем он нужен, где была ловушка. Это попадёт в конспект; остальное (формулы, твой код, ошибки по пути) соберётся автоматически."
            : "Вывод заменит раздел «Главное, что я понял» в конспекте; остальное пересоберётся."}
        </p>
        <textarea
          ref={area}
          value={text}
          rows={4}
          placeholder={"Например: маску применяют до softmax, иначе знаменатель «видит» будущее.\nДеление на √D держит дисперсию логитов около 1."}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              void submit();
            }
          }}
        />
        <div className="modal-foot">
          <span className={`muted mono ${problem && chars > 0 ? "warn-text" : ""}`}>
            {lines}/{MAX_LINES} строк · {chars}/{MAX_CHARS}
            {problem && chars > 0 ? ` — ${problem}` : ""}
          </span>
          {error && <span className="error-text">{error}</span>}
          <span className="spacer" />
          <button className="btn" onClick={props.onClose}>
            {props.mode === "create" ? "Позже" : "Отмена"}
          </button>
          <button className="btn primary" onClick={submit} disabled={!!problem || busy} title="Ctrl/⌘ + Enter">
            {busy && <span className="spinner" />}
            {props.mode === "create" ? "Создать конспект" : "Сохранить"}
          </button>
        </div>
      </div>
    </div>
  );
}
