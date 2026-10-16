import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type Attempt, type RunResult, type TaskDetail } from "../api";
import { noteHref, taskHref } from "../router";
import { InsightModal } from "./InsightModal";
import { usePersistent, type Theme } from "../theme";
import { CodeEditor, DiffView, type CodeEditorHandle, type ErrorMark } from "./CodeEditor";
import { Markdown, VizBlock } from "./Markdown";
import { DIFFICULTY_LABEL, StatusIcon, TYPE_LABEL } from "./StatusIcon";
import { TestResults } from "./TestResults";

type Tab = "readme" | "hints" | "reading" | "viz" | "attempts";
type SaveState = "saved" | "dirty" | "saving" | "conflict" | "error";

const AUTOSAVE_MS = 700;
const POLL_MS = 1500;

export function TaskPage({ id, theme }: { id: string; theme: Theme }) {
  const [task, setTask] = useState<TaskDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("readme");
  const [askInsight, setAskInsight] = useState(false);
  const [noteToast, setNoteToast] = useState<string | null>(null);

  const refresh = useCallback(() => api.task(id).then(setTask), [id]);

  useEffect(() => {
    setTask(null);
    setError(null);
    setTab("readme");
    refresh().catch((e) => setError(e.message));
  }, [id, refresh]);

  if (error) return <div className="page-msg error">{error}</div>;
  if (!task) return <div className="page-msg">Загрузка…</div>;

  const onNote = (note: RunResult["note"]) => {
    if (note?.status === "needs_insight") setAskInsight(true);
    else if (note?.status === "updated") setNoteToast("Конспект обновлён: свежий код и история попыток");
  };

  return (
    <div className="task-page">
      <TaskHeader task={task} onCreateNote={() => setAskInsight(true)} />
      {noteToast && (
        <div className="toast" role="status">
          {noteToast} — <a href={noteHref(task.id)}>открыть конспект</a>
          <button className="link-btn" onClick={() => setNoteToast(null)}>
            закрыть
          </button>
        </div>
      )}
      {askInsight && (
        <InsightModal
          taskId={task.id}
          title={task.title}
          mode="create"
          onClose={() => setAskInsight(false)}
          onDone={() => {
            setAskInsight(false);
            setNoteToast("Конспект создан");
            void refresh();
          }}
        />
      )}
      {task.has_code ? (
        <Workbench key={task.id} task={task} theme={theme} tab={tab} setTab={setTab} onProgress={refresh} onNote={onNote} />
      ) : (
        <div className="task-doc">
          <LeftPane task={task} tab={tab} setTab={setTab} attempts={[]} />
        </div>
      )}
    </div>
  );
}

function TaskHeader({ task, onCreateNote }: { task: TaskDetail; onCreateNote(): void }) {
  return (
    <div className="task-head">
      <StatusIcon status={task.progress.status} />
      <span className="task-id">{task.id}</span>
      <h1>{task.title}</h1>
      <span className={`chip type-${task.type}`}>{TYPE_LABEL[task.type]}</span>
      <span className={`chip diff-${task.difficulty}`}>{DIFFICULTY_LABEL[task.difficulty]}</span>
      <span className="muted mono">~{task.time_estimate_min} мин</span>
      {task.diff_from_info && (
        <span className="chip diff" title={task.diff_from_info.description ?? "С чем сравниваем"}>
          Δ{" "}
          {task.diff_from_info.kind === "task" ? (
            <a href={taskHref(task.diff_from_info.id)}>{task.diff_from_info.id}</a>
          ) : (
            task.diff_from_info.title
          )}
        </span>
      )}
      {task.depends_on.length > 0 && (
        <span className="muted head-deps">
          после:{" "}
          {task.depends_on.map((d, i) => (
            <span key={d}>
              {i > 0 && ", "}
              <a href={taskHref(d)}>{d}</a>
            </span>
          ))}
        </span>
      )}
      <span className="head-note">
        {task.note ? (
          <a className="btn small" href={noteHref(task.id)} title={`notes/${task.note.path}`}>
            📝 Конспект
          </a>
        ) : (
          task.has_code &&
          task.progress.status === "solved" && (
            <button className="btn small primary" onClick={onCreateNote} title="Задача решена, но конспекта ещё нет">
              📝 Создать конспект
            </button>
          )
        )}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------- левая панель

function LeftPane(props: { task: TaskDetail; tab: Tab; setTab(t: Tab): void; attempts: Attempt[] }) {
  const { task, tab, setTab, attempts } = props;
  const tabs: [Tab, string, number?][] = [
    ["readme", "Теория"],
    ["hints", "Подсказки", task.hints.length],
    ["reading", "Чтение", task.reading.length + task.papers.length],
    ["viz", "Визуализация", task.viz.length],
  ];
  if (task.has_code) tabs.push(["attempts", "Попытки", attempts.length]);

  return (
    <div className="left">
      <nav className="tabs" role="tablist">
        {tabs.map(([key, label, count]) => (
          <button key={key} role="tab" aria-selected={tab === key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>
            {label}
            {count !== undefined && count > 0 && <span className="count">{count}</span>}
          </button>
        ))}
      </nav>
      <div className="left-body">
        {tab === "readme" && <Markdown text={task.readme} viz={task.viz} />}
        {tab === "hints" && <Hints task={task} />}
        {tab === "reading" && <Reading task={task} />}
        {tab === "viz" &&
          (task.viz.length ? (
            task.viz.map((v) => (
              <div key={v.id}>
                <VizBlock id={v.id} info={v} />
                {v.description && <p className="muted">{v.description}</p>}
              </div>
            ))
          ) : (
            <p className="muted">У этой задачи пока нет визуализаций.</p>
          ))}
        {tab === "attempts" && <Attempts attempts={attempts} />}
      </div>
    </div>
  );
}

function Hints({ task }: { task: TaskDetail }) {
  const [shown, setShown] = usePersistent(`llmlab.hints.${task.id}`, 0);
  if (!task.hints.length) return <p className="muted">Подсказок нет.</p>;
  return (
    <div className="hints">
      <p className="muted">Открывай по одной: каждая следующая подсказка раскрывает больше.</p>
      {task.hints.slice(0, shown).map((h, i) => (
        <div key={i} className="hint">
          <div className="hint-title">{h.title}</div>
          <Markdown text={h.body} />
        </div>
      ))}
      {shown < task.hints.length ? (
        <button className="btn" onClick={() => setShown(shown + 1)}>
          Показать подсказку {shown + 1} из {task.hints.length}
        </button>
      ) : (
        <button className="link-btn" onClick={() => setShown(0)}>
          Скрыть все подсказки
        </button>
      )}
    </div>
  );
}

function Reading({ task }: { task: TaskDetail }) {
  if (!task.reading.length && !task.papers.length) return <p className="muted">Материалов для чтения нет.</p>;
  return (
    <div className="reading">
      {task.papers.length > 0 && (
        <>
          <h3>Статьи</h3>
          <ul>
            {task.papers.map((p) => (
              <li key={p.id}>
                <a href={p.url} target="_blank" rel="noreferrer">
                  {p.title}
                </a>
                <div className="muted">
                  {p.authors}, {p.year}
                  {p.venue ? ` · ${p.venue}` : ""}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
      {task.reading.length > 0 && (
        <>
          <h3>Ссылки</h3>
          <ul>
            {task.reading.map((r) => (
              <li key={r.url}>
                <a href={r.url} target="_blank" rel="noreferrer">
                  {r.title}
                </a>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function Attempts({ attempts }: { attempts: Attempt[] }) {
  if (!attempts.length) return <p className="muted">Попыток пока не было.</p>;
  return (
    <table className="attempts">
      <thead>
        <tr>
          <th>#</th>
          <th>когда</th>
          <th>статус</th>
          <th>тесты</th>
          <th>время</th>
        </tr>
      </thead>
      <tbody>
        {attempts.map((a, i) => (
          <tr key={a.id} className={a.status}>
            <td className="mono">{attempts.length - i}</td>
            <td>{new Date(a.created_at).toLocaleString()}</td>
            <td className={`st-${a.status}`}>{a.status}</td>
            <td className="mono">
              {a.passed}/{a.total}
            </td>
            <td className="mono">{a.duration_s.toFixed(1)} с</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- рабочее место: редактор + тесты

function Workbench(props: {
  task: TaskDetail;
  theme: Theme;
  tab: Tab;
  setTab(t: Tab): void;
  onProgress(): Promise<void>;
  onNote(note: RunResult["note"]): void;
}) {
  const { task, theme } = props;
  const editor = useRef<CodeEditorHandle>(null);
  const [initial, setInitial] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [notice, setNotice] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [running, setRunning] = useState(false);
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [diff, setDiff] = useState<{ mine: string; reference: string } | null>(null);
  const [split, setSplit] = usePersistent("llmlab.split", 0.42);
  const [resultsH, setResultsH] = usePersistent("llmlab.resultsH", 0.38);
  const [filePath, setFilePath] = useState("");

  // Состояние синхронизации с диском в ref'ах: его читают таймеры и команды Monaco.
  const baseSha = useRef<string | null>(null); // sha версии на диске, с которой совпадает/от которой начат редактор
  const dirty = useRef(false);
  const saving = useRef<Promise<boolean> | null>(null);
  const saveTimer = useRef<number | undefined>(undefined);
  const conflict = useRef(false);

  const flash = (msg: string) => {
    setNotice(msg);
    window.setTimeout(() => setNotice((n) => (n === msg ? null : n)), 4000);
  };

  useEffect(() => {
    api.code(task.id).then(
      (f) => {
        baseSha.current = f.sha256;
        setFilePath(f.path);
        setInitial(f.code);
      },
      (e) => setLoadError(e.message),
    );
    api.attempts(task.id).then((list) => {
      setAttempts(list);
      if (list[0]?.result) setResult(list[0].result);
    });
    return () => window.clearTimeout(saveTimer.current);
  }, [task.id]);

  const save = useCallback(
    async (force = false): Promise<boolean> => {
      window.clearTimeout(saveTimer.current);
      if (saving.current) await saving.current;
      if (!dirty.current && !force) return true;
      if (conflict.current && !force) return false;
      const text = editor.current?.getValue() ?? "";
      const p = (async () => {
        setSaveState("saving");
        try {
          const meta = await api.saveCode(task.id, text, baseSha.current, force);
          baseSha.current = meta.sha256;
          conflict.current = false;
          // Пока сохраняли, пользователь мог продолжить печатать.
          dirty.current = (editor.current?.getValue() ?? "") !== text;
          setSaveState(dirty.current ? "dirty" : "saved");
          return true;
        } catch (e) {
          if (e instanceof ApiError && e.status === 409) {
            conflict.current = true;
            setSaveState("conflict");
          } else {
            setSaveState("error");
          }
          return false;
        }
      })();
      saving.current = p;
      try {
        return await p;
      } finally {
        saving.current = null;
      }
    },
    [task.id],
  );

  const onChange = useCallback(() => {
    dirty.current = true;
    if (!conflict.current) setSaveState("dirty");
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => void save(), AUTOSAVE_MS);
  }, [save]);

  // Внешний IDE: опрашиваем sha файла. Чистый редактор — тихо подхватываем; есть несохранённые
  // правки — показываем конфликт и даём выбрать.
  useEffect(() => {
    if (initial === null) return;
    const timer = window.setInterval(async () => {
      if (document.hidden || saving.current || conflict.current) return;
      try {
        const meta = await api.codeMeta(task.id);
        if (!meta.sha256 || meta.sha256 === baseSha.current || saving.current) return;
        if (dirty.current) {
          conflict.current = true;
          setSaveState("conflict");
          return;
        }
        const f = await api.code(task.id);
        baseSha.current = f.sha256;
        editor.current?.replaceValue(f.code);
        flash("Подхвачены изменения файла из внешнего редактора");
      } catch {
        /* сервер перезапускается — попробуем в следующий раз */
      }
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [initial, task.id]);

  const takeDisk = async () => {
    const f = await api.code(task.id);
    baseSha.current = f.sha256;
    dirty.current = false;
    conflict.current = false;
    editor.current?.replaceValue(f.code);
    setSaveState("saved");
  };

  const keepMine = async () => {
    dirty.current = true;
    if (await save(true)) flash("Твоя версия записана в файл");
  };

  const run = useCallback(async () => {
    if (running) return;
    if (!(await save())) {
      flash("Код не сохранён — сначала разреши конфликт с файлом на диске");
      return;
    }
    setRunning(true);
    try {
      const r = await api.run(task.id);
      setResult(r);
      setAttempts(await api.attempts(task.id));
      await props.onProgress();
      if (r.status === "passed") props.onNote(r.note);
    } catch (e) {
      flash(`Не удалось запустить тесты: ${(e as Error).message}`);
    } finally {
      setRunning(false);
    }
  }, [running, save, task.id, props]);

  // Ctrl/⌘+Enter работает и когда фокус не в редакторе (например, в README).
  const runRef = useRef(run);
  runRef.current = run;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Enter" || !(e.ctrlKey || e.metaKey)) return;
      if ((e.target as Element | null)?.closest?.(".monaco-editor")) return; // там своя команда
      e.preventDefault();
      void runRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const reset = async () => {
    const what = task.type === "debug" ? "исходному багованному коду" : "заготовке";
    if (!window.confirm(`Сбросить твой код к ${what}? Текущая версия будет потеряна (история попыток останется).`)) return;
    window.clearTimeout(saveTimer.current);
    const f = await api.reset(task.id);
    baseSha.current = f.sha256;
    dirty.current = false;
    conflict.current = false;
    editor.current?.replaceValue(f.code);
    setSaveState("saved");
    setDiff(null);
  };

  const toggleDiff = async () => {
    if (diff) return setDiff(null);
    await save();
    try {
      const ref = await api.reference(task.id);
      setDiff({ mine: editor.current?.getValue() ?? "", reference: ref.code });
    } catch (e) {
      flash((e as Error).message);
    }
  };

  // Ошибки на строках кода пользователя — подсветка в редакторе.
  const errors: ErrorMark[] = useMemo(() => {
    if (!result) return [];
    const byLine = new Map<number, string[]>();
    for (const t of result.tests) {
      if ((t.outcome === "failed" || t.outcome === "error") && t.location) {
        const list = byLine.get(t.location.line) ?? [];
        list.push(`**${t.title}**\n\n${t.message ?? ""}`);
        byLine.set(t.location.line, list);
      }
    }
    return [...byLine].map(([line, msgs]) => ({ line, message: msgs.join("\n\n---\n\n") }));
  }, [result]);

  const left = useRef<HTMLDivElement>(null);
  const dragX = (e: React.PointerEvent) => startDrag(e, "x", left.current?.parentElement ?? null, (f) => setSplit(clamp(f, 0.22, 0.7)));
  const dragY = (e: React.PointerEvent) =>
    startDrag(e, "y", left.current?.parentElement?.querySelector(".right") ?? null, (f) => setResultsH(clamp(1 - f, 0.12, 0.8)));

  const diffLeft = task.diff_unlock_after - task.failed_attempts;

  return (
    <div className="workbench" style={{ gridTemplateColumns: `${split * 100}% 6px 1fr` }}>
      <div ref={left} className="left-wrap">
        <LeftPane task={task} tab={props.tab} setTab={props.setTab} attempts={attempts} />
      </div>
      <div className="splitter-x" onPointerDown={dragX} title="Перетащи, чтобы изменить ширину" />
      <div className="right" style={{ gridTemplateRows: `auto ${(1 - resultsH) * 100}% 6px 1fr` }}>
        <div className="toolbar">
          <button className="btn primary" onClick={run} disabled={running || initial === null} title="Ctrl/⌘ + Enter">
            {running ? <span className="spinner" /> : "▶"} Run
          </button>
          <button className="btn" onClick={reset} disabled={running}>
            Reset
          </button>
          <button
            className={`btn ${diff ? "active" : ""}`}
            onClick={toggleDiff}
            disabled={!task.diff_unlocked}
            title={task.diff_unlocked ? "Сравнить с эталонным решением" : `Откроется после решения или ещё ${diffLeft} неудачных попыток`}
          >
            {task.diff_unlocked ? "" : "🔒 "}Diff с эталоном
          </button>
          <span className="toolbar-file mono muted" title="Этот файл можно редактировать и во внешнем IDE">
            {filePath}
          </span>
          <SaveBadge state={saveState} />
        </div>
        <div className="editor-area">
          {saveState === "conflict" && (
            <div className="banner warn">
              Файл на диске изменён во внешнем редакторе, а здесь есть несохранённые правки.
              <button className="btn small" onClick={takeDisk}>
                Взять версию с диска
              </button>
              <button className="btn small" onClick={keepMine}>
                Оставить мою
              </button>
            </div>
          )}
          {notice && <div className="banner info">{notice}</div>}
          {loadError && <div className="page-msg error">{loadError}</div>}
          {initial !== null && (
            <div className={`editor-stack ${diff ? "hidden" : ""}`}>
              <CodeEditor ref={editor} initialValue={initial} theme={theme} errors={errors} onChange={onChange} onRun={run} onSave={() => void save()} />
            </div>
          )}
          {diff && (
            <div className="editor-stack">
              <div className="diff-legend muted">
                слева — твой код, справа — эталон
                <button className="link-btn" onClick={() => setDiff(null)}>
                  закрыть
                </button>
              </div>
              <DiffView mine={diff.mine} reference={diff.reference} theme={theme} />
            </div>
          )}
        </div>
        <div className="splitter-y" onPointerDown={dragY} />
        <TestResults result={result} running={running} onReveal={(line) => {
          setDiff(null);
          editor.current?.revealLine(line);
        }} />
      </div>
    </div>
  );
}

function SaveBadge({ state }: { state: SaveState }) {
  const text: Record<SaveState, string> = {
    saved: "сохранено",
    dirty: "изменено…",
    saving: "сохраняю…",
    conflict: "конфликт с диском",
    error: "ошибка сохранения",
  };
  return <span className={`save-badge ${state}`}>{text[state]}</span>;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function startDrag(e: React.PointerEvent, axis: "x" | "y", container: Element | null, onFrac: (f: number) => void) {
  if (!container) return;
  e.preventDefault();
  const rect = container.getBoundingClientRect();
  const move = (ev: PointerEvent) =>
    onFrac(axis === "x" ? (ev.clientX - rect.left) / rect.width : (ev.clientY - rect.top) / rect.height);
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    document.body.classList.remove("dragging");
  };
  document.body.classList.add("dragging");
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}
