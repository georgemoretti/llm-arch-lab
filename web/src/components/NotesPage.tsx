// Вкладка «Конспекты»: дерево блок → модуль → конспект, поиск (FTS5), фильтр по тегам,
// просмотр конспекта, граф «кто от кого произошёл», таблица архитектур, экспорт vault.
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  NOTES_EXPORT_URL,
  type ArchTable,
  type CourseData,
  type NoteDetail,
  type NotesList,
  type SearchHit,
  type TaskSummary,
} from "../api";
import { NOTES_ARCH_HREF, NOTES_GRAPH_HREF, noteHref, taskHref, type NotesView } from "../router";
import { usePersistent } from "../theme";
import { DepGraph } from "./DepGraph";
import { InsightModal } from "./InsightModal";
import { Highlighted, NoteMarkdown } from "./Markdown";

export function NotesPage({ view }: { view: NotesView }) {
  const [list, setList] = useState<NotesList | null>(null);
  const [course, setCourse] = useState<CourseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [tags, setTags] = usePersistent<string[]>("llmlab.notes.tags", []);

  const reload = useCallback(
    () => Promise.all([api.notes().then(setList), api.course().then(setCourse)]).catch((e) => setError(e.message)),
    [],
  );
  useEffect(() => {
    void reload();
  }, [reload]);

  // Поиск с небольшой задержкой, чтобы не дёргать сервер на каждую букву.
  useEffect(() => {
    if (!query.trim()) return setHits(null);
    const timer = window.setTimeout(() => {
      api.searchNotes(query).then((r) => setHits(r.results), () => setHits([]));
    }, 180);
    return () => window.clearTimeout(timer);
  }, [query]);

  const byId = useMemo(() => new Map((list?.notes ?? []).map((n) => [n.task_id, n])), [list]);
  const knownTags = new Set(list?.tags.map((t) => t.tag));
  const activeTags = tags.filter((t) => knownTags.has(t));
  const matchesTags = (id: string) => activeTags.every((t) => byId.get(id)?.tags.includes(t));
  const toggleTag = (t: string) => setTags(activeTags.includes(t) ? activeTags.filter((x) => x !== t) : [...activeTags, t]);

  if (error) return <div className="page-msg error">{error}</div>;
  if (!list || !course) return <div className="page-msg">Загрузка…</div>;

  const activeId = view.kind === "note" ? view.id : null;

  return (
    <div className="notes-page">
      <aside className="notes-side">
        <input
          className="search"
          type="search"
          placeholder="Поиск по конспектам…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Поиск по конспектам"
        />
        <nav className="notes-views">
          <a href="#/notes" className={view.kind === "home" ? "active" : ""}>
            Обзор
          </a>
          <a href={NOTES_GRAPH_HREF} className={view.kind === "graph" ? "active" : ""}>
            Граф связей
          </a>
          <a href={NOTES_ARCH_HREF} className={view.kind === "arch" ? "active" : ""}>
            Таблица архитектур
          </a>
        </nav>
        {list.tags.length > 0 && (
          <div className="tag-filter">
            {list.tags.map((t) => (
              <button key={t.tag} className={`tag ${activeTags.includes(t.tag) ? "on" : ""}`} onClick={() => toggleTag(t.tag)}>
                #{t.tag} <span className="muted">{t.count}</span>
              </button>
            ))}
          </div>
        )}
        <div className="notes-list">
          {hits ? (
            <SearchResults hits={hits.filter((h) => matchesTags(h.task_id))} activeId={activeId} />
          ) : (
            <NotesTree course={course} list={list} activeId={activeId} filter={matchesTags} />
          )}
        </div>
        <div className="notes-side-foot">
          <a className="btn small" href={NOTES_EXPORT_URL} download title="zip с папкой Markdown-файлов: распакуй и открой в Obsidian как vault">
            ⇩ Экспорт vault (.zip)
          </a>
          <RebuildButton onDone={reload} />
        </div>
      </aside>
      <section className="notes-main">
        {view.kind === "note" && <NoteView key={view.id} id={view.id} onChanged={reload} />}
        {view.kind === "graph" && <NotesGraph course={course} noteIds={new Set(byId.keys())} />}
        {view.kind === "arch" && <ArchView />}
        {view.kind === "home" && <NotesHome list={list} course={course} />}
      </section>
    </div>
  );
}

function codeTasks(course: CourseData): TaskSummary[] {
  return course.blocks.flatMap((b) => b.modules.flatMap((m) => m.tasks)).filter((t) => t.has_code);
}

function NotesTree(props: { course: CourseData; list: NotesList; activeId: string | null; filter(id: string): boolean }) {
  const ids = new Set(props.list.notes.map((n) => n.task_id));
  const shown = props.course.blocks
    .map((b) => ({
      block: b,
      modules: b.modules
        .map((m) => ({ module: m, tasks: m.tasks.filter((t) => ids.has(t.id) && props.filter(t.id)) }))
        .filter((m) => m.tasks.length),
    }))
    .filter((b) => b.modules.length);

  if (!props.list.notes.length) return <p className="muted small-text">Конспектов пока нет.</p>;
  if (!shown.length) return <p className="muted small-text">Нет конспектов с выбранными тегами.</p>;
  return (
    <div className="tree">
      {shown.map(({ block, modules }) => (
        <div key={block.id}>
          <div className="tree-block">
            <span className="block-id">{block.id}</span> {block.title}
          </div>
          {modules.map(({ module, tasks }) => (
            <div key={module.id} className="tree-module">
              <div className="tree-module-title muted">{module.title}</div>
              {tasks.map((t) => (
                <a key={t.id} href={noteHref(t.id)} className={`tree-note ${props.activeId === t.id ? "active" : ""}`}>
                  <span className="mono">{t.id.split("-")[0]}</span> {t.title}
                </a>
              ))}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function SearchResults({ hits, activeId }: { hits: SearchHit[]; activeId: string | null }) {
  if (!hits.length) return <p className="muted small-text">Ничего не найдено.</p>;
  return (
    <div className="hits">
      {hits.map((h) => (
        <a key={h.path} href={noteHref(h.task_id)} className={`hit ${activeId === h.task_id ? "active" : ""}`}>
          <div className="hit-title">
            <span className="mono muted">{h.task_id}</span> {h.title}
          </div>
          <div className="hit-snippet">
            <Highlighted text={h.snippet} />
          </div>
        </a>
      ))}
    </div>
  );
}

function RebuildButton({ onDone }: { onDone(): Promise<unknown> }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    try {
      const r = await api.rebuildNotes();
      setMsg(`Пересобрано: ${r.built.length}${r.skipped.length ? `, пропущено: ${r.skipped.length}` : ""}`);
      await onDone();
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <button className="btn small" onClick={run} disabled={busy} title="Пересобрать все конспекты из summary.md, кода и истории попыток. Твои выводы сохранятся.">
        {busy ? <span className="spinner" /> : "↻"} Пересобрать
      </button>
      {msg && <span className="muted small-text">{msg}</span>}
    </>
  );
}

function NotesHome({ list, course }: { list: NotesList; course: CourseData }) {
  const total = codeTasks(course).length;
  const recent = [...list.notes].sort((a, b) => b.mtime_ns - a.mtime_ns).slice(0, 6);
  return (
    <div className="notes-home">
      <h1>Конспекты</h1>
      <p className="muted">
        {list.notes.length} из {total} задач с кодом. Конспект создаётся после первого зелёного прогона: авторская выжимка,
        твой код, отличия от эталона, ошибки по пути и твой вывод.
      </p>
      {list.notes.length === 0 ? (
        <div className="empty-card">
          Пока пусто. Реши любую задачу на <a href="#/">карте курса</a> — после прохождения всех тестов появится конспект.
        </div>
      ) : (
        <>
          <h3>Недавние</h3>
          <div className="note-cards">
            {recent.map((n) => (
              <a key={n.task_id} href={noteHref(n.task_id)} className="note-card">
                <div className="mono muted">{n.task_id}</div>
                <div className="note-card-title">{n.title}</div>
                <div className="muted small-text">
                  {n.solved_at && `решено ${n.solved_at}`}
                  {n.attempts ? ` · попыток: ${n.attempts}` : ""}
                </div>
              </a>
            ))}
          </div>
        </>
      )}
      <h3>Obsidian</h3>
      <p className="muted">
        Конспекты — обычные Markdown-файлы в <code>{list.dir}</code>. Открой эту папку в Obsidian как vault или скачай zip
        кнопкой «Экспорт vault». Раздел «Главное, что я понял» можно править прямо в файле — при пересборке он сохранится.
      </p>
    </div>
  );
}

function NoteView({ id, onChanged }: { id: string; onChanged(): Promise<unknown> }) {
  const [note, setNote] = useState<NoteDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [copied, setCopied] = useState(false);

  const load = useCallback(() => api.note(id).then(setNote, (e) => setError(e.message)), [id]);
  useEffect(() => {
    void load();
  }, [load]);

  if (error)
    return (
      <div className="page-msg">
        {error}. <a href={taskHref(id)}>Открыть задачу {id}</a>
      </div>
    );
  if (!note) return <div className="page-msg">Загрузка…</div>;

  const copyPath = async () => {
    try {
      await navigator.clipboard.writeText(note.abs_path);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* буфер обмена недоступен */
    }
  };

  return (
    <article className="note-view">
      <div className="note-toolbar">
        <span className="mono muted">{note.task_id}</span>
        {note.tags.map((t) => (
          <span key={t} className="chip">
            #{t}
          </span>
        ))}
        {note.solved_at && <span className="muted small-text">решено {note.solved_at}</span>}
        {note.attempts ? <span className="muted small-text">· попыток: {note.attempts}</span> : null}
        <span className="spacer" />
        <a className="btn small" href={taskHref(note.task_id)}>
          К задаче
        </a>
        <button className="btn small" onClick={() => setEditing(true)}>
          Изменить вывод
        </button>
        <button className="btn small" onClick={copyPath} title={note.abs_path}>
          {copied ? "Скопировано" : "Путь к файлу"}
        </button>
      </div>
      <NoteMarkdown text={note.body} />
      {editing && (
        <InsightModal
          taskId={note.task_id}
          title={note.title}
          mode="edit"
          initial={note.insight ?? ""}
          onClose={() => setEditing(false)}
          onDone={() => {
            setEditing(false);
            void load();
            void onChanged();
          }}
        />
      )}
    </article>
  );
}

function NotesGraph({ course, noteIds }: { course: CourseData; noteIds: Set<string> }) {
  // Ребро «родитель → потомок»: diff_from (если это задача) и depends_on.
  const tasks = codeTasks(course).map((t) => ({
    ...t,
    depends_on: [...new Set([...t.depends_on, ...(t.diff_from && t.diff_from !== t.id ? [t.diff_from] : [])])],
  }));
  return (
    <div className="notes-graph">
      <h1>Граф связей</h1>
      <p className="muted">
        Кто от кого произошёл: стрелка ведёт от родителя (<code>diff_from</code>, <code>depends_on</code>) к задаче. Яркие узлы —
        задачи с конспектом, клик открывает конспект; бледные — ещё не решены.
      </p>
      {tasks.length ? (
        <DepGraph
          tasks={tasks}
          label="Граф связей конспектов"
          hrefFor={(id) => (noteIds.has(id) ? noteHref(id) : taskHref(id))}
          classFor={(t) => (noteIds.has(t.id) ? "has-note" : "no-note")}
        />
      ) : (
        <p className="muted">В курсе пока нет задач с кодом.</p>
      )}
    </div>
  );
}

function ArchView() {
  const [table, setTable] = useState<ArchTable | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.architectures().then(setTable, (e) => setError(e.message));
  }, []);
  if (error) return <div className="page-msg error">{error}</div>;
  if (!table) return <div className="page-msg">Загрузка…</div>;
  return (
    <div className="arch">
      <h1>Таблица архитектур</h1>
      <p className="muted">
        Ячейка открывается, когда решена задача, которая объясняет этот компонент. Открыто {table.opened} из {table.total}{" "}
        доступных сейчас; остальные строки заполнятся в следующих модулях курса.
      </p>
      <div className="arch-scroll">
        <table className="arch-table">
          <thead>
            <tr>
              <th>Модель</th>
              {table.columns.map((c) => (
                <th key={c.id}>{c.title}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((r) => (
              <tr key={r.id}>
                <th scope="row">
                  <div>{r.title}</div>
                  <div className="muted small-text">
                    {r.year}
                    {r.source && (
                      <>
                        {" · "}
                        <a href={r.source} target="_blank" rel="noreferrer" title="Источник гиперпараметров">
                          источник
                        </a>
                      </>
                    )}
                  </div>
                </th>
                {table.columns.map((c) => {
                  const cell = r.cells[c.id];
                  if (cell.status === "open")
                    return (
                      <td key={c.id} className="open">
                        {cell.value}
                        <a className="cell-src mono" href={noteHref(cell.task_id)} title={cell.task_title}>
                          {cell.task_id.split("-")[0]}
                        </a>
                      </td>
                    );
                  if (cell.status === "locked")
                    return (
                      <td key={c.id} className="locked">
                        <a href={taskHref(cell.task_id)} title={`Откроется после задачи «${cell.task_title}»`}>
                          🔒 {cell.task_id}
                        </a>
                      </td>
                    );
                  return (
                    <td key={c.id} className="empty" title={r.module_title ? `Модуль «${r.module_title}»` : undefined}>
                      ·
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small-text">🔒 — откроется после решения задачи · «·» — появится в модуле, указанном во всплывающей подсказке.</p>
    </div>
  );
}
