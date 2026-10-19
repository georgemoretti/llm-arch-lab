// Навигация по курсу со страницы задачи: выезжающая слева панель «модули → задачи» и кнопки
// «предыдущая / следующая», чтобы не возвращаться на карту ради перехода к соседней задаче.
import { useEffect, useRef, useState } from "react";
import { api, type CourseData, type TaskSummary } from "../api";
import { taskHref } from "../router";
import { TYPE_ICON, taskLabel, withDebugUnderParent } from "../courseIndex";
import { usePersistent } from "../theme";
import { StatusIcon } from "./StatusIcon";

type ModuleNav = { id: string; block: string; title: string; num: string; tasks: TaskSummary[] };

function useCourse(reloadKey: unknown): CourseData | null {
  const [data, setData] = useState<CourseData | null>(null);
  useEffect(() => {
    api.course().then(setData, () => setData(null));
  }, [reloadKey]);
  return data;
}

function modulesOf(data: CourseData): ModuleNav[] {
  return data.blocks.flatMap((b) =>
    b.modules.filter((m) => m.tasks.length).map((m) => ({ id: m.id, block: b.id, title: m.title, num: `${b.id}${m.number}`, tasks: withDebugUnderParent(m.tasks) })),
  );
}

/** Кнопка «☰ Курс» + панель и стрелки к соседним задачам. `progressKey` меняется после прогона — обновить статусы. */
export function CourseNav({ currentId, progressKey }: { currentId: string; progressKey: unknown }) {
  const [open, setOpen] = useState(false);
  const data = useCourse(progressKey);
  const modules = data ? modulesOf(data) : [];
  const flat = modules.flatMap((m) => m.tasks);
  const i = flat.findIndex((t) => t.id === currentId);
  const prev = i > 0 ? flat[i - 1] : undefined;
  const next = i >= 0 && i < flat.length - 1 ? flat[i + 1] : undefined;
  const currentBlock = modules.find((m) => m.tasks.some((t) => t.id === currentId))?.block;
  // в панели — модули одного блока (по умолчанию блок текущей задачи), другие — переключателем вверху
  const [block, setBlock] = useState<string | undefined>();
  const shownBlock = block ?? currentBlock ?? data?.blocks[0]?.id;
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!open) return setBlock(undefined);
    // текущая задача сразу в зоне видимости, без ручной прокрутки
    navRef.current?.querySelector('[aria-current="page"]')?.scrollIntoView({ block: "center" });
  }, [open, data]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  // переход по ссылке внутри панели закрывает её
  useEffect(() => setOpen(false), [currentId]);

  return (
    <>
      <span className="course-nav-btns">
        <button className="btn small" onClick={() => setOpen(!open)} aria-expanded={open} title="Модули и задачи курса">
          ☰ Курс
        </button>
        <a className={`btn small icon ${prev ? "" : "disabled"}`} href={prev ? taskHref(prev.id) : undefined} title={prev ? `Назад: ${taskLabel(prev)}` : "Это первая задача"}>
          ←
        </a>
        <a className={`btn small icon ${next ? "" : "disabled"}`} href={next ? taskHref(next.id) : undefined} title={next ? `Дальше: ${taskLabel(next)}` : "Это последняя задача"}>
          →
        </a>
      </span>
      {open && (
        <div className="course-nav-backdrop" onMouseDown={(e) => e.target === e.currentTarget && setOpen(false)}>
          <nav className="course-nav" aria-label="Модули и задачи курса" ref={navRef}>
            <div className="course-nav-head">
              <a href="#/">Карта курса</a>
              <span className="course-nav-spacer" />
              {data && data.blocks.length > 1 && (
                <span className="course-nav-blocks" role="tablist" aria-label="Блоки">
                  {data.blocks.map((b) => (
                    <button
                      key={b.id}
                      role="tab"
                      aria-selected={b.id === shownBlock}
                      className={`course-nav-block ${b.id === shownBlock ? "active" : ""} ${b.id === currentBlock ? "current" : ""}`}
                      onClick={() => setBlock(b.id)}
                      title={b.title}
                    >
                      {b.id}
                    </button>
                  ))}
                </span>
              )}
              <button className="icon-btn" onClick={() => setOpen(false)} aria-label="Закрыть">
                ×
              </button>
            </div>
            {!data ? (
              <div className="muted course-nav-empty">Загрузка…</div>
            ) : (
              <>
                {data.blocks.length > 1 && <div className="course-nav-block-title muted">{data.blocks.find((b) => b.id === shownBlock)?.title}</div>}
                {modules
                  .filter((m) => m.block === shownBlock)
                  .map((m) => (
                    <ModuleSection key={m.id} m={m} currentId={currentId} />
                  ))}
              </>
            )}
          </nav>
        </div>
      )}
    </>
  );
}

function ModuleSection({ m, currentId }: { m: ModuleNav; currentId: string }) {
  const hasCurrent = m.tasks.some((t) => t.id === currentId);
  // раскрытость модуля помним между открытиями; модуль текущей задачи раскрыт всегда при открытии
  const [expanded, setExpanded] = usePersistent(`llmlab.nav.${m.id}`, false);
  const isOpen = expanded || hasCurrent;
  const solved = m.tasks.filter((t) => t.progress.status === "solved").length;
  return (
    <section className={`course-nav-module ${hasCurrent ? "current" : ""}`}>
      <button className="course-nav-module-head" onClick={() => setExpanded(!isOpen)} aria-expanded={isOpen}>
        <span className="module-num">{m.num}</span>
        <span className="course-nav-module-title">{m.title}</span>
        <span className="muted mono">
          {solved}/{m.tasks.length}
        </span>
        <span className="muted">{isOpen ? "▾" : "▸"}</span>
      </button>
      {isOpen && (
        <div className="course-nav-tasks">
          {m.tasks.map((t) => (
            <a
              key={t.id}
              href={taskHref(t.id)}
              className={`course-nav-task ${t.id === currentId ? "active" : ""} ${t.parent ? "sub" : ""}`}
              aria-current={t.id === currentId ? "page" : undefined}
              title={t.title}
            >
              <StatusIcon status={t.progress.status} />
              <span className="mono muted course-nav-num">{t.parent || !t.number ? TYPE_ICON[t.type] : t.number}</span>
              <span className="course-nav-name">{t.short_title.replace(/^Найди баг: .*/, "Найди баг")}</span>
            </a>
          ))}
        </div>
      )}
    </section>
  );
}
