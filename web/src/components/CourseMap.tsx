import { useEffect, useState } from "react";
import { api, type CourseData } from "../api";
import { TYPE_ICON, taskLabel, withDebugUnderParent } from "../courseIndex";
import { taskHref } from "../router";
import { usePersistent } from "../theme";
import { DepGraph } from "./DepGraph";
import { DIFFICULTY_LABEL, StatusIcon, TYPE_LABEL } from "./StatusIcon";

export function CourseMap() {
  const [data, setData] = useState<CourseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showGraph, setShowGraph] = usePersistent("llmlab.map.graph", true);

  useEffect(() => {
    api.course().then(setData, (e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <div className="page-msg error">Не удалось загрузить курс: {error}</div>;
  if (!data) return <div className="page-msg">Загрузка…</div>;

  const all = data.blocks.flatMap((b) => b.modules.flatMap((m) => m.tasks));
  const solved = all.filter((t) => t.progress.status === "solved").length;
  const byId = new Map(all.map((t) => [t.id, t]));
  const next = all.find((t) => t.progress.status !== "solved" && t.depends_on.every((d) => all.find((x) => x.id === d)?.progress.status === "solved"));

  return (
    <div className="map">
      <div className="map-head">
        <div>
          <h1>Карта курса</h1>
          <div className="muted">
            Решено {solved} из {all.length}
            {next && (
              <>
                {" · "}следующая: <a href={taskHref(next.id)}>{taskLabel(next)}</a>
              </>
            )}
          </div>
        </div>
        <div className="progress-bar" title={`${solved}/${all.length}`}>
          <div style={{ width: `${all.length ? (100 * solved) / all.length : 0}%` }} />
        </div>
      </div>

      {data.issues.filter((i) => i.level === "error").length > 0 && (
        <details className="issues">
          <summary>Ошибки в контенте курса: {data.issues.filter((i) => i.level === "error").length} (см. `llmlab validate --all`)</summary>
          <ul>
            {data.issues.map((i, k) => (
              <li key={k} className={i.level}>
                <code>{i.where}</code>: {i.message}
              </li>
            ))}
          </ul>
        </details>
      )}

      {data.blocks.map((block) => {
        const tasks = block.modules.flatMap((m) => m.tasks);
        return (
          <section key={block.id} className="block">
            <h2>
              {block.title}
            </h2>
            {block.description && <p className="muted block-desc">{block.description}</p>}

            {tasks.length > 0 && (
              <div className="graph-section">
                <button className="link-btn" onClick={() => setShowGraph(!showGraph)}>
                  {showGraph ? "▾" : "▸"} Граф зависимостей
                </button>
                {showGraph && <DepGraph tasks={tasks.filter((t) => t.type !== "debug")} />}
              </div>
            )}

            {block.modules.map((m, mi) => (
              <div key={m.id} className={`module ${m.tasks.length ? "" : "empty"}`}>
                <div className="module-head">
                  <h3>
                    <span className="module-num" title={`Модуль ${mi}`}>{mi}</span>
                    {m.title}
                  </h3>
                  <span className="muted">{m.description}</span>
                  {m.tasks.length > 0 ? (
                    <span className="module-count">
                      {m.tasks.filter((t) => t.progress.status === "solved").length}/{m.tasks.length}
                    </span>
                  ) : (
                    <span className="module-count muted">скоро</span>
                  )}
                </div>
                {m.tasks.length > 0 && (
                  <div className="task-list">
                    {withDebugUnderParent(m.tasks).map((t) => {
                      const baseline = t.diff_from ? block.baselines[t.diff_from] : undefined;
                      const deps = t.depends_on.filter((d) => d !== t.parent);
                      return (
                        <a key={t.id} className={`task-row ${t.parent ? "sub" : ""}`} href={taskHref(t.id)}>
                          <StatusIcon status={t.progress.status} />
                          <span className={`task-num ${t.parent || !t.number ? "icon" : ""}`}>
                            {t.parent || !t.number ? TYPE_ICON[t.type] : t.number}
                          </span>
                          <span className="task-title">{t.parent ? capitalize(t.title.replace(/^Отладка:\s*/, "")) : t.title}</span>
                          <span className="task-meta">
                            {t.diff_from && (
                              <span className="chip diff" title="С чем сравниваем: что поменялось">
                                Δ {byId.get(t.diff_from)?.short_title ?? baseline?.title ?? t.diff_from}
                              </span>
                            )}
                            {deps.length > 0 && (
                              <span className="chip" title={`Сначала: ${deps.map((d) => (byId.get(d) ? taskLabel(byId.get(d)!) : d)).join(", ")}`}>
                                ← {deps.map((d) => byId.get(d)?.number ?? byId.get(d)?.short_title ?? d).join(", ")}
                              </span>
                            )}
                            <span className={`chip type-${t.type}`}>{TYPE_LABEL[t.type]}</span>
                            <span className={`chip diff-${t.difficulty}`}>{DIFFICULTY_LABEL[t.difficulty]}</span>
                            <span className="muted mono">{t.time_estimate_min}′</span>
                            {t.progress.attempts > 0 && <span className="muted mono" title="попыток">×{t.progress.attempts}</span>}
                          </span>
                        </a>
                      );
                    })}
                  </div>
                )}
              </div>
            ))}
          </section>
        );
      })}
    </div>
  );
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
