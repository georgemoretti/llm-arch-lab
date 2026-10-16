import { lazy, Suspense } from "react";
import { CourseMap } from "./components/CourseMap";
import { InterviewPage } from "./components/InterviewPage";
import { NotesPage } from "./components/NotesPage";
import { useRoute } from "./router";
import { useTheme } from "./theme";

// Monaco тяжёлый (~3 МБ) — грузим его только на странице задачи.
const TaskPage = lazy(() => import("./components/TaskPage").then((m) => ({ default: m.TaskPage })));

export function App() {
  const route = useRoute();
  const [theme, toggleTheme] = useTheme();

  return (
    <div className="app">
      <header className="topbar">
        <a className="logo" href="#/">
          <span className="logo-mark">LL</span> LLM Arch Lab
        </a>
        <nav className="crumbs">
          <a href="#/" className={route.name === "map" ? "active" : ""}>
            Карта курса
          </a>
          {route.name === "task" && (
            <>
              <span className="muted">/</span>
              <span className="mono">{route.id}</span>
            </>
          )}
          <span className="crumbs-sep" />
          <a href="#/notes" className={route.name === "notes" ? "active" : ""}>
            Конспекты
          </a>
          <a href="#/interview" className={route.name === "interview" ? "active" : ""}>
            Interview
          </a>
        </nav>
        <div className="topbar-right">
          <span className="muted kbd-hint">
            <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Enter</kbd> — Run
          </span>
          <button className="icon-btn" onClick={toggleTheme} title="Светлая/тёмная тема" aria-label="Переключить тему">
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </header>
      <main className={route.name === "task" ? "main-task" : route.name === "notes" ? "main-notes" : "main-map"}>
        {route.name === "task" ? (
          <Suspense fallback={<div className="page-msg">Загрузка редактора…</div>}>
            <TaskPage key={route.id} id={route.id} theme={theme} />
          </Suspense>
        ) : route.name === "notes" ? (
          <NotesPage view={route.view} />
        ) : route.name === "interview" ? (
          <InterviewPage key={route.qid ?? ""} qid={route.qid} />
        ) : (
          <CourseMap />
        )}
      </main>
    </div>
  );
}
