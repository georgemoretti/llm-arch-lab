import { useState } from "react";
import type { RunResult, TestEntry } from "../api";

const ICON: Record<TestEntry["outcome"], string> = {
  passed: "✓",
  failed: "✗",
  error: "!",
  skipped: "–",
  "not-run": "·",
};

const STATUS_TEXT: Record<RunResult["status"], string> = {
  passed: "Все тесты прошли",
  failed: "Есть упавшие тесты",
  error: "Тесты не запустились",
  timeout: "Превышен таймаут",
};

export function TestResults(props: {
  result: RunResult | null;
  running: boolean;
  onReveal(line: number): void;
}) {
  const { result, running, onReveal } = props;
  const [showOutput, setShowOutput] = useState(false);

  if (!result) {
    return (
      <div className="results empty">
        {running ? <span className="spinner" /> : null}
        {running ? "Запускаю тесты…" : "Нажми Run (Ctrl/⌘ + Enter), чтобы прогнать тесты."}
      </div>
    );
  }

  return (
    <div className={`results ${running ? "stale" : ""}`}>
      <div className={`results-head ${result.status}`}>
        {running && <span className="spinner" />}
        <strong>{STATUS_TEXT[result.status]}</strong>
        <span className="mono">
          {result.passed}/{result.total}
        </span>
        <span className="muted mono">{result.duration_s.toFixed(1)} с</span>
        <span className="muted results-time">{new Date(result.created_at).toLocaleTimeString()}</span>
      </div>
      {result.message && <pre className="results-msg">{result.message}</pre>}
      <ul className="tests">
        {result.tests.map((t) => (
          <li key={t.nodeid} className={`test ${t.outcome}`}>
            <div className="test-line">
              <span className="test-icon">{ICON[t.outcome]}</span>
              <span className="test-title">{t.title}</span>
              <span className="test-name mono">{t.name}</span>
            </div>
            {(t.outcome === "failed" || t.outcome === "error") && t.message && (
              <div className="test-detail">
                <pre>{t.message}</pre>
                {t.location && (
                  <button className="loc" onClick={() => onReveal(t.location!.line)} title="Перейти к строке">
                    {t.location.file}:{t.location.line} · {t.location.function} ·{" "}
                    <code>{t.location.source}</code>
                  </button>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
      {result.output_tail && (
        <div className="output">
          <button className="link-btn" onClick={() => setShowOutput(!showOutput)}>
            {showOutput ? "▾" : "▸"} Вывод pytest
          </button>
          {showOutput && <pre>{result.output_tail}</pre>}
        </div>
      )}
    </div>
  );
}
