// Monaco собирается локально (без CDN): ядро редактора + стандартные фичи + подсветка Python.
// Языковые воркеры (TS/JSON/CSS) не нужны — только базовый editor.worker.
import * as monaco from "monaco-editor/esm/vs/editor/editor.api";
import "monaco-editor/esm/vs/editor/editor.all.js";
import "monaco-editor/esm/vs/basic-languages/python/python.contribution.js";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";

(self as unknown as { MonacoEnvironment: monaco.Environment }).MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};

export { monaco };
export const monacoTheme = (theme: "light" | "dark") => (theme === "dark" ? "vs-dark" : "vs");
