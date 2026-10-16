// Обёртка над Monaco без @monaco-editor/react (тот по умолчанию грузит Monaco с CDN).
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { monaco, monacoTheme } from "../monaco";
import type { Theme } from "../theme";

export interface ErrorMark {
  line: number;
  message: string;
}

export interface CodeEditorHandle {
  getValue(): string;
  /** Заменить текст, сохранив историю undo (для правок, пришедших с диска). */
  replaceValue(text: string): void;
  revealLine(line: number): void;
  focus(): void;
}

interface Props {
  initialValue: string;
  theme: Theme;
  errors: ErrorMark[];
  onChange(text: string): void;
  onRun(): void;
  onSave(): void;
}

export const CodeEditor = forwardRef<CodeEditorHandle, Props>(function CodeEditor(props, ref) {
  const host = useRef<HTMLDivElement>(null);
  const editor = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const decorations = useRef<monaco.editor.IEditorDecorationsCollection | null>(null);
  const suppress = useRef(false);
  // Колбэки меняются на каждом рендере; команды Monaco регистрируются один раз — читаем через ref.
  const cb = useRef(props);
  cb.current = props;

  useEffect(() => {
    const ed = monaco.editor.create(host.current!, {
      value: props.initialValue,
      language: "python",
      theme: monacoTheme(props.theme),
      automaticLayout: true,
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace",
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      tabSize: 4,
      insertSpaces: true,
      rulers: [100],
      renderWhitespace: "selection",
      glyphMargin: true,
      fixedOverflowWidgets: true,
      // Кириллица в комментариях и docstring — норма для курса, не «подозрительные» символы.
      unicodeHighlight: { ambiguousCharacters: false, invisibleCharacters: true },
    });
    editor.current = ed;
    decorations.current = ed.createDecorationsCollection();
    // ⌘+Enter на macOS и Ctrl+Enter везде (WinCtrl — физический Ctrl и на Mac).
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => cb.current.onRun());
    ed.addCommand(monaco.KeyMod.WinCtrl | monaco.KeyCode.Enter, () => cb.current.onRun());
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => cb.current.onSave());
    const sub = ed.onDidChangeModelContent(() => {
      if (!suppress.current) cb.current.onChange(ed.getValue());
    });
    return () => {
      sub.dispose();
      ed.getModel()?.dispose();
      ed.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => monaco.editor.setTheme(monacoTheme(props.theme)), [props.theme]);

  useEffect(() => {
    decorations.current?.set(
      props.errors.map((e) => ({
        range: new monaco.Range(e.line, 1, e.line, 1),
        options: {
          isWholeLine: true,
          className: "err-line",
          glyphMarginClassName: "err-glyph",
          hoverMessage: { value: e.message },
          glyphMarginHoverMessage: { value: e.message },
        },
      })),
    );
  }, [props.errors]);

  useImperativeHandle(ref, () => ({
    getValue: () => editor.current?.getValue() ?? "",
    replaceValue(text: string) {
      const ed = editor.current;
      const model = ed?.getModel();
      if (!ed || !model || model.getValue() === text) return;
      suppress.current = true;
      ed.pushUndoStop();
      ed.executeEdits("disk", [{ range: model.getFullModelRange(), text }]);
      ed.pushUndoStop();
      suppress.current = false;
    },
    revealLine(line: number) {
      const ed = editor.current;
      if (!ed) return;
      ed.revealLineInCenter(line);
      ed.setPosition({ lineNumber: line, column: 1 });
      ed.focus();
    },
    focus: () => editor.current?.focus(),
  }));

  return <div className="editor-host" ref={host} />;
});

export function DiffView({ mine, reference, theme }: { mine: string; reference: string; theme: Theme }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const diff = monaco.editor.createDiffEditor(host.current!, {
      theme: monacoTheme(theme),
      readOnly: true,
      originalEditable: false,
      automaticLayout: true,
      renderSideBySide: (host.current?.clientWidth ?? 1000) > 760,
      minimap: { enabled: false },
      fontSize: 13,
      scrollBeyondLastLine: false,
    });
    const original = monaco.editor.createModel(mine, "python");
    const modified = monaco.editor.createModel(reference, "python");
    diff.setModel({ original, modified });
    return () => {
      diff.dispose();
      original.dispose();
      modified.dispose();
    };
  }, [mine, reference, theme]);
  return <div className="editor-host" ref={host} />;
}
