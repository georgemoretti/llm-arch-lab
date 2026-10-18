// V-wordpiece: нормализация BERT, пре-токенизация и жадный разбор WordPiece по шагам на игрушечном словаре.
import { useMemo, useState } from "react";
import { Segmented } from "./common";
import { basicTokenize, normalize, wordpiece } from "./wordpieceToy";

const PRESETS = [
  "Unaffable transformers: attention-is-all you need!",
  "Привет, мир! Модель обучения токенизатора",
  "Йогурт и ёжик — «предобучение» слов",
  "Naïve café 😀 tokenization",
];

export default function Wordpiece() {
  const [text, setText] = useState(PRESETS[0]);
  const [mode, setMode] = useState<"uncased" | "cased">("uncased");
  const [sel, setSel] = useState(0);
  const lower = mode === "uncased";
  const words = useMemo(() => basicTokenize(text, lower), [text, lower]);
  const parsed = useMemo(() => words.map((w) => wordpiece(w)), [words]);
  const total = parsed.reduce((n, p) => n + p.pieces.length, 0);
  const unk = parsed.filter((p) => p.pieces[0] === "[UNK]").length;
  const cur = parsed[Math.min(sel, parsed.length - 1)];
  const curWord = words[Math.min(sel, words.length - 1)];

  return (
    <div>
      <div className="viz-controls" style={{ flexWrap: "wrap" }}>
        <input
          className="mono"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setSel(0);
          }}
          style={{ flex: "1 1 280px", minWidth: 200, padding: "4px 8px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--panel)", color: "var(--text)" }}
        />
        <Segmented
          value={mode}
          options={[
            ["uncased", "uncased"],
            ["cased", "cased"],
          ]}
          onChange={setMode}
        />
      </div>
      <div className="viz-controls" style={{ flexWrap: "wrap" }}>
        {PRESETS.map((p) => (
          <button key={p} className="btn small" onClick={() => (setText(p), setSel(0))}>
            {p.slice(0, 22)}…
          </button>
        ))}
      </div>
      <div className="viz-subtitle">1. нормализация{lower ? " (нижний регистр, NFD без диакритики)" : " (только очистка)"}</div>
      <div className="mono" style={{ fontSize: 12.5, padding: "4px 8px", background: "var(--panel-2)", borderRadius: 6 }}>
        {normalize(text, lower) || " "}
      </div>
      <div className="viz-subtitle" style={{ marginTop: 8 }}>
        2. слова (пробелы + каждый знак пунктуации) и 3. WordPiece — кликните слово, чтобы увидеть разбор
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {parsed.map((p, i) => (
          <button
            key={i}
            onClick={() => setSel(i)}
            className="mono"
            style={{
              display: "inline-flex",
              gap: 2,
              padding: 3,
              borderRadius: 6,
              cursor: "pointer",
              background: "transparent",
              border: `1px solid ${i === sel ? "var(--accent)" : "var(--border)"}`,
            }}
            title={`слово «${words[i]}»`}
          >
            {p.pieces.map((t, j) => (
              <span
                key={j}
                style={{
                  padding: "1px 5px",
                  borderRadius: 4,
                  fontSize: 12,
                  color: t === "[UNK]" ? "var(--bad)" : "var(--text)",
                  background: t.startsWith("##")
                    ? "color-mix(in srgb, var(--series-2) 22%, transparent)"
                    : "color-mix(in srgb, var(--series-1) 22%, transparent)",
                }}
              >
                {t}
              </span>
            ))}
          </button>
        ))}
      </div>
      {cur && (
        <>
          <div className="viz-subtitle" style={{ marginTop: 10 }}>
            разбор «{curWord}»: с каждой позиции пробуем кусок от самого длинного к короткому
          </div>
          <div className="mono" style={{ fontSize: 12, display: "grid", gap: 4 }}>
            {cur.steps.map((tries, k) => (
              <div key={k} style={{ display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
                <span className="muted" style={{ width: 56 }}>
                  шаг {k + 1}
                </span>
                {tries.map((t, j) => (
                  <span
                    key={j}
                    style={{
                      padding: "0 4px",
                      borderRadius: 3,
                      textDecoration: t.found ? "none" : "line-through",
                      color: t.found ? "var(--ok)" : "var(--muted)",
                      fontWeight: t.found ? 700 : 400,
                    }}
                  >
                    {t.piece}
                  </span>
                ))}
                {!tries.some((t) => t.found) && <span style={{ color: "var(--bad)" }}>ничего не нашлось → всё слово [UNK]</span>}
              </div>
            ))}
          </div>
        </>
      )}
      <div className="kv-tiles">
        <div>
          <span className="muted">слов</span>
          <b>{words.length}</b>
        </div>
        <div>
          <span className="muted">токенов</span>
          <b>{total}</b>
          <span className="muted">{words.length ? (total / words.length).toFixed(1) : "0"} на слово</span>
        </div>
        <div>
          <span className="muted">[UNK]</span>
          <b>{unk}</b>
        </div>
      </div>
      <p className="viz-caption">
        Словарь игрушечный (~200 кусков): частые слова целиком, частые продолжения с «##» и все буквы — как запасной
        вариант. Синие куски начинают слово, оранжевые — продолжают. Слово, которое не разбирается до конца (например, с
        эмодзи), целиком становится [UNK]. В режиме uncased «й» превращается в «и», а «ё» — в «е»: NFD раскладывает их на
        букву и знак, а знаки удаляются.
      </p>
    </div>
  );
}
