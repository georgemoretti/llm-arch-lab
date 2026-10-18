// V-bigram-table: биграммная модель по введённому тексту — тепловая карта «символ → следующий символ», распределение
// для выбранного символа, перплексия и сэмплы. Ориентир, который MLP-модель в задаче должна обогнать.
import { useMemo, useState } from "react";
import { Bars, rng } from "./common";

const DEFAULT = `Представь, что тебе дали длинную ленту с буквами и попросили угадать, какая буква будет следующей. Сначала ты угадываешь наугад. Потом замечаешь, что после пробела часто идёт буква «п» или «с», что после «ч» почти никогда не бывает «ы», а после точки обычно стоит пробел и большая буква. Ты ещё ничего не знаешь о смысле слов, но уже угадываешь лучше случайного.`;
const TOP = 18;
const show = (c: string) => (c === " " ? "␣" : c === "\n" ? "↵" : c);

export default function BigramTable() {
  const [text, setText] = useState(DEFAULT);
  const [row, setRow] = useState(" ");
  const [seed, setSeed] = useState(1);

  const m = useMemo(() => {
    const chars = [...new Set(text)].sort();
    const idx = new Map(chars.map((c, i) => [c, i]));
    const V = chars.length;
    const counts = Array.from({ length: V }, () => new Array(V).fill(0));
    const freq = new Array(V).fill(0);
    for (let i = 0; i + 1 < text.length; i++) {
      counts[idx.get(text[i])!][idx.get(text[i + 1])!]++;
      freq[idx.get(text[i])!]++;
    }
    // вероятности с +1 сглаживанием (как ориентир в тестах задачи)
    const P = counts.map((r) => {
      const s = r.reduce((a, b) => a + b, 0) + V;
      return r.map((v) => (v + 1) / s);
    });
    let nll = 0;
    for (let i = 0; i + 1 < text.length; i++) nll -= Math.log(P[idx.get(text[i])!][idx.get(text[i + 1])!]);
    const ppl = Math.exp(nll / Math.max(1, text.length - 1));
    const top = chars
      .map((c, i) => [c, freq[i]] as [string, number])
      .sort((a, b) => b[1] - a[1])
      .slice(0, TOP)
      .map(([c]) => c);
    return { chars, idx, V, counts, P, ppl, top };
  }, [text]);

  const sample = useMemo(() => {
    if (m.V < 2) return "";
    const r = rng(seed);
    let c = text[0];
    let out = c;
    for (let i = 0; i < 160; i++) {
      const p = m.P[m.idx.get(c)!];
      let u = r();
      let j = 0;
      while (j < p.length - 1 && u > p[j]) u -= p[j++];
      c = m.chars[j];
      out += c;
    }
    return out;
  }, [m, seed, text]);

  const ri = m.idx.get(row) ?? m.idx.get(m.top[0]) ?? 0;
  const rowP = m.P[ri] ?? [];
  const best = rowP
    .map((p, j) => [m.chars[j], p] as [string, number])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12);
  const maxCount = Math.max(1, ...m.top.flatMap((a) => m.top.map((b) => m.counts[m.idx.get(a)!][m.idx.get(b)!])));
  const cell = 18;

  return (
    <div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} style={{ width: "100%", font: "inherit", fontSize: 13 }} aria-label="Текст для биграммной модели" />
      <div className="mono" style={{ fontSize: 12.5, margin: "4px 0 8px" }}>
        символов в словаре V = {m.V} · перплексия биграммной модели на этом тексте = {m.ppl.toFixed(2)} (случайная модель: {m.V})
      </div>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
        <svg width={(TOP + 1) * cell + 4} height={(TOP + 1) * cell + 4} role="img" aria-label="Таблица биграмм" style={{ maxWidth: "100%" }}>
          {m.top.map((b, j) => (
            <text key={`h${j}`} x={(j + 1) * cell + cell / 2} y={12} fontSize={11} textAnchor="middle" fill="var(--muted)" className="mono">
              {show(b)}
            </text>
          ))}
          {m.top.map((a, i) => (
            <g key={a} onClick={() => setRow(a)} style={{ cursor: "pointer" }}>
              <text x={cell / 2} y={(i + 1) * cell + 13} fontSize={11} textAnchor="middle" fill={a === m.chars[ri] ? "var(--series-2)" : "var(--muted)"} className="mono">
                {show(a)}
              </text>
              {m.top.map((b, j) => {
                const v = m.counts[m.idx.get(a)!][m.idx.get(b)!];
                return (
                  <rect key={b} x={(j + 1) * cell} y={(i + 1) * cell} width={cell - 1} height={cell - 1} fill="var(--series-1)" opacity={v ? 0.12 + 0.88 * Math.sqrt(v / maxCount) : 0.04} stroke={a === m.chars[ri] ? "var(--series-2)" : "none"}>
                    <title>{`«${show(a)}» → «${show(b)}»: ${v}`}</title>
                  </rect>
                );
              })}
            </g>
          ))}
        </svg>
        <div>
          <div style={{ fontSize: 13, marginBottom: 4 }}>
            после «<b>{show(m.chars[ri] ?? "")}</b>» модель ждёт:
          </div>
          <Bars values={best.map(([, p]) => p)} labels={best.map(([c]) => show(c))} signed={false} color="var(--series-2)" />
        </div>
      </div>
      <div className="viz-controls" style={{ marginTop: 8 }}>
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новый сэмпл
        </button>
      </div>
      <div className="mono" style={{ fontSize: 12.5, whiteSpace: "pre-wrap", padding: 8, background: "var(--panel-2)", borderRadius: 6 }}>{sample}</div>
      <p className="viz-caption">
        Строки таблицы соответствуют текущему символу, столбцы — следующему (показаны 18 самых частых). Щёлкните по
        строке, чтобы увидеть распределение. Биграммная модель помнит только один символ назад, поэтому её сэмпл похож
        на текст по буквам и пробелам, но слова в нём выдуманные. MLP-модель в задаче смотрит на 4 символа и должна
        дать перплексию заметно ниже. Вставьте свой текст: чем он длиннее, тем точнее частоты.
      </p>
    </div>
  );
}
