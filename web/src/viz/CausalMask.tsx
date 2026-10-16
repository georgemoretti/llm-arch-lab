// V-causal-mask: как causal-маска превращает матрицу оценок в веса внимания.
import { useMemo, useState } from "react";

// Детерминированный ГПСЧ (mulberry32), чтобы картинка не прыгала при перерисовке.
function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(next: () => number) {
  const u = Math.max(next(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
}

export default function CausalMask() {
  const [T, setT] = useState(6);
  const [D, setD] = useState(16);
  const [masked, setMasked] = useState(true);
  const [scaled, setScaled] = useState(true);
  const [seed, setSeed] = useState(1);
  const [row, setRow] = useState<number | null>(null);

  const weights = useMemo(() => {
    const next = rng(seed);
    // q·k для случайных q, k ~ N(0, 1)^D имеет дисперсию D — отсюда и деление на √D.
    const scores = Array.from({ length: T }, () =>
      Array.from({ length: T }, () => {
        let s = 0;
        for (let i = 0; i < D; i++) s += gaussian(next) * gaussian(next);
        return scaled ? s / Math.sqrt(D) : s;
      }),
    );
    return scores.map((r, t) => {
      const logits = r.map((v, s) => (masked && s > t ? -Infinity : v));
      const m = Math.max(...logits);
      const e = logits.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    });
  }, [T, D, masked, scaled, seed]);

  const maxW = Math.max(...weights.flat());
  const cell = T > 8 ? 30 : 38;

  return (
    <div className="viz-causal">
      <div className="viz-controls">
        <label>
          T = {T}
          <input type="range" min={2} max={12} value={T} onChange={(e) => setT(+e.target.value)} />
        </label>
        <label>
          D = {D}
          <input type="range" min={1} max={128} value={D} onChange={(e) => setD(+e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={masked} onChange={(e) => setMasked(e.target.checked)} /> causal-маска
        </label>
        <label className="check">
          <input type="checkbox" checked={scaled} onChange={(e) => setScaled(e.target.checked)} /> делить на √D
        </label>
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые q, k
        </button>
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `28px repeat(${T}, ${cell}px) 52px` }}>
        <div />
        {Array.from({ length: T }, (_, s) => (
          <div key={s} className="viz-axis">
            k{s}
          </div>
        ))}
        <div className="viz-axis">Σ</div>
        {weights.map((r, t) => (
          <Row key={t} t={t} r={r} maxW={maxW} masked={masked} active={row === t} onHover={setRow} />
        ))}
      </div>
      <p className="viz-caption">
        Строка t — распределение внимания запроса q<sub>t</sub> по ключам. С маской ячейки правее диагонали
        равны 0, а строка всё равно суммируется в 1. Уберите деление на √D при большом D — softmax
        «насыщается»: почти весь вес уходит в одну ячейку.
        {row !== null && (
          <>
            {" "}
            Сейчас: q<sub>{row}</sub> видит {masked ? row + 1 : T} из {T} позиций, максимум веса ={" "}
            {Math.max(...weights[row]).toFixed(2)}.
          </>
        )}
      </p>
    </div>
  );
}

function Row(props: {
  t: number;
  r: number[];
  maxW: number;
  masked: boolean;
  active: boolean;
  onHover: (t: number | null) => void;
}) {
  const { t, r, maxW, masked, active, onHover } = props;
  return (
    <>
      <div className={`viz-axis ${active ? "active" : ""}`}>q{t}</div>
      {r.map((w, s) => {
        const blocked = masked && s > t;
        return (
          <div
            key={s}
            className={`viz-cell ${blocked ? "blocked" : ""} ${active ? "active" : ""}`}
            style={blocked ? undefined : { background: `color-mix(in srgb, var(--accent) ${Math.round((w / maxW) * 85)}%, transparent)` }}
            onMouseEnter={() => onHover(t)}
            onMouseLeave={() => onHover(null)}
            title={blocked ? `s=${s} > t=${t}: будущее, вес 0` : `w[${t},${s}] = ${w.toFixed(4)}`}
          >
            {blocked ? "0" : w.toFixed(2).replace(/^0/, "")}
          </div>
        );
      })}
      <div className="viz-sum">{r.reduce((a, b) => a + b, 0).toFixed(2)}</div>
    </>
  );
}
