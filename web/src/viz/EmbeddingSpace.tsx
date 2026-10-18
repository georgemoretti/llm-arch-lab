// V-embedding-space: предзаписанный прогон проекта B20 (bi-encoder на ModernBERT, обучение InfoNCE на парах
// «фрагмент ↔ абзац» корпуса курса): проекция PCA эмбеддингов 200 абзацев до и после обучения, распределения
// косинусов «своих» и случайных пар (анизотропия), кривые recall и сравнение с TF-IDF и bge-m3.
// Данные: content/viz/embedding-space.data.json (solution.py --record).
import { useState } from "react";
import run from "../../../content/viz/embedding-space.data.json";
import { LineChart } from "./LineChart";
import { Segmented } from "./common";

interface Side {
  xy: number[][];
  pos: number[];
  rand: number[];
}

interface Run {
  probe: { task: string; block: string; text: string }[];
  before: Side;
  after: Side;
  curve: { step: number; loss: number | null; r_frag: number; r_q: number; aniso: number }[];
  results: Record<string, Record<string, Record<string, number>>>;
}

const R = run as Run;
const BLOCK_COLOR: Record<string, string> = { A: "var(--series-1)", B: "var(--series-2)", C: "var(--series-3)" };
const BLOCK_NAME: Record<string, string> = { A: "блок A: основы", B: "блок B: энкодеры", C: "блок C: декодеры" };
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

function histogram(values: number[], lo: number, hi: number, bins: number) {
  const h = new Array(bins).fill(0);
  for (const v of values) h[Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)))]++;
  return h.map((c) => c / values.length);
}

function Histogram({ side }: { side: Side }) {
  const lo = -0.4;
  const hi = 1;
  const bins = 28;
  const hp = histogram(side.pos, lo, hi, bins);
  const hr = histogram(side.rand, lo, hi, bins);
  const W = 440;
  const H = 120;
  const top = Math.max(...hp, ...hr);
  const bw = (W - 20) / bins;
  const x = (v: number) => 10 + ((v - lo) / (hi - lo)) * (W - 20);
  const bars = (h: number[], color: string, dx: number) =>
    h.map((f, i) => (
      <rect key={i} x={10 + i * bw + dx} width={bw / 2 - 1} y={H - 18 - (f / top) * (H - 30)} height={(f / top) * (H - 30)} fill={color} opacity={0.85} />
    ));
  const ticks = [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} role="img" aria-label="распределения косинусов">
      {bars(hr, "var(--muted)", 0)}
      {bars(hp, "var(--accent)", bw / 2)}
      {[
        [mean(side.rand), "var(--muted)"],
        [mean(side.pos), "var(--accent)"],
      ].map(([m, c], i) => (
        <line key={i} x1={x(m as number)} x2={x(m as number)} y1={8} y2={H - 18} stroke={c as string} strokeDasharray="3 3" />
      ))}
      <line x1={10} x2={W - 10} y1={H - 18} y2={H - 18} stroke="var(--border)" />
      {ticks.map((t) => (
        <text key={t} x={x(t)} y={H - 4} fontSize={10} textAnchor="middle" fill="var(--muted)">
          {String(t).replace("-", "−")}
        </text>
      ))}
    </svg>
  );
}

export default function EmbeddingSpace() {
  const [phase, setPhase] = useState<"before" | "after">("after");
  const [hover, setHover] = useState<number | null>(null);
  const side = R[phase];
  const steps = R.curve.map((c) => c.step);
  const S = 300;
  const px = (v: number) => S / 2 + v * (S / 2 - 12);
  const blocks = ["A", "B", "C"];
  const rows = Object.entries(R.results);
  const first = R.curve[0];
  const last = R.curve[R.curve.length - 1];

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={phase}
          options={[
            ["before", "до обучения (случайные веса)"],
            ["after", `после ${last.step} шагов InfoNCE`],
          ]}
          onChange={setPhase}
        />
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-start" }}>
        <div>
          <div className="viz-subtitle">PCA эмбеддингов 200 абзацев курса (цвет — блок). Наведи на точку.</div>
          <svg viewBox={`0 0 ${S} ${S}`} width={S} height={S} style={{ border: "1px solid var(--border)", borderRadius: 6 }}>
            {side.xy.map(([a, b], i) => (
              <circle
                key={i}
                cx={px(a)}
                cy={px(-b)}
                r={hover === i ? 6 : 3.5}
                fill={BLOCK_COLOR[R.probe[i].block]}
                opacity={hover === null || hover === i ? 0.85 : 0.35}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              />
            ))}
          </svg>
          <div className="mask-legend">
            {blocks.map((b) => (
              <span key={b}>
                <i className="mask-swatch" style={{ background: BLOCK_COLOR[b] }} /> {BLOCK_NAME[b]}
              </span>
            ))}
          </div>
          <div className="viz-caption" style={{ maxWidth: S, minHeight: 48 }}>
            {hover !== null ? (
              <>
                <b>{R.probe[hover].task}</b>: {R.probe[hover].text}…
              </>
            ) : (
              "Сравни картинки до и после: метки блоков модель не видела, а абзацы о похожем начинают держаться вместе."
            )}
          </div>
        </div>
        <div style={{ flex: "1 1 320px", minWidth: 0 }}>
          <div className="viz-subtitle">
            Косинусы: <span style={{ color: "var(--accent)" }}>■</span> фрагмент ↔ свой абзац,{" "}
            <span className="muted">■</span> два случайных абзаца; пунктир — среднее.
          </div>
          <Histogram side={side} />
          <div className="kv-tiles">
            <div>
              <span className="muted">средний cos случайных пар</span>
              <b>{mean(side.rand).toFixed(2)}</b>
              <span className="muted">анизотропия: чем ближе к 0, тем лучше</span>
            </div>
            <div>
              <span className="muted">средний cos «своих» пар</span>
              <b>{mean(side.pos).toFixed(2)}</b>
              <span className="muted">зазор {(mean(side.pos) - mean(side.rand)).toFixed(2)}</span>
            </div>
            <div>
              <span className="muted">recall@1 по фрагментам</span>
              <b>{(phase === "before" ? first.r_frag : last.r_frag).toFixed(2)}</b>
              <span className="muted">300 фрагментов по 8 слов</span>
            </div>
          </div>
        </div>
      </div>
      <div className="viz-subtitle" style={{ marginTop: 12 }}>Во время обучения: recall и средний косинус случайных пар.</div>
      <LineChart
        xs={steps}
        series={[
          { id: "f", label: "фрагменты r@1", color: "var(--series-1)", ys: R.curve.map((c) => c.r_frag) },
          { id: "q", label: "вопросы r@5", color: "var(--series-2)", ys: R.curve.map((c) => c.r_q) },
          { id: "a", label: "cos случайных", color: "var(--series-3)", ys: R.curve.map((c) => c.aniso) },
        ]}
        xTicks={steps.filter((_, i) => i % 2 === 0)}
        xFormat={(v) => String(Math.round(v))}
        yFormat={(v) => v.toFixed(2)}
        yLabel="доля / косинус"
        xLabel="шаг обучения"
        height={190}
      />
      <table className="budget-table">
        <thead>
          <tr>
            <td className="muted">метод</td>
            <td className="muted">фрагменты r@1</td>
            <td className="muted">вопросы r@1</td>
            <td className="muted">r@5</td>
            <td className="muted">r@10</td>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, r]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="mono">{r.fragments["1"].toFixed(2)}</td>
              <td className="mono">{r.questions["1"].toFixed(2)}</td>
              <td className="mono">{r.questions["5"].toFixed(2)}</td>
              <td className="mono">{r.questions["10"].toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="viz-caption">
        До обучения эмбеддинги случайного ModernBERT уже немного «видят» общие слова (среднее случайных векторов токенов),
        но все тексты похожи на все: средний косинус случайных пар {mean(R.before.rand).toFixed(2)}. InfoNCE разводит
        чужие пары к нулю ({mean(R.after.rand).toFixed(2)}) и стягивает свои ({mean(R.before.pos).toFixed(2)} →{" "}
        {mean(R.after.pos).toFixed(2)}). На вопросах курса наша модель (0.5 М параметров, 1.5 минуты обучения) уступает
        TF-IDF, а предобученная bge-m3 — лучшая; гибрид с мешком слов помогает обеим моделям.
      </p>
    </div>
  );
}
