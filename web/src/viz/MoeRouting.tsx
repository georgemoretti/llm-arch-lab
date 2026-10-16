// V-moe-routing: игрушечный роутер MoE прямо в браузере — тепловая карта «токены × эксперты»,
// выбранные top-k, гистограмма загрузки экспертов и две стратегии балансировки (C12–C14):
// aux loss (градиентный шаг) и bias DeepSeek-V3 (шаг ±γ, влияет только на выбор).
import { useMemo, useState } from "react";

const N = 32; // токенов
const E = 8; // экспертов
const DIM = 8; // размер скрытого состояния

type Mode = "none" | "aux" | "bias";

// Детерминированный ГПСЧ (mulberry32) + нормальное распределение.
function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(next: () => number) {
  return Math.sqrt(-2 * Math.log(Math.max(next(), 1e-9))) * Math.cos(2 * Math.PI * next());
}

function softmax(row: number[]) {
  const m = Math.max(...row);
  const e = row.map((v) => Math.exp(v - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

function topk(row: number[], k: number) {
  return row
    .map((v, i) => [v, i] as [number, number])
    .sort((a, b) => b[0] - a[0])
    .slice(0, k)
    .map(([, i]) => i);
}

export default function MoeRouting() {
  const [k, setK] = useState(2);
  const [skew, setSkew] = useState(1.5);
  const [mode, setMode] = useState<Mode>("aux");
  const [seed, setSeed] = useState(3);
  // Обучаемое смещение логитов: offset — для aux loss (через градиент), bias — для DeepSeek-V3 (правило).
  const [offset, setOffset] = useState<number[]>(Array(E).fill(0));
  const [bias, setBias] = useState<number[]>(Array(E).fill(0));
  const [steps, setSteps] = useState(0);
  const [hover, setHover] = useState<[number, number] | null>(null);

  // Базовые логиты: x·W плюс перекос в пользу первых экспертов (имитация «коллапса»).
  const base = useMemo(() => {
    const next = rng(seed);
    const W = Array.from({ length: E }, () => Array.from({ length: DIM }, () => gauss(next) / Math.sqrt(DIM)));
    return Array.from({ length: N }, () => {
      const x = Array.from({ length: DIM }, () => gauss(next));
      return W.map((w, e) => w.reduce((s, wi, i) => s + wi * x[i], 0) + skew * (1 - (2 * e) / (E - 1)));
    });
  }, [seed, skew]);

  const state = useMemo(() => computeState(base, offset, bias, k), [base, offset, bias, k]);

  function reset() {
    setOffset(Array(E).fill(0));
    setBias(Array(E).fill(0));
    setSteps(0);
  }

  function step(times: number) {
    let off = offset;
    let b = bias;
    for (let s = 0; s < times; s++) {
      const st = computeState(base, off, b, k);
      if (mode === "aux") {
        // dL/da_j = E · Σ_i f_i · mean_n p_ni (δ_ij − p_nj): градиент aux loss по смещению логитов эксперта j.
        const grad = Array(E).fill(0);
        for (const p of st.probs)
          for (let j = 0; j < E; j++)
            for (let i = 0; i < E; i++) grad[j] += (E * st.load[i] * p[i] * ((i === j ? 1 : 0) - p[j])) / N;
        off = off.map((v, j) => v - 2.0 * grad[j]);
      } else if (mode === "bias") {
        const mean = 1 / E;
        b = b.map((v, j) => v - 0.05 * Math.sign(st.load[j] - mean));
      }
    }
    setOffset(off);
    setBias(b);
    if (mode !== "none") setSteps((s) => s + times);
  }

  const maxLoad = Math.max(...state.load);
  const minLoad = Math.min(...state.load);
  const barMax = Math.max(maxLoad, 2 / E);

  return (
    <div className="viz-moe">
      <div className="viz-controls">
        <label>
          top-k = {k}
          <input type="range" min={1} max={4} value={k} onChange={(e) => setK(+e.target.value)} />
        </label>
        <label>
          перекос = {skew.toFixed(1)}
          <input type="range" min={0} max={3} step={0.1} value={skew} onChange={(e) => setSkew(+e.target.value)} />
        </label>
        <label className="check">
          балансировка
          <select
            value={mode}
            onChange={(e) => {
              setMode(e.target.value as Mode);
              reset();
            }}
          >
            <option value="none">нет</option>
            <option value="aux">aux loss (Switch, Mixtral)</option>
            <option value="bias">bias без градиента (DeepSeek-V3)</option>
          </select>
        </label>
        <button className="btn small" disabled={mode === "none"} onClick={() => step(1)}>
          шаг
        </button>
        <button className="btn small" disabled={mode === "none"} onClick={() => step(20)}>
          ×20
        </button>
        <button className="btn small" onClick={reset}>
          сброс
        </button>
        <button
          className="btn small"
          onClick={() => {
            setSeed((s) => s + 1);
            reset();
          }}
        >
          новые токены
        </button>
      </div>

      <div className="kv-tiles">
        <div>
          <span className="muted">aux loss (идеал 1, коллапс {E})</span>
          <b>{state.aux.toFixed(2)}</b>
        </div>
        <div>
          <span className="muted">макс. доля эксперта</span>
          <b>{(maxLoad * 100).toFixed(0)}%</b>
        </div>
        <div>
          <span className="muted">мин. доля эксперта</span>
          <b>{(minLoad * 100).toFixed(0)}%</b>
        </div>
        <div>
          <span className="muted">шагов балансировки</span>
          <b>{steps}</b>
        </div>
      </div>

      <div className="moe-main">
        <div className="moe-heat" style={{ gridTemplateColumns: `34px repeat(${E}, 26px)` }} onMouseLeave={() => setHover(null)}>
          <div />
          {Array.from({ length: E }, (_, e) => (
            <div key={e} className="viz-axis">
              E{e}
            </div>
          ))}
          {state.probs.map((p, n) => (
            <HeatRow key={n} n={n} p={p} chosen={state.chosen[n]} onHover={(e) => setHover([n, e])} />
          ))}
        </div>

        <div className="moe-load">
          <div className="moe-load-title muted">доля назначений на эксперта</div>
          <svg viewBox="0 0 260 180" role="img" aria-label="Загрузка экспертов">
            {Array.from({ length: E }, (_, e) => {
              const h = (state.load[e] / barMax) * 140;
              return (
                <g key={e}>
                  <rect className="moe-bar" x={10 + e * 31} y={150 - h} width={27} height={Math.max(h, 0.5)} rx={3}>
                    <title>{`E${e}: ${(state.load[e] * 100).toFixed(1)}% назначений`}</title>
                  </rect>
                  <text className="moe-bar-label" x={10 + e * 31 + 13.5} y={166} textAnchor="middle">
                    E{e}
                  </text>
                </g>
              );
            })}
            <line className="moe-ideal" x1={6} x2={256} y1={150 - (1 / E / barMax) * 140} y2={150 - (1 / E / barMax) * 140} />
            <text className="moe-bar-label" x={256} y={150 - (1 / E / barMax) * 140 - 4} textAnchor="end">
              идеал 1/E
            </text>
            <line className="moe-axis" x1={6} x2={256} y1={150} y2={150} />
          </svg>
        </div>
      </div>

      <div className="mask-legend">
        <span>
          <i className="mask-swatch allowed" /> вероятность роутера (темнее — больше)
        </span>
        <span>
          <i className="mask-swatch moe-chosen-swatch" /> эксперт выбран (top-{k})
        </span>
      </div>
      <p className="viz-caption">
        {hover ? (
          <>
            Токен {hover[0]}, эксперт E{hover[1]}: вероятность {(state.probs[hover[0]][hover[1]] * 100).toFixed(1)}%,{" "}
            {state.chosen[hover[0]].includes(hover[1]) ? "выбран" : "не выбран"}
            {mode === "bias" ? `; bias эксперта ${bias[hover[1]].toFixed(2)}` : ""}.{" "}
          </>
        ) : null}
        Строка — токен, столбец — эксперт; цвет — softmax-вероятность роутера, рамка — выбранные top-k. Ползунок
        «перекос» имитирует коллапс: первые эксперты получают почти все токены. <b>aux loss</b> — градиентный шаг по
        смещению логитов: вероятности перегруженных экспертов падают, и тепловая карта меняется. <b>bias</b> — шаг ±0.05
        по знаку перекоса; он меняет только выбор (рамки), а вероятности — веса выходов — остаются прежними. Для
        наглядности обучается только смещение логитов каждого эксперта, а не вся матрица роутера.
      </p>
    </div>
  );
}

function computeState(base: number[][], offset: number[], bias: number[], k: number) {
  const probs = base.map((row) => softmax(row.map((v, e) => v + offset[e])));
  const chosen = probs.map((p) => topk(p.map((v, e) => Math.log(v) + bias[e]), k));
  const counts = Array(E).fill(0);
  for (const c of chosen) for (const e of c) counts[e]++;
  const load = counts.map((c) => c / (N * k));
  const P = Array.from({ length: E }, (_, e) => probs.reduce((s, p) => s + p[e], 0) / N);
  const aux = E * load.reduce((s, f, e) => s + f * P[e], 0);
  return { probs, chosen, load, aux };
}

function HeatRow({ n, p, chosen, onHover }: { n: number; p: number[]; chosen: number[]; onHover: (e: number) => void }) {
  return (
    <>
      <div className="viz-axis">{n % 4 === 0 ? `t${n}` : ""}</div>
      {p.map((v, e) => (
        <div
          key={e}
          className={`moe-cell ${chosen.includes(e) ? "chosen" : ""}`}
          style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(Math.sqrt(v) * 90)}%, var(--panel))` }}
          onMouseEnter={() => onHover(e)}
        />
      ))}
    </>
  );
}
