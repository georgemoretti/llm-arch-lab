// V-rope: как RoPE поворачивает пары координат и почему score зависит только от m − n.
// Раскладка «половинами», как в C03: пара i = (x_i, x_{i + D/2}), частота θ^(−2i/D).
import { useMemo, useState } from "react";
import { LineChart } from "./LineChart";

function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randVec(d: number, seed: number) {
  const next = rng(seed);
  return Array.from({ length: d }, () => {
    const u = Math.max(next(), 1e-9);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  });
}

const freqs = (d: number, theta: number) => Array.from({ length: d / 2 }, (_, i) => theta ** ((-2 * i) / d));

function rope(x: number[], m: number, f: number[]) {
  const h = x.length / 2;
  const out = x.slice();
  for (let i = 0; i < h; i++) {
    const c = Math.cos(m * f[i]);
    const s = Math.sin(m * f[i]);
    out[i] = x[i] * c - x[i + h] * s;
    out[i + h] = x[i + h] * c + x[i] * s;
  }
  return out;
}

const dot = (a: number[], b: number[]) => a.reduce((acc, v, i) => acc + v * b[i], 0);

const DELTAS = Array.from({ length: 129 }, (_, i) => i * 2); // 0 … 256

export default function RopeRotation() {
  const [D, setD] = useState(32);
  const [theta, setTheta] = useState(10000);
  const [m, setM] = useState(5);
  const [n, setN] = useState(2);
  const [pair, setPair] = useState(0);
  const [shift, setShift] = useState(0);
  const [seed, setSeed] = useState(1);

  const f = useMemo(() => freqs(D, theta), [D, theta]);
  const q = useMemo(() => randVec(D, seed), [D, seed]);
  const k = useMemo(() => randVec(D, seed + 1000), [D, seed]);
  const i = Math.min(pair, D / 2 - 1);
  const h = D / 2;

  const score = dot(rope(q, m + shift, f), rope(k, n + shift, f));
  const score0 = dot(rope(q, m, f), rope(k, n, f));
  // score от расстояния: q и k с одинаковым содержанием (k = q), позиции m и m + Δ
  const curve = useMemo(() => {
    const norm = dot(q, q);
    return DELTAS.map((dlt) => dot(q, rope(q, dlt, f)) / norm);
  }, [q, f]);

  // пара i: исходные и повёрнутые 2D-векторы
  const qa = [q[i], q[i + h]];
  const ka = [k[i], k[i + h]];
  const rot = (v: number[], ang: number) => [v[0] * Math.cos(ang) - v[1] * Math.sin(ang), v[0] * Math.sin(ang) + v[1] * Math.cos(ang)];
  const qr = rot(qa, (m + shift) * f[i]);
  const kr = rot(ka, (n + shift) * f[i]);
  const R = 70;
  const scale = R / Math.max(Math.hypot(...qa), Math.hypot(...ka), 1e-6);
  const P = (v: number[]) => [90 + v[0] * scale, 90 - v[1] * scale]; // y вверх

  return (
    <div className="viz-rope">
      <div className="viz-controls">
        <label>
          D = {D}
          <input type="range" min={8} max={64} step={8} value={D} onChange={(e) => setD(+e.target.value)} />
        </label>
        <label className="check">
          θ
          <select value={theta} onChange={(e) => setTheta(+e.target.value)}>
            <option value={10000}>10 000 (Llama 2)</option>
            <option value={500000}>500 000 (Llama 3)</option>
          </select>
        </label>
        <label>
          m (q) = {m}
          <input type="range" min={0} max={64} value={m} onChange={(e) => setM(+e.target.value)} />
        </label>
        <label>
          n (k) = {n}
          <input type="range" min={0} max={64} value={n} onChange={(e) => setN(+e.target.value)} />
        </label>
        <label>
          сдвиг обеих = {shift}
          <input type="range" min={0} max={200} value={shift} onChange={(e) => setShift(+e.target.value)} />
        </label>
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые q, k
        </button>
      </div>

      <div className="rope-dials" title="Каждый круг — одна пара координат q; стрелка — угол поворота на позиции m">
        {f.map((fi, j) => {
          const a = (m + shift) * fi;
          return (
            <button key={j} className={`rope-dial ${j === i ? "active" : ""}`} onClick={() => setPair(j)} title={`пара ${j}: θ_${j} = ${fi.toPrecision(3)} рад/позиция, угол ${(a % (2 * Math.PI)).toFixed(2)} рад`}>
              <svg viewBox="-12 -12 24 24">
                <circle r={10} className="rope-dial-ring" />
                <line x1={0} y1={0} x2={9 * Math.cos(a)} y2={-9 * Math.sin(a)} className="rope-dial-hand" />
              </svg>
              <span>{j}</span>
            </button>
          );
        })}
      </div>

      <div className="rope-main">
        <figure className="rope-plane">
          <svg viewBox="0 0 180 180" width={180} height={180} role="img" aria-label={`Пара ${i}: повороты q и k`}>
            <circle cx={90} cy={90} r={R} className="rope-ring" />
            <line x1={10} x2={170} y1={90} y2={90} className="rope-axis" />
            <line x1={90} x2={90} y1={10} y2={170} className="rope-axis" />
            {[
              [qa, "var(--series-1)", true],
              [ka, "var(--series-2)", true],
              [qr, "var(--series-1)", false],
              [kr, "var(--series-2)", false],
            ].map(([v, c, ghost], j) => {
              const [x, y] = P(v as number[]);
              return <line key={j} x1={90} y1={90} x2={x} y2={y} stroke={c as string} strokeWidth={ghost ? 1.5 : 2.5} strokeDasharray={ghost ? "3 3" : undefined} opacity={ghost ? 0.6 : 1} />;
            })}
            <circle cx={P(qr)[0]} cy={P(qr)[1]} r={4} fill="var(--series-1)" className="lc-dot" />
            <circle cx={P(kr)[0]} cy={P(kr)[1]} r={4} fill="var(--series-2)" className="lc-dot" />
          </svg>
          <figcaption>
            <div className="lc-legend">
              <span>
                <i style={{ background: "var(--series-1)" }} />q, пара {i}
              </span>
              <span>
                <i style={{ background: "var(--series-2)" }} />k, пара {i}
              </span>
            </div>
            <div className="muted">
              пунктир — до поворота; q повёрнут на {((m + shift) * f[i]).toFixed(2)} рад, k — на{" "}
              {((n + shift) * f[i]).toFixed(2)} рад; угол между ними меняется на (n − m)·θ<sub>{i}</sub> ={" "}
              {((n - m) * f[i]).toFixed(2)} рад
            </div>
          </figcaption>
        </figure>

        <div className="rope-score">
          <div className="kv-tiles">
            <div>
              <span className="muted">⟨q_m, k_n⟩ при сдвиге {shift}</span>
              <b>{score.toFixed(4)}</b>
            </div>
            <div>
              <span className="muted">без сдвига</span>
              <b>{score0.toFixed(4)}</b>
            </div>
            <div>
              <span className="muted">|q| до / после</span>
              <b>
                {Math.sqrt(dot(q, q)).toFixed(3)} / {Math.sqrt(dot(rope(q, m + shift, f), rope(q, m + shift, f))).toFixed(3)}
              </b>
            </div>
          </div>
        </div>
      </div>

      <div className="lc-title">Сходство q с самим собой на расстоянии Δ: ⟨R_m q, R_(m+Δ) q⟩ / |q|²</div>
      <LineChart
        xs={DELTAS}
        series={[{ id: "s", label: "score", color: "var(--series-1)", ys: curve }]}
        xTicks={[0, 64, 128, 192, 256]}
        xFormat={(x) => String(x)}
        yFormat={(y) => String(+y.toFixed(2))}
        yLabel="сходство"
        xLabel="расстояние Δ"
        markerX={Math.min(Math.abs(m - n), 256)}
        height={190}
        zeroLine
      />

      <p className="viz-caption">
        Сдвиньте обе позиции ползунком «сдвиг» — score не меняется: важна только разность m − n. Круги сверху — все
        D/2 пар: первые вращаются быстро (1 рад на позицию), последние почти стоят; с θ = 500 000 медленных пар
        больше. График — сходство вектора с самим собой на расстоянии Δ: в среднем оно убывает с расстоянием
        (long-term decay из статьи RoFormer), но не монотонно. Нажмите на круг, чтобы увидеть пару крупно.
      </p>
    </div>
  );
}
