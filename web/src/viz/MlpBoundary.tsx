// V-mlp-boundary: MLP 2 → H → 1 обучается прямо в браузере (полный батч, MSE, SGD) — как train() в задаче.
// Фон — выход сети на сетке, точки — данные. Без tanh сеть линейна и кольца не разделяет.
import { useEffect, useMemo, useRef, useState } from "react";
import { gaussian, rng, Segmented, Slider } from "./common";

type DataId = "circles" | "xor" | "spiral";
type Pt = { x: number; y: number; t: number };

function makeData(id: DataId): Pt[] {
  const r = rng(11);
  const pts: Pt[] = [];
  for (let i = 0; i < 60; i++) {
    const t = i % 2 === 0 ? 1 : -1;
    if (id === "circles") {
      const a = 2 * Math.PI * r();
      const rad = (t > 0 ? 0.35 : 0.85) + 0.07 * gaussian(r);
      pts.push({ x: rad * Math.cos(a), y: rad * Math.sin(a), t });
    } else if (id === "xor") {
      const sx = r() < 0.5 ? -1 : 1;
      const sy = t > 0 ? -sx : sx;
      pts.push({ x: sx * (0.2 + 0.6 * r()), y: sy * (0.2 + 0.6 * r()), t });
    } else {
      const k = i / 60;
      const a = 3 * Math.PI * k + (t > 0 ? 0 : Math.PI);
      const rad = 0.1 + 0.8 * k;
      pts.push({ x: rad * Math.cos(a) + 0.03 * gaussian(r), y: rad * Math.sin(a) + 0.03 * gaussian(r), t });
    }
  }
  return pts;
}

type Net = { W1: number[][]; b1: number[]; W2: number[]; b2: number };

function initNet(h: number, seed: number): Net {
  const r = rng(seed);
  return {
    W1: Array.from({ length: h }, () => [2 * r() - 1, 2 * r() - 1]),
    b1: Array(h).fill(0),
    W2: Array.from({ length: h }, () => 2 * r() - 1),
    b2: 0,
  };
}

function forward(net: Net, x: number, y: number, nonlin: boolean) {
  const z = net.W1.map((w, j) => w[0] * x + w[1] * y + net.b1[j]);
  const a = nonlin ? z.map(Math.tanh) : z;
  return { a, out: a.reduce((s, v, j) => s + v * net.W2[j], net.b2) };
}

/** Один шаг полного батча: MSE, ручной backprop (то, что Value делает автоматически), SGD. */
function step(net: Net, data: Pt[], lr: number, nonlin: boolean): number {
  const H = net.b1.length;
  const gW1 = net.W1.map(() => [0, 0]);
  const gb1 = Array(H).fill(0);
  const gW2 = Array(H).fill(0);
  let gb2 = 0;
  let loss = 0;
  for (const p of data) {
    const { a, out } = forward(net, p.x, p.y, nonlin);
    const d = out - p.t;
    loss += d * d;
    const dout = (2 * d) / data.length;
    gb2 += dout;
    for (let j = 0; j < H; j++) {
      gW2[j] += dout * a[j];
      const dz = dout * net.W2[j] * (nonlin ? 1 - a[j] * a[j] : 1);
      gW1[j][0] += dz * p.x;
      gW1[j][1] += dz * p.y;
      gb1[j] += dz;
    }
  }
  for (let j = 0; j < H; j++) {
    net.W1[j][0] -= lr * gW1[j][0];
    net.W1[j][1] -= lr * gW1[j][1];
    net.b1[j] -= lr * gb1[j];
    net.W2[j] -= lr * gW2[j];
  }
  net.b2 -= lr * gb2;
  return loss / data.length;
}

const S = 300;
const GRID = 30;
const toPx = (v: number) => ((v + 1.1) / 2.2) * S;

export default function MlpBoundary() {
  const [dataId, setDataId] = useState<DataId>("circles");
  const [hidden, setHidden] = useState(8);
  const [nonlin, setNonlin] = useState(true);
  const [lr, setLr] = useState(0.1);
  const [seed, setSeed] = useState(1);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const data = useMemo(() => makeData(dataId), [dataId]);
  const net = useRef<Net>(initNet(hidden, seed));
  const losses = useRef<number[]>([]);

  const reset = () => {
    net.current = initNet(hidden, seed);
    losses.current = [];
    setTick((t) => t + 1);
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reset, [dataId, hidden, nonlin, seed]);

  const advance = (n: number) => {
    for (let i = 0; i < n; i++) losses.current.push(step(net.current, data, lr, nonlin));
    setTick((t) => t + 1);
  };

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => advance(5), 40);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, lr, data, nonlin]);

  const cells = [];
  for (let i = 0; i < GRID; i++)
    for (let j = 0; j < GRID; j++) {
      const x = -1.1 + (2.2 * (i + 0.5)) / GRID;
      const y = -1.1 + (2.2 * (j + 0.5)) / GRID;
      const v = Math.max(-1, Math.min(1, forward(net.current, x, y, nonlin).out));
      cells.push(
        <rect
          key={`${i}-${j}`}
          x={(i * S) / GRID}
          y={S - ((j + 1) * S) / GRID}
          width={S / GRID + 0.5}
          height={S / GRID + 0.5}
          fill={v > 0 ? "var(--series-1)" : "var(--series-2)"}
          opacity={0.08 + 0.4 * Math.abs(v)}
        />,
      );
    }
  const acc = data.filter((p) => forward(net.current, p.x, p.y, nonlin).out > 0 === p.t > 0).length / data.length;
  const L = losses.current;
  const maxL = Math.max(...L.slice(0, 5), 1e-9);
  const path = L.map((v, i) => `${i ? "L" : "M"}${(i / Math.max(L.length - 1, 1)) * 260},${80 - (Math.min(v, maxL) / maxL) * 76}`).join(" ");

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={dataId}
          options={[
            ["circles", "кольца"],
            ["xor", "XOR"],
            ["spiral", "спирали"],
          ]}
          onChange={setDataId}
        />
        <Segmented
          value={String(hidden) as "2" | "4" | "8" | "16"}
          options={[
            ["2", "H = 2"],
            ["4", "4"],
            ["8", "8"],
            ["16", "16"],
          ]}
          onChange={(v) => setHidden(+v)}
        />
        <label>
          <input type="checkbox" checked={nonlin} onChange={(e) => setNonlin(e.target.checked)} /> tanh в скрытом слое
        </label>
        <Slider label={<>lr = {lr.toFixed(2)}</>} value={lr} min={0.01} max={1} step={0.01} onChange={setLr} />
      </div>
      <div className="viz-controls">
        <button className="btn small" onClick={() => setRunning(!running)}>
          {running ? "⏸ пауза" : "▶ обучать"}
        </button>
        <button className="btn small" onClick={() => advance(1)} disabled={running}>
          +1 шаг
        </button>
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новая инициализация
        </button>
        <span className="muted mono">
          шаг {L.length} · loss {L.length ? L[L.length - 1].toFixed(4) : "—"} · точность {(acc * 100).toFixed(0)}%
        </span>
      </div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
        <svg viewBox={`0 0 ${S} ${S}`} style={{ width: "100%", maxWidth: S, border: "1px solid var(--border)", borderRadius: 6 }} role="img" aria-label="Граница решения MLP">
          {cells}
          {data.map((p, k) => (
            <circle
              key={k}
              cx={toPx(p.x)}
              cy={S - toPx(p.y)}
              r={4}
              fill={p.t > 0 ? "var(--series-1)" : "var(--series-2)"}
              stroke="var(--panel)"
              strokeWidth={1.5}
            />
          ))}
        </svg>
        <svg viewBox="0 0 270 100" style={{ width: 270, maxWidth: "100%" }} role="img" aria-label="Кривая лосса">
          <text x={0} y={96} fontSize="11" fill="var(--muted)">
            loss по шагам
          </text>
          <line x1={0} y1={80} x2={260} y2={80} stroke="var(--border)" />
          {L.length > 1 && <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={1.8} />}
        </svg>
      </div>
      <p className="viz-caption">
        Фон показывает знак и уверенность выхода сети в каждой точке плоскости. Шаг — тот же, что в train(): все точки,
        MSE, backward, θ ← θ − lr·∇θ. Снимите галочку tanh: скрытый слой станет линейным, вся сеть тоже, и граница
        превратится в прямую. Кольца, XOR и спирали так не разделить. С H = 2 кольца тоже плохо разделяются: двух
        «полуплоскостей» мало, чтобы окружить центр. Слишком большой lr заставляет лосс скакать.
      </p>
    </div>
  );
}
