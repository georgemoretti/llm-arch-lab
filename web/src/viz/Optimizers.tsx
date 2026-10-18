// V-optimizers: траектории SGD, SGD + momentum и Adam на 2D-поверхности лосса — овраг, повёрнутый овраг или
// Розенброк. Формулы шагов — те же, что в задаче (как в torch.optim).
import { useMemo, useState } from "react";
import { Segmented, Slider } from "./common";

type Surf = "ravine" | "rotated" | "rosen";
const SURF: Record<Surf, { f(x: number, y: number): number; g(x: number, y: number): [number, number]; start: [number, number]; box: [number, number, number, number] }> = {
  ravine: {
    f: (x, y) => 0.5 * (x * x + 40 * y * y),
    g: (x, y) => [x, 40 * y],
    start: [-4.5, 0.8],
    box: [-5, 5, -1.5, 1.5],
  },
  rotated: {
    f: (x, y) => {
      const u = (x + y) / Math.SQRT2;
      const v = (x - y) / Math.SQRT2;
      return 0.5 * (u * u + 40 * v * v);
    },
    g: (x, y) => {
      const u = (x + y) / Math.SQRT2;
      const v = (x - y) / Math.SQRT2;
      return [(u + 40 * v) / Math.SQRT2, (u - 40 * v) / Math.SQRT2];
    },
    start: [-3.5, -2.5],
    box: [-4, 4, -4, 4],
  },
  rosen: {
    f: (x, y) => (1 - x) ** 2 + 20 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 80 * x * (y - x * x), 40 * (y - x * x)],
    start: [-1.5, 2.2],
    box: [-2, 2, -1, 3],
  },
};

function trajectories(s: Surf, lr: number, mu: number, adamLr: number, steps: number) {
  const { g, start } = SURF[s];
  const sgd: [number, number][] = [start];
  const mom: [number, number][] = [start];
  const adam: [number, number][] = [start];
  let p = [...start];
  for (let t = 0; t < steps; t++) {
    const [gx, gy] = g(p[0], p[1]);
    p = [p[0] - lr * gx, p[1] - lr * gy];
    sgd.push([p[0], p[1]]);
  }
  p = [...start];
  let b = [0, 0];
  for (let t = 0; t < steps; t++) {
    const [gx, gy] = g(p[0], p[1]);
    b = [mu * b[0] + gx, mu * b[1] + gy];
    p = [p[0] - lr * b[0], p[1] - lr * b[1]];
    mom.push([p[0], p[1]]);
  }
  p = [...start];
  let m = [0, 0];
  let v = [0, 0];
  for (let t = 1; t <= steps; t++) {
    const gr = g(p[0], p[1]);
    m = m.map((mi, i) => 0.9 * mi + 0.1 * gr[i]);
    v = v.map((vi, i) => 0.999 * vi + 0.001 * gr[i] * gr[i]);
    p = p.map((pi, i) => pi - (adamLr * (m[i] / (1 - 0.9 ** t))) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8));
    adam.push([p[0], p[1]]);
  }
  return { sgd, mom, adam };
}

const W = 520;
const H = 300;
const GRID = 44;

export default function Optimizers() {
  const [surf, setSurf] = useState<Surf>("ravine");
  const [logLr, setLogLr] = useState(-1.7);
  const [mu, setMu] = useState(0.9);
  const [logAdam, setLogAdam] = useState(-1);
  const [steps, setSteps] = useState(60);
  const lr = 10 ** logLr;
  const adamLr = 10 ** logAdam;
  const S = SURF[surf];
  const [x0, x1, y0, y1] = S.box;
  const sx = (x: number) => ((x - x0) / (x1 - x0)) * W;
  const sy = (y: number) => H - ((y - y0) / (y1 - y0)) * H;
  const tr = useMemo(() => trajectories(surf, lr, mu, adamLr, steps), [surf, lr, mu, adamLr, steps]);

  const bg = useMemo(() => {
    const cells = [];
    let maxL = 0;
    const vals: number[] = [];
    for (let i = 0; i < GRID; i++)
      for (let j = 0; j < GRID; j++) {
        const v = Math.log1p(S.f(x0 + ((i + 0.5) / GRID) * (x1 - x0), y0 + ((j + 0.5) / GRID) * (y1 - y0)));
        vals.push(v);
        maxL = Math.max(maxL, v);
      }
    for (let i = 0; i < GRID; i++)
      for (let j = 0; j < GRID; j++)
        cells.push(
          <rect key={`${i}-${j}`} x={(i * W) / GRID} y={H - ((j + 1) * H) / GRID} width={W / GRID + 0.5} height={H / GRID + 0.5} fill="var(--ax-model)" opacity={0.05 + 0.4 * (1 - vals[i * GRID + j] / maxL)} />,
        );
    return cells;
  }, [surf]); // eslint-disable-line react-hooks/exhaustive-deps

  const clip = (v: number, lo: number, hi: number) => Math.max(lo - 1e3, Math.min(hi + 1e3, Number.isFinite(v) ? v : hi + 1e3));
  const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${clip(sx(x), 0, W).toFixed(1)},${clip(sy(y), 0, H).toFixed(1)}`).join(" ");
  const last = (pts: [number, number][]) => S.f(...pts[pts.length - 1]);
  const fmtL = (v: number) => (!Number.isFinite(v) || v > 1e6 ? "∞ (разошёлся)" : v.toPrecision(3));
  const series = [
    { id: "sgd", label: "SGD", color: "var(--series-2)", pts: tr.sgd },
    { id: "mom", label: `SGD + momentum ${mu.toFixed(2)}`, color: "var(--series-1)", pts: tr.mom },
    { id: "adam", label: "Adam", color: "var(--series-3)", pts: tr.adam },
  ];

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={surf}
          options={[
            ["ravine", "овраг"],
            ["rotated", "повёрнутый овраг"],
            ["rosen", "Розенброк"],
          ]}
          onChange={setSurf}
        />
        <Slider label={<>шагов = {steps}</>} value={steps} min={5} max={200} onChange={setSteps} />
      </div>
      <div className="viz-controls">
        <Slider label={<>lr SGD = {lr.toPrecision(2)}</>} value={logLr} min={-3} max={-1.15} step={0.02} onChange={setLogLr} />
        <Slider label={<>momentum = {mu.toFixed(2)}</>} value={mu} min={0} max={0.99} step={0.01} onChange={setMu} />
        <Slider label={<>lr Adam = {adamLr.toPrecision(2)}</>} value={logAdam} min={-2.5} max={0} step={0.02} onChange={setLogAdam} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, border: "1px solid var(--border)", borderRadius: 6 }} role="img" aria-label="Траектории оптимизаторов">
        {bg}
        <circle cx={sx(surf === "rosen" ? 1 : 0)} cy={sy(surf === "rosen" ? 1 : 0)} r={5} fill="none" stroke="var(--text)" strokeWidth={1.5} />
        {series.map((s) => (
          <path key={s.id} d={path(s.pts)} fill="none" stroke={s.color} strokeWidth={1.8} strokeLinejoin="round" />
        ))}
        <circle cx={sx(S.start[0])} cy={sy(S.start[1])} r={4} fill="var(--text)" />
      </svg>
      <div className="mono" style={{ fontSize: 12.5, margin: "6px 0", display: "flex", gap: 16, flexWrap: "wrap" }}>
        {series.map((s) => (
          <span key={s.id}>
            <span style={{ color: s.color }}>■</span> {s.label}: loss {fmtL(last(s.pts))}
          </span>
        ))}
      </div>
      <p className="viz-caption">
        Фон показывает лосс (чем темнее, тем меньше), кружок — минимум, точка — старт. В овраге кривизна по вертикали в 40
        раз больше, чем по горизонтали. SGD с безопасным lr почти стоит вдоль оврага, а с lr больше 2/40 = 0.05 начинает
        прыгать между стенками и расходится. Momentum гасит прыжки и разгоняется вдоль дна. Adam нормирует каждую
        координату отдельно и идёт к минимуму почти по диагонали, шагами около lr. В повёрнутом овраге оси не совпадают
        с координатами, и это преимущество Adam пропадает.
      </p>
    </div>
  );
}
