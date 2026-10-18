// V-init-depth: std активаций по слоям глубокой сети без bias при четырёх инициализациях (лог-шкала).
// Считается в браузере: ширина 64, 200 примеров, тот же эксперимент, что activation_stds() в задаче.
import { useMemo, useState } from "react";
import { gaussian, rng, Segmented, Slider } from "./common";
import { LineChart } from "./LineChart";

type Act = "relu" | "tanh" | "linear";
const W = 64;
const N = 200;
const GAIN: Record<Act, number> = { linear: 1, tanh: 5 / 3, relu: Math.SQRT2 };
const INITS = [
  { id: "normal-1", label: "N(0, 1)", color: "var(--series-2)" },
  { id: "normal-0.01", label: "N(0, 0.01²)", color: "var(--ax-heads)" },
  { id: "xavier", label: "Xavier (gain 1)", color: "var(--ax-model)" },
  { id: "kaiming", label: "Kaiming", color: "var(--series-3)" },
] as const;

function run(init: string, act: Act, depth: number, seed: number) {
  const r = rng(seed);
  let x = Array.from({ length: N * W }, () => gaussian(r));
  const f = act === "relu" ? (v: number) => Math.max(0, v) : act === "tanh" ? Math.tanh : (v: number) => v;
  const logStd: number[] = [];
  let saturated = 0;
  for (let l = 0; l < depth; l++) {
    const w = new Float64Array(W * W);
    const a = Math.sqrt(6 / (2 * W));
    for (let i = 0; i < W * W; i++)
      w[i] =
        init === "normal-1" ? gaussian(r) : init === "normal-0.01" ? 0.01 * gaussian(r) : init === "xavier" ? (2 * r() - 1) * a : (GAIN[act] / Math.sqrt(W)) * gaussian(r);
    const y = new Array(N * W);
    for (let n = 0; n < N; n++)
      for (let o = 0; o < W; o++) {
        let s = 0;
        for (let i = 0; i < W; i++) s += x[n * W + i] * w[o * W + i];
        y[n * W + o] = f(s);
      }
    x = y;
    let m = 0;
    for (const v of x) m += v;
    m /= x.length;
    let v2 = 0;
    for (const v of x) v2 += (v - m) ** 2;
    const sd = Math.sqrt(v2 / x.length);
    logStd.push(Number.isFinite(sd) && sd > 0 ? Math.max(-40, Math.min(40, Math.log10(sd))) : sd > 0 ? 40 : -40);
    if (l === depth - 1) saturated = x.filter((v) => Math.abs(v) > 0.99).length / x.length;
  }
  return { logStd, saturated };
}

export default function InitDepth() {
  const [act, setAct] = useState<Act>("relu");
  const [depth, setDepth] = useState(30);
  const [seed, setSeed] = useState(1);
  const runs = useMemo(() => INITS.map((i) => ({ ...i, ...run(i.id, act, depth, seed) })), [act, depth, seed]);
  const xs = Array.from({ length: depth }, (_, i) => i + 1);
  const tanhSat = act === "tanh" ? runs[0].saturated : null;

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={act}
          options={[
            ["relu", "ReLU"],
            ["tanh", "tanh"],
            ["linear", "без активации"],
          ]}
          onChange={setAct}
        />
        <Slider label={<>глубина = {depth}</>} value={depth} min={5} max={50} onChange={setDepth} />
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые веса
        </button>
      </div>
      <LineChart
        xs={xs}
        series={runs.map((r) => ({ id: r.id, label: r.label, color: r.color, ys: r.logStd }))}
        xTicks={[1, 10, 20, 30, 40, 50].filter((t) => t <= depth)}
        xFormat={(v) => String(v)}
        yFormat={(v) => `1e${Math.round(v)}`}
        yLabel="std активаций"
        xLabel="номер слоя"
        zeroLine
      />
      {tanhSat !== null && (
        <div className="mono" style={{ fontSize: 12.5 }}>
          N(0, 1) + tanh: std ≈ 1 выглядит нормально, но {(tanhSat * 100).toFixed(0)}% активаций последнего слоя по модулю {">"} 0.99 — tanh
          насыщен, его производная ≈ 0, градиенты не проходят.
        </div>
      )}
      <p className="viz-caption">
        Линия 1e0 означает, что масштаб сохраняется. Каждый слой умножает дисперсию на fan_in·Var(w)·(множитель
        активации), поэтому ошибка в масштабе весов растёт экспоненциально с глубиной. С ReLU Xavier без множителя √2
        теряет половину дисперсии на каждом слое, а Kaiming её возвращает. С tanh и без активации Xavier и Kaiming при
        квадратных слоях почти совпадают.
      </p>
    </div>
  );
}
