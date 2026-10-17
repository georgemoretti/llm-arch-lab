// V-state-memory: состояние фиксированного размера как ассоциативная память.
// Линейное внимание только складывает k ⊗ v, DeltaNet перезаписывает, Gated DeltaNet ещё и забывает.
import { useMemo, useState } from "react";
import { randn, rng, Segmented, Slider } from "./common";
import { LineChart } from "./LineChart";

const DK = 16;
const DV = 16;
const STEPS = 64;
const TRIALS = 12; // усредняем по нескольким наборам пар, чтобы кривые не шумели
const XS = Array.from({ length: STEPS }, (_, i) => i + 1);

const unit = (v: number[]) => {
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
};

type Rule = "linear" | "delta" | "gated";

/** Ошибка извлечения ‖Sᵀk − v‖ / ‖v‖ после каждого шага записи. */
function run(rule: Rule, ks: number[][], vs: number[][], beta: number, alpha: number, probe: "first" | "last") {
  const S = Array.from({ length: DK }, () => new Array(DV).fill(0));
  const read = (k: number[]) => Array.from({ length: DV }, (_, j) => k.reduce((acc, ki, i) => acc + ki * S[i][j], 0));
  const errs: number[] = [];
  for (let t = 0; t < STEPS; t++) {
    const k = ks[t];
    const v = vs[t];
    if (rule === "gated") for (const row of S) for (let j = 0; j < DV; j++) row[j] *= alpha;
    const old = rule === "linear" ? new Array(DV).fill(0) : read(k);
    const b = rule === "linear" ? 1 : beta;
    for (let i = 0; i < DK; i++) for (let j = 0; j < DV; j++) S[i][j] += k[i] * b * (v[j] - old[j]);
    const p = probe === "first" ? 0 : t;
    const got = read(ks[p]);
    errs.push(Math.hypot(...got.map((g, j) => g - vs[p][j])) / Math.hypot(...vs[p]));
  }
  return errs;
}

export default function StateMemory() {
  const [beta, setBeta] = useState(1);
  const [alpha, setAlpha] = useState(0.95);
  const [probe, setProbe] = useState<"first" | "last">("last");
  const [seed, setSeed] = useState(2);

  const series = useMemo(() => {
    const next = rng(seed);
    const trials = Array.from({ length: TRIALS }, () => ({
      ks: Array.from({ length: STEPS }, () => unit(randn(next, DK))),
      vs: Array.from({ length: STEPS }, () => randn(next, DV)),
    }));
    const avg = (rule: Rule) => {
      const runs = trials.map(({ ks, vs }) => run(rule, ks, vs, beta, alpha, probe));
      return XS.map((_, t) => runs.reduce((s, r) => s + r[t], 0) / TRIALS);
    };
    return [
      { id: "linear", label: "линейное", color: "var(--series-2)", ys: avg("linear") },
      { id: "delta", label: "DeltaNet", color: "var(--series-1)", ys: avg("delta") },
      { id: "gated", label: "Gated DeltaNet", color: "var(--series-3)", ys: avg("gated") },
    ];
  }, [beta, alpha, probe, seed]);

  return (
    <div>
      <div className="viz-controls">
        <span>
          спрашиваем{" "}
          <Segmented
            value={probe}
            options={[
              ["last", "последнюю пару"],
              ["first", "первую пару"],
            ]}
            onChange={setProbe}
          />
        </span>
        <Slider label={<>β = {beta.toFixed(2)}</>} value={beta} min={0.1} max={1} step={0.05} onChange={setBeta} />
        <Slider label={<>гейт α = {alpha.toFixed(2)}</>} value={alpha} min={0.7} max={1} step={0.01} onChange={setAlpha} />
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые данные
        </button>
      </div>
      <LineChart
        xs={XS}
        series={series}
        xTicks={[1, 16, 32, 48, 64]}
        xFormat={(x) => `${x}`}
        yFormat={(y) => y.toFixed(2)}
        xLabel="записано пар (k, v)"
        yLabel="ошибка чтения"
        markerX={DK}
        height={220}
      />
      <p className="viz-caption">
        Состояние S размером d<sub>k</sub> × d<sub>v</sub> = {DK} × {DV}; по шагу пишем пару (k, v) с |k| = 1 и читаем Sᵀk.
        Линейное внимание только прибавляет k ⊗ v: чужие пары «просвечивают» через неортогональные ключи, и ошибка растёт с
        числом записей даже для последней пары. Правило дельты сначала вычитает то, что S уже помнит по ключу k: при β = 1
        последняя пара читается точно, а старые постепенно затираются. Гейт α &lt; 1 вдобавок плавно забывает всё старое — у
        первой пары ошибка уходит к 1, зато память не забивается. Отметка — t = d<sub>k</sub>: больше d<sub>k</sub> ортогональных
        ключей в состояние не поместить. Среднее по {TRIALS} случайным наборам; упрощение: без φ, нормировки и проекций.
      </p>
    </div>
  );
}
