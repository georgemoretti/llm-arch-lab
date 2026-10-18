// V-lr-schedule: warmup + cosine (как lr_at в задаче), WSD и постоянный lr на одном графике.
import { useState } from "react";
import { Slider } from "./common";
import { LineChart } from "./LineChart";

const T = 1000;

function cosine(t: number, max: number, w: number, total: number, min: number) {
  if (t < w) return (max * (t + 1)) / w;
  if (t >= total) return min;
  const p = (t - w) / Math.max(1, total - w);
  return min + 0.5 * (max - min) * (1 + Math.cos(Math.PI * p));
}

function wsd(t: number, max: number, w: number, total: number, min: number, decayFrac: number) {
  if (t < w) return (max * (t + 1)) / w;
  const decayStart = total - Math.round(decayFrac * total);
  if (t < decayStart) return max;
  if (t >= total) return min;
  return max - ((max - min) * (t - decayStart)) / Math.max(1, total - decayStart);
}

export default function LrSchedule() {
  const [warm, setWarm] = useState(50);
  const [minFrac, setMinFrac] = useState(0.1);
  const [decay, setDecay] = useState(0.2);
  const [step, setStep] = useState(300);
  const xs = Array.from({ length: T }, (_, i) => i);
  const max = 1;
  const min = minFrac * max;
  const cos = xs.map((t) => cosine(t, max, warm, T, min));
  const w = xs.map((t) => wsd(t, max, warm, T, min, decay));
  const constant = xs.map(() => max);

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>warmup = {warm}</>} value={warm} min={0} max={300} step={10} onChange={setWarm} />
        <Slider label={<>min_lr = {minFrac.toFixed(2)}·max</>} value={minFrac} min={0} max={0.5} step={0.01} onChange={setMinFrac} />
        <Slider label={<>WSD: спад = последние {(decay * 100).toFixed(0)}%</>} value={decay} min={0.05} max={0.5} step={0.01} onChange={setDecay} />
        <Slider label={<>шаг = {step}</>} value={step} min={0} max={T - 1} onChange={setStep} />
      </div>
      <LineChart
        xs={xs}
        series={[
          { id: "cos", label: "warmup + cosine", color: "var(--series-1)", ys: cos },
          { id: "wsd", label: "WSD", color: "var(--series-2)", ys: w },
          { id: "const", label: "константа", color: "var(--muted)", ys: constant },
        ]}
        xTicks={[0, 250, 500, 750, 1000]}
        xFormat={(v) => String(v)}
        yFormat={(v) => v.toFixed(2)}
        yLabel="lr / max_lr"
        xLabel="шаг"
        markerX={step}
      />
      <div className="mono" style={{ fontSize: 12.5, margin: "6px 0" }}>
        шаг {step}: cosine {cos[step].toFixed(3)} · WSD {w[step].toFixed(3)}
      </div>
      <p className="viz-caption">
        Warmup поднимает lr от max/warmup на шаге 0 до max за warmup шагов. Cosine начинает спуск сразу после warmup,
        поэтому большую часть обучения lr уже ниже максимума. WSD держит максимум до последних нескольких процентов
        шагов и спускается коротко. Удобно, что любой чекпойнт стабильной фазы можно продолжить: решение о длине
        обучения откладывается до момента спуска. Без warmup (ползунок в 0) расписание сразу начинает с max_lr.
      </p>
    </div>
  );
}
