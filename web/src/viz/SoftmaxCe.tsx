// V-softmax-ce: логиты → softmax → cross-entropy и градиент p − onehot. Множитель масштаба раздувает логиты:
// наивный exp(z)/Σexp(z) в double ломается после e^709, устойчивый (со сдвигом на максимум) — нет.
import { useState } from "react";
import { BarRow, Bars, fmt, Slider } from "./common";

const CLASSES = ["кот", "пёс", "лис", "ёж", "сыч"];

function naive(z: number[]) {
  const e = z.map(Math.exp);
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function stable(z: number[]) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return { p: e.map((v) => v / s), logZ: m + Math.log(s) };
}

const show = (v: number) => (Number.isFinite(v) ? fmt(v, 3) : String(v));

export default function SoftmaxCe() {
  const [base, setBase] = useState([2.0, 0.5, -1.0, 1.2, -0.3]);
  const [target, setTarget] = useState(0);
  const [logScale, setLogScale] = useState(0);
  const scale = 10 ** logScale;
  const z = base.map((v) => v * scale);
  const { p, logZ } = stable(z);
  const loss = logZ - z[target];
  const grad = p.map((v, j) => v - (j === target ? 1 : 0));
  const pn = naive(z);
  const naiveLoss = -Math.log(pn[target]);
  const broken = !pn.every(Number.isFinite) || !Number.isFinite(naiveLoss);

  return (
    <div>
      <div className="viz-controls">
        {base.map((v, i) => (
          <Slider
            key={i}
            label={<>z<sub>{CLASSES[i]}</sub> = {v.toFixed(1)}</>}
            value={v}
            min={-4}
            max={4}
            step={0.1}
            width={90}
            onChange={(nv) => setBase(base.map((b, k) => (k === i ? nv : b)))}
          />
        ))}
      </div>
      <div className="viz-controls">
        <span>
          правильный класс:{" "}
          <select value={target} onChange={(e) => setTarget(+e.target.value)}>
            {CLASSES.map((c, i) => (
              <option key={c} value={i}>
                {c}
              </option>
            ))}
          </select>
        </span>
        <Slider label={<>масштаб логитов ×{scale < 10 ? scale.toFixed(2) : Math.round(scale)}</>} value={logScale} min={-1} max={3} step={0.05} onChange={setLogScale} />
      </div>
      <BarRow title="логиты z" stat={<>max {fmt(Math.max(...z), 1)}</>}>
        <Bars values={z} labels={CLASSES} colors={z.map((_, j) => (j === target ? "var(--series-3)" : undefined))} />
      </BarRow>
      <BarRow title="p = softmax(z)" stat={<>p[цель] = {fmt(p[target], 3)}</>}>
        <Bars values={p} max={1} signed={false} labels={CLASSES} colors={p.map((_, j) => (j === target ? "var(--series-3)" : "var(--series-1)"))} />
      </BarRow>
      <BarRow title="∂ℓ/∂z = p − onehot" stat={<>Σ = {fmt(grad.reduce((a, b) => a + b, 0), 3)}</>}>
        <Bars values={grad} max={1} signed labels={CLASSES} color="var(--series-2)" />
      </BarRow>
      <div className="mono" style={{ fontSize: 12.5, margin: "8px 0" }}>
        loss = log Σ e^z − z[цель] = {show(loss)}
        <br />
        наивно: exp(z) / Σ exp(z) → p[цель] = {show(pn[target])}, loss = {show(naiveLoss)}{" "}
        {broken ? <b style={{ color: "var(--series-2)" }}>← сломалось</b> : <span className="muted">(совпадает)</span>}
      </div>
      <p className="viz-caption">
        Градиент по правильному классу отрицательный: шаг против градиента поднимает его логит. Остальные классы
        получают +p, и их логиты опускаются. Сумма градиента по строке всегда 0. Увеличивайте масштаб: распределение
        становится острее, а когда max z переходит порог 709, наивная формула получает exp = ∞ и ∞/∞ = NaN. Устойчивая
        вычитает максимум и считает тот же ответ. Если правильный класс далеко от максимума, наивная формула ломается
        ещё раньше: p[цель] округляется до 0, и log 0 = −∞.
      </p>
    </div>
  );
}
