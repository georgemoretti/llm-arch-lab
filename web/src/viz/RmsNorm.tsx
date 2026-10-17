// V-rmsnorm: LayerNorm и RMSNorm на одном векторе. Сдвиг, масштаб и eps — слайдерами.
import { useMemo, useState } from "react";
import { BarRow, Bars, fmt, mean, randn, rms, rng, Segmented, Slider } from "./common";

const C = 16;

export default function RmsNorm() {
  const [shift, setShift] = useState(0);
  const [logScale, setLogScale] = useState(0);
  const [eps, setEps] = useState<"1e-5" | "1e-6">("1e-5");
  const [seed, setSeed] = useState(3);

  const { x, ln, rn } = useMemo(() => {
    const base = randn(rng(seed), C);
    const scale = 10 ** logScale;
    const x = base.map((v) => scale * (v + shift));
    const e = +eps;
    const mu = mean(x);
    const varc = mean(x.map((v) => (v - mu) ** 2));
    const ln = x.map((v) => (v - mu) / Math.sqrt(varc + e));
    const ms = mean(x.map((v) => v * v));
    const rn = x.map((v) => v / Math.sqrt(ms + e));
    return { x, ln, rn };
  }, [shift, logScale, eps, seed]);

  const stat = (a: number[]) => (
    <>
      mean {fmt(mean(a))} · RMS {fmt(rms(a))}
    </>
  );
  const outMax = Math.max(1, ...ln.map(Math.abs), ...rn.map(Math.abs));

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>сдвиг = {shift.toFixed(1)}</>} value={shift} min={-3} max={3} step={0.1} onChange={setShift} />
        <Slider label={<>масштаб = 10^{logScale.toFixed(1)}</>} value={logScale} min={-4} max={3} step={0.1} onChange={setLogScale} />
        <span>
          eps{" "}
          <Segmented
            value={eps}
            options={[
              ["1e-5", "1e-5"],
              ["1e-6", "1e-6"],
            ]}
            onChange={setEps}
          />
        </span>
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новый вектор
        </button>
      </div>
      <BarRow title="x (вход, свой масштаб)" stat={stat(x)}>
        <Bars values={x} color="var(--ax-batch)" />
      </BarRow>
      <BarRow title="LayerNorm(x)" stat={stat(ln)}>
        <Bars values={ln} max={outMax} color="var(--series-2)" />
      </BarRow>
      <BarRow title="RMSNorm(x)" stat={stat(rn)}>
        <Bars values={rn} max={outMax} color="var(--series-1)" />
      </BarRow>
      <p className="viz-caption">
        Масштаб входа не влияет ни на одну норму — до тех пор, пока средний квадрат x много больше eps. Сдвиньте вход:
        LayerNorm вычитает среднее и снова центрирует вектор, RMSNorm — нет, при большом сдвиге все компоненты становятся
        почти одинаковыми. Уменьшите масштаб до 10<sup>−3</sup> и ниже: eps под корнем начинает доминировать, RMS выхода
        падает ниже 1 — так и должно быть, это защищает от деления на ноль. Вес нормы здесь = 1.
      </p>
    </div>
  );
}
