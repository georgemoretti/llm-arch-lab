// V-gradcheck: ошибка численной производной от шага h (в float64 браузера — те же double, что в тестах).
// Видна U-образная кривая: справа ошибка формулы (~h у односторонней, ~h² у центральной), слева — округление.
import { useMemo, useState } from "react";
import { Segmented, Slider } from "./common";
import { LineChart } from "./LineChart";

const FUNCS = {
  exp: { label: "eˣ", f: Math.exp, df: Math.exp },
  sin: { label: "sin x", f: Math.sin, df: Math.cos },
  cube: { label: "x³", f: (x: number) => x ** 3, df: (x: number) => 3 * x * x },
  tanh: { label: "tanh x", f: Math.tanh, df: (x: number) => 1 - Math.tanh(x) ** 2 },
};
type FuncId = keyof typeof FUNCS;

const HS = Array.from({ length: 61 }, (_, i) => 10 ** (-14 + i * 0.225)); // 1e-14 … ~3e-1
const FLOOR = -17;

export default function GradCheck() {
  const [fn, setFn] = useState<FuncId>("exp");
  const [x, setX] = useState(0.7);
  const [logH, setLogH] = useState(-5);

  const { fwd, cen } = useMemo(() => {
    const { f, df } = FUNCS[fn];
    const exact = df(x);
    const rel = (g: number) => Math.max(FLOOR, Math.log10(Math.abs(g - exact) / Math.max(Math.abs(exact), 1e-12) || 1e-300));
    return {
      fwd: HS.map((h) => rel((f(x + h) - f(x)) / h)),
      cen: HS.map((h) => rel((f(x + h) - f(x - h)) / (2 * h))),
    };
  }, [fn, x]);

  const h = 10 ** logH;
  const { f, df } = FUNCS[fn];
  const exact = df(x);
  const gF = (f(x + h) - f(x)) / h;
  const gC = (f(x + h) - f(x - h)) / (2 * h);
  const relErr = (g: number) => Math.abs(g - exact) / Math.max(Math.abs(exact), 1e-12);

  return (
    <div>
      <div className="viz-controls">
        <Segmented value={fn} options={Object.entries(FUNCS).map(([k, v]) => [k as FuncId, v.label])} onChange={setFn} />
        <Slider label={<>x = {x.toFixed(2)}</>} value={x} min={-2} max={2} step={0.05} onChange={setX} />
        <Slider label={<>h = 10^{logH.toFixed(1)}</>} value={logH} min={-14} max={-1} step={0.1} onChange={setLogH} />
      </div>
      <LineChart
        xs={HS}
        logX
        series={[
          { id: "fwd", label: "односторонняя", color: "var(--series-2)", ys: fwd },
          { id: "cen", label: "центральная", color: "var(--series-1)", ys: cen },
        ]}
        xTicks={[1e-14, 1e-11, 1e-8, 1e-5, 1e-2]}
        xFormat={(v) => `1e${Math.round(Math.log10(v))}`}
        yFormat={(v) => `1e${Math.round(v)}`}
        yLabel="относительная ошибка"
        xLabel="шаг h"
        markerX={h}
      />
      <div className="mono" style={{ fontSize: 12.5, margin: "6px 0" }}>
        точно {exact.toPrecision(12)} · односторонняя {gF.toPrecision(12)} (ошибка {relErr(gF).toExponential(1)}) · центральная{" "}
        {gC.toPrecision(12)} (ошибка {relErr(gC).toExponential(1)})
      </div>
      <p className="viz-caption">
        Справа от минимума главную роль играет ошибка формулы. У односторонней разности она падает как h (наклон 1),
        у центральной — как h² (наклон 2). Слева от минимума растёт ошибка округления: f(x+h) и f(x−h) почти равны, и
        при вычитании пропадают значащие цифры. Лучшее h у центральной разности 1e-6…1e-5, ошибка порядка 1e-11. У
        односторонней лучшее h около 1e-8, ошибка порядка 1e-9. Для x³ центральная разность ошибается только из-за
        округления: третья производная постоянна, и формула почти точна.
      </p>
    </div>
  );
}
