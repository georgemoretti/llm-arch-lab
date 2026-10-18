// V-sinusoidal: таблица синусоидальных позиций (Vaswani et al., §3.5) тепловой картой «позиция × столбец» и
// скалярное произведение PE(p)·PE(p + k) от сдвига k — оно одно и то же для любого p.
import { useEffect, useMemo, useRef, useState } from "react";
import { LineChart } from "./LineChart";
import { Slider, fmt } from "./common";

function table(nPos: number, d: number): number[][] {
  return Array.from({ length: nPos }, (_, p) =>
    Array.from({ length: d }, (_, c) => {
      const omega = Math.pow(10000, -(c - (c % 2)) / d);
      return c % 2 === 0 ? Math.sin(p * omega) : Math.cos(p * omega);
    }),
  );
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

export default function Sinusoidal() {
  const [d, setD] = useState(64);
  const [nPos, setNPos] = useState(64);
  const [p, setP] = useState(10);
  const [k, setK] = useState(5);
  const canvas = useRef<HTMLCanvasElement>(null);

  const pe = useMemo(() => table(Math.max(nPos, p + k + 1, 2 * 64 + 1), d), [nPos, d, p, k]);
  const K = 64;
  const ks = Array.from({ length: K + 1 }, (_, i) => i);
  const dots = ks.map((s) => dot(pe[0], pe[s]) / (d / 2));
  const dotsShifted = ks.map((s) => dot(pe[p], pe[p + s]) / (d / 2));

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const css = getComputedStyle(c);
    const pos = css.getPropertyValue("--series-1").trim() || "#3b6fd8";
    const neg = css.getPropertyValue("--series-2").trim() || "#d8743b";
    const W = c.width;
    const H = c.height;
    const cw = W / d;
    const ch = H / nPos;
    ctx.clearRect(0, 0, W, H);
    for (let r = 0; r < nPos; r++)
      for (let col = 0; col < d; col++) {
        const v = pe[r][col];
        ctx.globalAlpha = Math.min(1, Math.abs(v));
        ctx.fillStyle = v >= 0 ? pos : neg;
        ctx.fillRect(col * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
      }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = css.getPropertyValue("--text").trim() || "#000";
    ctx.lineWidth = 1.5;
    for (const r of [p, p + k]) if (r < nPos) ctx.strokeRect(0.5, r * ch + 0.5, W - 1, Math.max(ch - 1, 1));
  }, [pe, d, nPos, p, k]);

  // самая короткая и самая длинная длины волны: 2π/ω_0 и 2π/ω_{d/2−1}
  const longest = (2 * Math.PI) / Math.pow(10000, -(d - 2) / d);

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>d_model = {d}</>} value={d} min={8} max={128} step={8} onChange={setD} />
        <Slider label={<>позиций: {nPos}</>} value={nPos} min={16} max={128} step={8} onChange={setNPos} />
        <Slider label={<>p = {p}</>} value={p} min={0} max={60} onChange={setP} />
        <Slider label={<>k = {k}</>} value={k} min={1} max={60} onChange={setK} />
      </div>
      <div className="viz-subtitle">
        PE: строка — позиция 0 … {nPos - 1}, столбец — 0 … {d - 1} (чётные — sin, нечётные — cos); рамки — позиции p и p + k
      </div>
      <canvas
        ref={canvas}
        width={512}
        height={256}
        style={{ width: "100%", maxWidth: 512, height: "auto", aspectRatio: "2 / 1", borderRadius: 4, border: "1px solid var(--border)" }}
      />
      <div className="kv-tiles">
        <div>
          <span className="muted">PE(p)·PE(p + k) / (d/2)</span>
          <b>{fmt(dot(pe[p], pe[p + k]) / (d / 2), 3)}</b>
          <span className="muted">при p = 0: {fmt(dot(pe[0], pe[k]) / (d / 2), 3)}</span>
        </div>
        <div>
          <span className="muted">‖PE(p)‖²</span>
          <b>{fmt(dot(pe[p], pe[p]), 2)}</b>
          <span className="muted">= d/2 при любом p</span>
        </div>
        <div>
          <span className="muted">длины волн</span>
          <b>{fmt(2 * Math.PI, 1)} … {fmt(longest, 0)}</b>
          <span className="muted">позиций на период</span>
        </div>
      </div>
      <LineChart
        xs={ks}
        series={[
          { id: "p0", label: "от p = 0", color: "var(--series-1)", ys: dots },
          { id: "pp", label: `от p = ${p}`, color: "var(--series-2)", ys: dotsShifted },
        ]}
        xTicks={[0, 16, 32, 48, 64]}
        xFormat={(v) => String(Math.round(v))}
        yFormat={(v) => v.toFixed(1)}
        yLabel="PE(p)·PE(p+k) / (d/2)"
        xLabel="сдвиг k"
        markerX={k}
        height={190}
      />
      <p className="viz-caption">
        Левые столбцы меняются от позиции к позиции быстро (период 2π ≈ 6 позиций), правые — медленно: как стрелки
        часов, они вместе задают номер позиции. Сдвиг на k — поворот каждой пары (sin, cos) на угол k·ω<sub>i</sub>,
        поэтому скалярное произведение зависит только от k: две линии на графике совпадают при любом p. Оно
        максимально при k = 0 и в среднем убывает с расстоянием, но не монотонно — «близость» позиций модели
        ещё предстоит выучить.
      </p>
    </div>
  );
}
