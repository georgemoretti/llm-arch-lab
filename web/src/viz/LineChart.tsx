// Маленький SVG-график для визуализаций: линии на общей оси x, crosshair + тултип при наведении,
// легенда при ≥ 2 сериях и подписи на концах линий. Цвета серий — CSS-переменные --series-N.
import { useMemo, useRef, useState } from "react";

export interface Series {
  id: string;
  label: string;
  color: string; // CSS-цвет, обычно var(--series-1)
  ys: number[]; // значения в точках общего xs
}

interface Props {
  xs: number[];
  series: Series[];
  logX?: boolean;
  xTicks: number[];
  xFormat(x: number): string;
  yFormat(y: number): string;
  yLabel: string;
  xLabel: string;
  markerX?: number; // вертикальная отметка (текущее значение слайдера)
  height?: number;
  zeroLine?: boolean; // горизонтальная линия y = 0 (для знакопеременных данных)
}

const W = 560;
const PAD = { l: 52, r: 96, t: 10, b: 34 };

export function LineChart(p: Props) {
  const H = p.height ?? 220;
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const { sx, sy, yTicks } = useMemo(() => {
    const fx = (x: number) => (p.logX ? Math.log(x) : x);
    const x0 = fx(p.xs[0]);
    const x1 = fx(p.xs[p.xs.length - 1]);
    const all = p.series.flatMap((s) => s.ys);
    let lo = Math.min(0, ...all);
    let hi = Math.max(...all);
    if (hi === lo) hi = lo + 1;
    const ticks = niceTicks(lo, hi, 4);
    lo = Math.min(lo, ticks[0]);
    hi = Math.max(hi, ticks[ticks.length - 1]);
    return {
      sx: (x: number) => PAD.l + ((fx(x) - x0) / (x1 - x0)) * (W - PAD.l - PAD.r),
      sy: (y: number) => PAD.t + (1 - (y - lo) / (hi - lo)) * (H - PAD.t - PAD.b),
      yTicks: ticks,
    };
  }, [p.xs, p.series, p.logX, H]);

  const onMove = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    p.xs.forEach((v, i) => {
      if (Math.abs(sx(v) - x) < Math.abs(sx(p.xs[best]) - x)) best = i;
    });
    setHover(best);
  };

  const path = (ys: number[]) => ys.map((y, i) => `${i ? "L" : "M"}${sx(p.xs[i]).toFixed(1)},${sy(y).toFixed(1)}`).join("");
  const last = p.xs.length - 1;
  // подписи на концах линий, раздвинутые, чтобы не налезали друг на друга
  const ends = p.series
    .map((s) => ({ s, y: sy(s.ys[last]) }))
    .sort((a, b) => a.y - b.y)
    .reduce<{ s: Series; y: number }[]>((acc, e) => {
      const prev = acc[acc.length - 1];
      acc.push({ s: e.s, y: prev && e.y - prev.y < 13 ? prev.y + 13 : e.y });
      return acc;
    }, []);

  return (
    <div className="lc">
      {p.series.length >= 2 && (
        <div className="lc-legend">
          {p.series.map((s) => (
            <span key={s.id}>
              <i style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <div className="lc-wrap">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="lc-svg"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          role="img"
          aria-label={`${p.yLabel} от ${p.xLabel}`}
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} className={t === 0 && p.zeroLine ? "lc-zero" : "lc-grid"} />
              <text x={PAD.l - 6} y={sy(t) + 4} className="lc-tick" textAnchor="end">
                {p.yFormat(t)}
              </text>
            </g>
          ))}
          {p.xTicks.map((t) => (
            <text key={t} x={sx(t)} y={H - PAD.b + 16} className="lc-tick" textAnchor="middle">
              {p.xFormat(t)}
            </text>
          ))}
          <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} className="lc-axis" textAnchor="middle">
            {p.xLabel}
          </text>
          <text x={12} y={(PAD.t + H - PAD.b) / 2} className="lc-axis" textAnchor="middle" transform={`rotate(-90 12 ${(PAD.t + H - PAD.b) / 2})`}>
            {p.yLabel}
          </text>
          {p.markerX !== undefined && (
            <line x1={sx(p.markerX)} x2={sx(p.markerX)} y1={PAD.t} y2={H - PAD.b} className="lc-marker" />
          )}
          {p.series.map((s) => (
            <path key={s.id} d={path(s.ys)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
          ))}
          {ends.map(({ s, y }) => (
            <text key={s.id} x={W - PAD.r + 6} y={y + 4} className="lc-end">
              {s.label}
            </text>
          ))}
          {hover !== null && (
            <g>
              <line x1={sx(p.xs[hover])} x2={sx(p.xs[hover])} y1={PAD.t} y2={H - PAD.b} className="lc-cross" />
              {p.series.map((s) => (
                <circle key={s.id} cx={sx(p.xs[hover])} cy={sy(s.ys[hover])} r={4} fill={s.color} className="lc-dot" />
              ))}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div
            className="lc-tip"
            style={{ left: `${(sx(p.xs[hover]) / W) * 100}%`, transform: sx(p.xs[hover]) > W / 2 ? "translateX(calc(-100% - 10px))" : "translateX(10px)" }}
          >
            <b>
              {p.xLabel}: {p.xFormat(p.xs[hover])}
            </b>
            {p.series.map((s) => (
              <div key={s.id}>
                <i style={{ background: s.color }} />
                {s.label}: {p.yFormat(s.ys[hover])}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function niceTicks(lo: number, hi: number, n: number): number[] {
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  if (out[out.length - 1] < hi) out.push(out[out.length - 1] + step);
  return out;
}
