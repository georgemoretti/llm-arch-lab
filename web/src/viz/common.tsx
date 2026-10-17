// Общие детали визуализаций: детерминированный ГПСЧ, softmax и маленькие SVG-столбики.
import type { ReactNode } from "react";

/** mulberry32: картинка не прыгает при перерисовке, «новые данные» = новый seed. */
export function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(next: () => number) {
  const u = Math.max(next(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
}

export const randn = (next: () => number, n: number) => Array.from({ length: n }, () => gaussian(next));

/** softmax с поддержкой −Infinity (замаскированные позиции получают ровно 0). */
export function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  if (m === -Infinity) return xs.map(() => 0);
  const e = xs.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
export const rms = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

/** Энтропия распределения в битах — «на сколько вариантов размазано внимание». */
export const entropyBits = (p: number[]) => -p.reduce((s, v) => (v > 0 ? s + v * Math.log2(v) : s), 0);

export function Slider(props: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange(v: number): void;
  width?: number;
}) {
  return (
    <label>
      {props.label}
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={props.value}
        style={props.width ? { width: props.width } : undefined}
        onChange={(e) => props.onChange(+e.target.value)}
      />
    </label>
  );
}

/**
 * Ряд столбиков со знаком вокруг нулевой линии. `max` — общий масштаб (чтобы два ряда
 * можно было сравнивать на глаз); по умолчанию — максимум |value| в ряду.
 */
export function Bars(props: {
  values: number[];
  max?: number;
  height?: number;
  color?: string;
  colors?: (string | undefined)[];
  dimmed?: boolean[];
  labels?: string[];
  titles?: string[];
  signed?: boolean;
  barWidth?: number;
}) {
  const n = props.values.length;
  const H = props.height ?? 70;
  const bw = props.barWidth ?? 22;
  const gap = 4;
  const W = n * (bw + gap) - gap;
  const signed = props.signed ?? props.values.some((v) => v < 0);
  const max = props.max ?? Math.max(1e-12, ...props.values.map(Math.abs));
  const zero = signed ? H / 2 : H;
  const scale = (signed ? H / 2 - 2 : H - 2) / max;
  const labelH = props.labels ? 14 : 0;
  return (
    <svg className="viz-bars" width={W} height={H + labelH} viewBox={`0 0 ${W} ${H + labelH}`} style={{ maxWidth: "100%", height: "auto" }}>
      <line x1={0} x2={W} y1={zero} y2={zero} className="viz-bars-zero" />
      {props.values.map((v, i) => {
        const h = Math.min(Math.abs(v) * scale, signed ? H / 2 : H);
        const y = v >= 0 ? zero - h : zero;
        return (
          <g key={i} opacity={props.dimmed?.[i] ? 0.25 : 1}>
            <rect x={i * (bw + gap)} y={y} width={bw} height={Math.max(h, 0.5)} rx={2} fill={props.colors?.[i] ?? props.color ?? "var(--series-1)"}>
              <title>{props.titles?.[i] ?? v.toFixed(3)}</title>
            </rect>
            {props.labels && (
              <text x={i * (bw + gap) + bw / 2} y={H + 11} textAnchor="middle" className="viz-bars-label">
                {props.labels[i]}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Подпись + ряд столбиков + справа короткая статистика. */
export function BarRow(props: { title: ReactNode; stat?: ReactNode; children: ReactNode }) {
  return (
    <div className="viz-barrow">
      <div className="viz-barrow-title">{props.title}</div>
      <div className="viz-barrow-bars">{props.children}</div>
      {props.stat !== undefined && <div className="viz-barrow-stat">{props.stat}</div>}
    </div>
  );
}

/** Кнопки-переключатель вариантов. */
export function Segmented<T extends string>(props: { value: T; options: [NoInfer<T>, string][]; onChange(v: NoInfer<T>): void }) {
  return (
    <span className="viz-seg" role="radiogroup">
      {props.options.map(([v, label]) => (
        <button key={v} role="radio" aria-checked={props.value === v} className={props.value === v ? "active" : ""} onClick={() => props.onChange(v)}>
          {label}
        </button>
      ))}
    </span>
  );
}

export const fmt = (v: number, d = 2) => (Math.abs(v) < 1e-9 ? (0).toFixed(d) : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(1) : v.toFixed(d));

/** Русское множественное: plural(3, ["голова", "головы", "голов"]). */
export function plural(n: number, forms: [string, string, string]) {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return forms[2];
  if (b === 1) return forms[0];
  if (b >= 2 && b <= 4) return forms[1];
  return forms[2];
}
