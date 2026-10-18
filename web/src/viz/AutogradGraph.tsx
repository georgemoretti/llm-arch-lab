// V-autograd-graph: выражение как граф вычислений. Ползунки меняют входы (прямой проход), кнопки шагают по
// обратному проходу: узел за узлом от выхода ко входам, вклад каждого ребра = локальная производная × grad узла.
import { useMemo, useState } from "react";
import { fmt, Segmented, Slider } from "./common";

type Op = "in" | "+" | "*" | "tanh" | "exp";
type NodeDef = { id: string; label: string; op: Op; inputs: string[]; col: number; row: number };
type Expr = { title: string; nodes: NodeDef[]; inputs: Record<string, [number, number, number]> };

const EXPRS: Record<"neuron" | "reuse", Expr> = {
  neuron: {
    title: "L = tanh(w·x + b)",
    inputs: { x: [-2, 2, 1.5], w: [-2, 2, -0.8], b: [-2, 2, 0.4] },
    nodes: [
      { id: "x", label: "x", op: "in", inputs: [], col: 0, row: 0 },
      { id: "w", label: "w", op: "in", inputs: [], col: 0, row: 1 },
      { id: "b", label: "b", op: "in", inputs: [], col: 0, row: 2 },
      { id: "wx", label: "w·x", op: "*", inputs: ["w", "x"], col: 1, row: 0.5 },
      { id: "z", label: "z = w·x + b", op: "+", inputs: ["wx", "b"], col: 2, row: 1.25 },
      { id: "L", label: "L = tanh(z)", op: "tanh", inputs: ["z"], col: 3, row: 1.25 },
    ],
  },
  reuse: {
    title: "c = a·b,  d = c + a,  L = d·c",
    inputs: { a: [-2, 2, 1.2], b: [-2, 2, -0.5] },
    nodes: [
      { id: "a", label: "a", op: "in", inputs: [], col: 0, row: 0.4 },
      { id: "b", label: "b", op: "in", inputs: [], col: 0, row: 1.8 },
      { id: "c", label: "c = a·b", op: "*", inputs: ["a", "b"], col: 1, row: 1.6 },
      { id: "d", label: "d = c + a", op: "+", inputs: ["c", "a"], col: 2, row: 0.4 },
      { id: "L", label: "L = d·c", op: "*", inputs: ["d", "c"], col: 3, row: 1.1 },
    ],
  },
};

type Computed = { data: Record<string, number>; local: Record<string, number[]> };

function forward(expr: Expr, vals: Record<string, number>): Computed {
  const data: Record<string, number> = {};
  const local: Record<string, number[]> = {};
  for (const n of expr.nodes) {
    const xs = n.inputs.map((i) => data[i]);
    if (n.op === "in") data[n.id] = vals[n.id];
    else if (n.op === "+") [data[n.id], local[n.id]] = [xs[0] + xs[1], [1, 1]];
    else if (n.op === "*") [data[n.id], local[n.id]] = [xs[0] * xs[1], [xs[1], xs[0]]];
    else if (n.op === "tanh") {
      const t = Math.tanh(xs[0]);
      [data[n.id], local[n.id]] = [t, [1 - t * t]];
    } else {
      const e = Math.exp(xs[0]);
      [data[n.id], local[n.id]] = [e, [e]];
    }
  }
  return { data, local };
}

const W = 660;
const COL_W = 170;
const ROW_H = 70;
const NODE_W = 120;
const NODE_H = 44;

export default function AutogradGraph() {
  const [which, setWhich] = useState<"neuron" | "reuse">("reuse");
  const expr = EXPRS[which];
  const [vals, setVals] = useState<Record<string, number>>({});
  const [step, setStep] = useState(0);
  const value = (id: string) => vals[`${which}.${id}`] ?? expr.inputs[id][2];

  const { data, local } = useMemo(
    () => forward(expr, Object.fromEntries(Object.keys(expr.inputs).map((k) => [k, value(k)]))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [which, vals],
  );

  // обратный топологический порядок: узлы определены по возрастанию глубины → просто разворачиваем
  const order = [...expr.nodes].reverse().filter((n) => n.op !== "in");
  const steps = order.length;
  const done = order.slice(0, step);

  // градиенты после `step` шагов + вклады последнего шага
  const grad: Record<string, number | undefined> = { L: 1 };
  const contrib: { from: string; to: string; local: number; g: number }[] = [];
  for (const n of done) {
    n.inputs.forEach((c, k) => {
      const add = local[n.id][k] * (grad[n.id] ?? 0);
      grad[c] = (grad[c] ?? 0) + add;
      if (n === done[done.length - 1]) contrib.push({ from: n.id, to: c, local: local[n.id][k], g: grad[n.id] ?? 0 });
    });
  }
  const consumers = (id: string) => expr.nodes.filter((n) => n.inputs.includes(id));
  const complete = (id: string) => id === "L" || consumers(id).every((u) => done.includes(u));

  const pos = (n: NodeDef) => ({ x: 20 + n.col * COL_W, y: 14 + n.row * ROW_H });
  const byId = Object.fromEntries(expr.nodes.map((n) => [n.id, n]));
  const H = 14 + Math.max(...expr.nodes.map((n) => n.row)) * ROW_H + NODE_H + 14;
  const active = done[done.length - 1]?.id;

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={which}
          options={[
            ["reuse", "переиспользование узлов"],
            ["neuron", "нейрон"],
          ]}
          onChange={(v) => {
            setWhich(v);
            setStep(0);
          }}
        />
        {Object.entries(expr.inputs).map(([k, [lo, hi]]) => (
          <Slider
            key={`${which}.${k}`}
            label={<>{k} = {value(k).toFixed(2)}</>}
            value={value(k)}
            min={lo}
            max={hi}
            step={0.05}
            onChange={(v) => setVals({ ...vals, [`${which}.${k}`]: v })}
          />
        ))}
      </div>
      <div className="viz-controls">
        <button className="btn small" onClick={() => setStep(0)} disabled={step === 0}>
          ⟲ только прямой проход
        </button>
        <button className="btn small" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>
          ← шаг
        </button>
        <button className="btn small" onClick={() => setStep(Math.min(steps, step + 1))} disabled={step === steps}>
          шаг backward →
        </button>
        <span className="muted">
          {step === 0 ? "L.grad = 1, остальные градиенты ещё не посчитаны" : `обработано узлов: ${step} из ${steps}`}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} role="img" aria-label={`Граф вычислений ${expr.title}`}>
        <defs>
          <marker id="ag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--border-strong)" />
          </marker>
        </defs>
        {expr.nodes.flatMap((n) =>
          n.inputs.map((c, k) => {
            const a = pos(byId[c]);
            const b = pos(n);
            const x1 = a.x + NODE_W;
            const y1 = a.y + NODE_H / 2;
            const x2 = b.x;
            const y2 = b.y + NODE_H / 2 + (n.inputs.length > 1 ? (k - 0.5) * 12 : 0);
            const hot = n.id === active;
            const mx = (x1 + x2) / 2;
            const my = (y1 + y2) / 2;
            return (
              <g key={`${c}-${n.id}-${k}`}>
                <path
                  d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                  fill="none"
                  stroke={hot ? "var(--series-2)" : "var(--border-strong)"}
                  strokeWidth={hot ? 2.5 : 1.5}
                  markerEnd="url(#ag-arrow)"
                />
                <text x={mx} y={my - 4} textAnchor="middle" fontSize="11" fill={hot ? "var(--series-2)" : "var(--muted)"} className="mono">
                  ∂={fmt(local[n.id][k])}
                </text>
              </g>
            );
          }),
        )}
        {expr.nodes.map((n) => {
          const p = pos(n);
          const g = grad[n.id];
          const full = g !== undefined && complete(n.id);
          return (
            <g key={n.id}>
              <rect
                x={p.x}
                y={p.y}
                width={NODE_W}
                height={NODE_H}
                rx={7}
                fill="var(--panel)"
                stroke={n.id === active ? "var(--series-2)" : n.op === "in" ? "var(--series-1)" : "var(--border-strong)"}
                strokeWidth={n.id === active ? 2.5 : 1.5}
              />
              <text x={p.x + 8} y={p.y + 17} fontSize="12" fill="var(--text)">
                {n.label}
              </text>
              <text x={p.x + 8} y={p.y + 35} fontSize="11" className="mono" fill="var(--muted)">
                {fmt(data[n.id])}
                <tspan fill={g === undefined ? "var(--muted)" : full ? "var(--series-3)" : "var(--series-2)"}>
                  {"  "}grad {g === undefined ? "?" : fmt(g)}
                  {g !== undefined && !full ? "…" : ""}
                </tspan>
              </text>
            </g>
          );
        })}
      </svg>
      {contrib.length > 0 && (
        <ul className="mono" style={{ margin: "6px 0", paddingLeft: 18, fontSize: 12.5 }}>
          {contrib.map((c, i) => (
            <li key={i}>
              {byId[c.to].label.split(" ")[0]}.grad += ∂{c.from}/∂{byId[c.to].label.split(" ")[0]} · {c.from}.grad = {fmt(c.local)} · {fmt(c.g)} ={" "}
              {fmt(c.local * c.g)}
            </li>
          ))}
        </ul>
      )}
      <p className="viz-caption">
        Числа на рёбрах — локальные производные, их записывает прямой проход. Обратный проход идёт от L ко входам.
        Обработать узел значит раздать его входам вклады «локальная производная × grad узла». Градиент{" "}
        <span style={{ color: "var(--series-2)" }}>оранжевый с «…»</span>, пока пришли не все вклады, и{" "}
        <span style={{ color: "var(--series-3)" }}>зелёный</span>, когда он полный. В режиме «переиспользование» у a и c
        по два потребителя: их вклады складываются. Поэтому узел можно обрабатывать только после всех его потребителей.
      </p>
    </div>
  );
}
