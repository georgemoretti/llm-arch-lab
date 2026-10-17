// V-nope: слой без позиционного кодирования не замечает перестановку прошлых токенов, слой с RoPE — замечает.
import { useMemo, useState } from "react";
import { BarRow, Bars, dot, randn, rng, Segmented, softmax } from "./common";

const D = 8;
const T = 6;
const THETA = 10000;
const TOKEN_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--ax-model)", "var(--ax-wide)", "var(--ax-other)"];

// RoPE с раскладкой половинами (пара (j, j + D/2)), как в задаче про RoPE.
function rope(v: number[], m: number): number[] {
  const half = D / 2;
  const out = v.slice();
  for (let j = 0; j < half; j++) {
    const a = m * THETA ** ((-2 * j) / D);
    const [x1, x2] = [v[j], v[j + half]];
    out[j] = x1 * Math.cos(a) - x2 * Math.sin(a);
    out[j + half] = x2 * Math.cos(a) + x1 * Math.sin(a);
  }
  return out;
}

function lastTokenAttention(tokens: number[], q: number[][], k: number[][], v: number[][], useRope: boolean) {
  const t = tokens.length - 1;
  const qt = useRope ? rope(q[tokens[t]], t) : q[tokens[t]];
  const w = softmax(tokens.map((tok, s) => dot(qt, useRope ? rope(k[tok], s) : k[tok]) / Math.sqrt(D)));
  const out = Array.from({ length: D }, (_, i) => tokens.reduce((acc, tok, s) => acc + w[s] * v[tok][i], 0));
  return { w, out };
}

export default function Nope() {
  const [seed, setSeed] = useState(5);
  const [order, setOrder] = useState<number[]>(() => Array.from({ length: T }, (_, i) => i));
  const [mode, setMode] = useState<"nope" | "rope">("nope");

  // У каждого «слова» свои q, k, v (как после проекций одного эмбеддинга).
  const { q, k, v } = useMemo(() => {
    const next = rng(seed);
    const mk = () => Array.from({ length: T }, () => randn(next, D).map((x) => x * 0.9));
    return { q: mk(), k: mk(), v: mk() };
  }, [seed]);

  const base = Array.from({ length: T }, (_, i) => i);
  const ref = lastTokenAttention(base, q, k, v, mode === "rope");
  const cur = lastTokenAttention(order, q, k, v, mode === "rope");
  const diff = Math.sqrt(cur.out.reduce((s, x, i) => s + (x - ref.out[i]) ** 2, 0)) / Math.sqrt(ref.out.reduce((s, x) => s + x * x, 0));
  const max = Math.max(...ref.out.map(Math.abs), ...cur.out.map(Math.abs));

  const shuffle = () => {
    const next = rng(Date.now() % 100000);
    const prev = order.slice(0, T - 1);
    for (let i = prev.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      [prev[i], prev[j]] = [prev[j], prev[i]];
    }
    setOrder([...prev, order[T - 1]]);
  };

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={mode}
          options={[
            ["nope", "NoPE-слой"],
            ["rope", "RoPE-слой"],
          ]}
          onChange={setMode}
        />
        <button className="btn small" onClick={shuffle}>
          перемешать прошлые токены
        </button>
        <button className="btn small" onClick={() => setOrder(base)}>
          исходный порядок
        </button>
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые слова
        </button>
      </div>
      <div className="nope-seq">
        {order.map((tok, s) => (
          <span key={s} className={`nope-tok ${s === T - 1 ? "last" : ""}`} style={{ borderColor: TOKEN_COLORS[tok] }}>
            <span className="muted">{s}</span> слово {tok}
          </span>
        ))}
      </div>
      <BarRow title="веса внимания последнего токена" stat="по позициям">
        <Bars values={cur.w} max={1} colors={order.map((tok) => TOKEN_COLORS[tok])} labels={order.map((tok) => `с${tok}`)} />
      </BarRow>
      <BarRow title="выход, исходный порядок">
        <Bars values={ref.out} max={max} color="var(--ax-batch)" />
      </BarRow>
      <BarRow title="выход сейчас" stat={<>отличие {(100 * diff).toFixed(1)}%</>}>
        <Bars values={cur.out} max={max} color={mode === "rope" ? "var(--series-2)" : "var(--series-1)"} />
      </BarRow>
      <p className="viz-caption">
        Без позиционного кодирования score зависит только от содержимого q и k, поэтому для последнего токена прошлое — «мешок
        слов»: перестановка меняет порядок весов, но не выход (отличие 0%). С RoPE score зависит ещё и от расстояния, и тот же
        набор слов в другом порядке даёт другой выход. Порядок NoPE-слой всё же различает — через causal-маску: токен в
        позиции t видит ровно t + 1 соседей, и у разных позиций разный набор.
      </p>
    </div>
  );
}
