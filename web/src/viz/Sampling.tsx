// V-sampling: temperature → top-k → top-p над одними и теми же логитами (порядок как в задаче).
import { useMemo, useState } from "react";
import { Bars, entropyBits, rng, Slider, softmax } from "./common";

// Игрушечное продолжение «Кошка села на …»: логиты придуманы для наглядности.
const TOKENS = ["коврик", "диван", "стул", "пол", "окно", "кровать", "крышу", "траву", "лавку", "место", "стол", "руки"];
const LOGITS = [3.1, 2.6, 2.2, 2.0, 1.4, 1.2, 0.7, 0.4, 0.1, -0.3, -0.8, -1.5];
const V = TOKENS.length;

export default function Sampling() {
  const [temp, setTemp] = useState(1);
  const [k, setK] = useState(V);
  const [p, setP] = useState(1);
  const [draw, setDraw] = useState<{ tok: number; n: number } | null>(null);

  const { probs, kept } = useMemo(() => {
    if (temp === 0) {
      const best = LOGITS.indexOf(Math.max(...LOGITS));
      return { probs: LOGITS.map((_, i) => (i === best ? 1 : 0)), kept: LOGITS.map((_, i) => i === best) };
    }
    let z = LOGITS.map((l) => l / temp);
    // top-k: всё, что меньше k-го по величине, → −inf
    const kth = [...z].sort((a, b) => b - a)[Math.min(k, V) - 1];
    z = z.map((v) => (v < kth ? -Infinity : v));
    // top-p: оставляем токен, если масса строго выше него < p (первый всегда остаётся)
    const order = z.map((_, i) => i).sort((a, b) => z[b] - z[a]);
    const ps = softmax(z);
    let before = 0;
    for (const i of order) {
      if (before >= p) z[i] = -Infinity;
      before += ps[i];
    }
    const probs = softmax(z);
    return { probs, kept: z.map((v) => v !== -Infinity) };
  }, [temp, k, p]);

  const sample = () => {
    const next = rng(Math.floor(Math.random() * 1e9));
    let u = next();
    let tok = V - 1;
    for (let i = 0; i < V; i++) {
      u -= probs[i];
      if (u <= 0) {
        tok = i;
        break;
      }
    }
    setDraw((d) => ({ tok, n: (d?.n ?? 0) + 1 }));
  };

  const base = softmax(LOGITS);

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>T = {temp.toFixed(2)}</>} value={temp} min={0} max={2.5} step={0.05} onChange={setTemp} />
        <Slider label={<>top-k = {k}</>} value={k} min={1} max={V} onChange={setK} />
        <Slider label={<>top-p = {p.toFixed(2)}</>} value={p} min={0.05} max={1} step={0.01} onChange={setP} />
        <button className="btn small primary" onClick={sample}>
          сэмплировать
        </button>
        {draw && (
          <span>
            → <b>{TOKENS[draw.tok]}</b> <span className="muted">(#{draw.n})</span>
          </span>
        )}
      </div>
      <div className="sampling-bars">
        <Bars
          values={probs}
          max={Math.max(...probs, ...base)}
          colors={probs.map((_, i) => (draw?.tok === i ? "var(--series-2)" : "var(--series-1)"))}
          dimmed={kept.map((x) => !x)}
          labels={TOKENS.map((t) => (t.length > 7 ? t.slice(0, 6) + "…" : t))}
          barWidth={40}
          titles={TOKENS.map((t, i) => `${t}: было ${(100 * base[i]).toFixed(1)}%, стало ${(100 * probs[i]).toFixed(1)}%`)}
          height={110}
        />
      </div>
      <p className="viz-caption">
        Кандидатов: {kept.filter(Boolean).length} из {V}, энтропия {entropyBits(probs).toFixed(2)} бит (у исходного softmax —{" "}
        {entropyBits(base).toFixed(2)}). Температура делит логиты: T &lt; 1 заостряет распределение, T &gt; 1 размывает, T = 0 —
        argmax. top-k оставляет k самых вероятных, top-p — минимальный «верх», чья масса дошла до p; после отсечения
        вероятности перенормируются. Наведите на столбик — увидите, как изменилась вероятность токена.
      </p>
    </div>
  );
}
