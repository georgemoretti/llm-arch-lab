// V-norm-placement: где стоит норма в блоке и как от этого растёт RMS остаточного потока.
import { useMemo, useState } from "react";
import { Segmented, Slider } from "./common";
import { LineChart } from "./LineChart";

type Variant = "post" | "pre" | "olmo" | "sandwich";

const VARIANTS: Record<Variant, { label: string; who: string; lines: [string, string] }> = {
  post: { label: "Post-Norm", who: "GPT-1, BERT, исходный Transformer", lines: ["x = norm(x + attn(x))", "x = norm(x + mlp(x))"] },
  pre: { label: "Pre-Norm", who: "GPT-2, Llama", lines: ["x = x + attn(norm(x))", "x = x + mlp(norm(x))"] },
  olmo: { label: "OLMo 2", who: "норма на выходе подслоя", lines: ["x = x + norm(attn(x))", "x = x + norm(mlp(x))"] },
  sandwich: { label: "Sandwich", who: "Gemma 2 / 3: норма до и после", lines: ["x = x + norm(attn(norm(x)))", "x = x + norm(mlp(norm(x)))"] },
};

const LAYERS = Array.from({ length: 33 }, (_, i) => i);

export default function NormPlacement() {
  const [variant, setVariant] = useState<Variant>("pre");
  const [gain, setGain] = useState(1.5);

  // Модель: вход остатка с RMS 1, каждый подслой добавляет вектор, независимый от потока.
  // Его RMS = gain у Pre-Norm (зависит от весов) и ровно 1 у OLMo 2 / sandwich (норма на выходе, вес 1).
  // Независимые слагаемые складываются по квадратам: RMS² = 1 + (число подслоёв) · RMS²(подслоя).
  const series = useMemo(
    () => [
      { id: "post", label: "Post-Norm", color: "var(--series-3)", ys: LAYERS.map(() => 1) },
      { id: "pre", label: "Pre-Norm", color: "var(--series-2)", ys: LAYERS.map((l) => Math.sqrt(1 + 2 * l * gain * gain)) },
      { id: "olmo", label: "OLMo 2 / sandwich", color: "var(--series-1)", ys: LAYERS.map((l) => Math.sqrt(1 + 2 * l)) },
    ],
    [gain],
  );
  const v = VARIANTS[variant];

  return (
    <div>
      <div className="viz-controls">
        <Segmented value={variant} options={(Object.keys(VARIANTS) as Variant[]).map((k) => [k, VARIANTS[k].label])} onChange={setVariant} />
      </div>
      <div className="norm-block">
        <div className="norm-block-who muted">{v.who}</div>
        {v.lines.map((line, i) => (
          <code key={i} className="norm-line">
            {line.split(/(norm)/).map((part, j) => (part === "norm" ? <mark key={j}>norm</mark> : <span key={j}>{part}</span>))}
          </code>
        ))}
      </div>
      <div className="viz-controls">
        <Slider label={<>RMS выхода подслоя у Pre-Norm = {gain.toFixed(2)}</>} value={gain} min={0.25} max={4} step={0.05} onChange={setGain} width={120} />
      </div>
      <LineChart
        xs={LAYERS}
        series={series}
        xTicks={[0, 8, 16, 24, 32]}
        xFormat={(x) => `${x}`}
        yFormat={(y) => y.toFixed(2)}
        xLabel="блок"
        yLabel="RMS остатка"
        height={210}
      />
      <p className="viz-caption">
        Post-Norm нормирует сам остаточный поток — его RMS всегда 1, но градиент к ранним слоям проходит через L норм, и без
        warmup обучение глубокой сети нестабильно. Pre-Norm оставляет чистый путь x → x, зато каждый подслой добавляет в поток
        столько, сколько позволяют его веса, и RMS растёт с глубиной тем быстрее, чем больше веса. Норма на выходе подслоя
        (OLMo 2, sandwich) делает рост предсказуемым: ≈ √(1 + 2L) при любых весах. Упрощение: вклады подслоёв независимы и
        складываются по квадратам.
      </p>
    </div>
  );
}
