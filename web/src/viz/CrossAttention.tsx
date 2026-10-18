// V-cross-attention: предзаписанный прогон проекта B06 («число цифрами → число словами»): кривые лосса и exact
// match на отложенных числах, карта весов cross-attention последнего слоя (среднее по головам) для примеров.
// Данные: content/viz/cross-attention.data.json (solution.py --record).
import { useState } from "react";
import run from "../../../content/viz/cross-attention.data.json";
import { LineChart } from "./LineChart";

interface Example {
  n: number;
  src: string[];
  out: string[];
  want: string[];
  attn: number[][];
}

interface Run {
  config: Record<string, number>;
  curve: { step: number; loss: number; val_em: number }[];
  examples: Example[];
  errors: { n: number; want: string; got: string }[];
  val_errors: number;
}

const R = run as Run;

export default function CrossAttention() {
  const [i, setI] = useState(R.examples.findIndex((e) => e.n === 123456));
  const [hover, setHover] = useState<number | null>(null);
  const ex = R.examples[Math.max(0, i)];
  const ok = ex.out.join(" ") === ex.want.join(" ");
  const steps = R.curve.map((c) => c.step);
  const last = steps[steps.length - 1];
  const cell = 40;

  return (
    <div>
      <div className="viz-controls" style={{ flexWrap: "wrap" }}>
        {R.examples.map((e, j) => {
          const good = e.out.join(" ") === e.want.join(" ");
          return (
            <button
              key={e.n}
              className={`btn small ${j === i ? "primary" : ""}`}
              onClick={() => setI(j)}
              title={good ? "переведено верно" : "ошибка модели"}
              style={good ? undefined : { color: j === i ? undefined : "var(--bad)" }}
            >
              {e.n.toLocaleString("ru-RU")}
              {good ? "" : " ✗"}
            </button>
          );
        })}
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `112px repeat(${ex.src.length}, ${cell}px)` }}>
        <div />
        {ex.src.map((c, s) => (
          <div key={s} className="viz-axis">
            {c}
          </div>
        ))}
        {ex.attn.map((r, t) => {
          const wrong = ex.out[t] !== ex.want[t];
          return [
            <div
              key={`w${t}`}
              className={`viz-axis ${hover === t ? "active" : ""}`}
              style={{ justifyContent: "end", paddingRight: 6, ...(wrong ? { color: "var(--bad)" } : {}) }}
              title={wrong ? `надо: ${ex.want[t] ?? "—"}` : undefined}
            >
              {ex.out[t]}
            </div>,
            ...r.map((w, s) => (
              <div
                key={`${t}-${s}`}
                className={`viz-cell ${hover === t ? "active" : ""}`}
                style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(w * 90)}%, transparent)` }}
                title={`«${ex.out[t]}» → цифра ${ex.src[s]} (позиция ${s}): ${w.toFixed(3)}`}
                onMouseEnter={() => setHover(t)}
                onMouseLeave={() => setHover(null)}
              >
                {w >= 0.1 ? w.toFixed(1).replace(/^0/, "") : ""}
              </div>
            )),
          ];
        })}
      </div>
      <div className="viz-subtitle" style={{ marginTop: 8 }}>
        {ok ? (
          <>верно: {ex.want.slice(0, -1).join(" ")}</>
        ) : (
          <>
            <span style={{ color: "var(--bad)" }}>ошибка:</span> {ex.out.slice(0, -1).join(" ")} — надо{" "}
            {ex.want.slice(0, -1).join(" ")}
          </>
        )}
      </div>
      <LineChart
        xs={steps}
        series={[{ id: "em", label: "val EM", color: "var(--series-1)", ys: R.curve.map((c) => c.val_em * 100) }]}
        xTicks={[0, Math.round(last / 3), Math.round((2 * last) / 3), last]}
        xFormat={(v) => String(Math.round(v))}
        yFormat={(v) => `${Math.round(v)}%`}
        yLabel="exact match, %"
        xLabel="шаг обучения"
        height={170}
      />
      <p className="viz-caption">
        Прогон эталона B06: {R.config.train_size} чисел для обучения, {R.config.val_size} отложенных,{" "}
        {R.config.steps} шагов, d_model = {R.config.d_model}, по {R.config.n_layers} слоя в энкодере и декодере. Строка
        карты — сгенерированное слово, столбец — цифра источника, значение — вес cross-attention последнего слоя
        декодера (среднее по головам). Выравнивание никто не размечал: «сто» само смотрит на первую цифру,
        «двадцать» — на вторую, а слово «тысяч(и)» — на цифры после группы тысяч. В конце обучения ошибок на
        отложенных числах {R.val_errors} из {R.config.val_size}, и все они на коротких числах: среди случайных
        чисел до миллиона 90 % шестизначных, коротких в обучении почти нет (✗ — такие примеры).
      </p>
    </div>
  );
}
