// V-sft-curves: предзаписанный эксперимент D04 — SFT игрушечной модели тремя способами: база + лосс по ответу,
// база + лосс по всем токенам, та же модель без предобучения. Кривые точности формата и ответа на отложенных
// примерах и лосса ответа, таблица по задачам. Данные: content/viz/sft-curves.data.json (solution.py --record).
import { useState } from "react";
import data from "../../../content/viz/sft-curves.data.json";
import { LineChart } from "./LineChart";
import { Segmented, Slider } from "./common";

type Score = { format: number; exact: number };
interface Run {
  steps: number[];
  val_loss: number[];
  format: number[];
  exact: number[];
  by_task: Record<string, Score>[];
}
interface Data {
  config: Record<string, number>;
  plain: Score & { by_task: Record<string, Score> };
  runs: Record<"base_response" | "base_all" | "scratch_response", Run>;
}

const D = data as unknown as Data;
const RUNS = [
  { id: "base_response" as const, label: "база + лосс по ответу", color: "var(--series-1)" },
  { id: "base_all" as const, label: "база + лосс по всем", color: "var(--series-2)" },
  { id: "scratch_response" as const, label: "без предобучения", color: "var(--series-3)" },
];
const TASKS: [string, string][] = [
  ["arith", "арифметика словами"],
  ["upper", "заглавными буквами"],
  ["json", "JSON по шаблону"],
];
type Metric = "exact" | "format" | "val_loss";
const pct = (v: number) => `${Math.round(v * 100)} %`;

export default function SftCurves() {
  const [metric, setMetric] = useState<Metric>("exact");
  const steps = D.runs.base_response.steps;
  const [i, setI] = useState(steps.length - 1);
  const cfg = D.config;
  const last = steps[steps.length - 1];

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={metric}
          options={[
            ["exact", "верный ответ"],
            ["format", "верный формат"],
            ["val_loss", "лосс ответа"],
          ]}
          onChange={setMetric}
        />
        <Slider label={<>шаг SFT: {steps[i]}</>} value={i} min={0} max={steps.length - 1} onChange={setI} width={220} />
      </div>
      <LineChart
        xs={steps}
        series={RUNS.map((r) => ({ id: r.id, label: r.label, color: r.color, ys: D.runs[r.id][metric] }))}
        xTicks={[0, Math.round(last / 4), Math.round(last / 2), Math.round((3 * last) / 4), last]}
        xFormat={(v) => String(Math.round(v))}
        yFormat={(v) => (metric === "val_loss" ? v.toFixed(1) : pct(v))}
        yLabel={metric === "val_loss" ? "лосс на токен ответа" : "доля отложенных примеров"}
        xLabel="шаг SFT"
        markerX={steps[i]}
        height={220}
      />
      <div className="viz-subtitle">Верный ответ по задачам на шаге {steps[i]} (в скобках — верный формат)</div>
      <table className="budget-table">
        <thead>
          <tr>
            <th />
            {TASKS.map(([t, name]) => (
              <th key={t}>{name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="muted">база, сплошной текст «… Ответ: …»</td>
            {TASKS.map(([t]) => (
              <td key={t}>{pct(D.plain.by_task[t]?.exact ?? 0)}</td>
            ))}
          </tr>
          {RUNS.map((r) => (
            <tr key={r.id}>
              <td style={{ color: r.color }}>{r.label}</td>
              {TASKS.map(([t]) => {
                const s = D.runs[r.id].by_task[i][t];
                return (
                  <td key={t}>
                    {pct(s.exact)} <span className="muted">({pct(s.format)})</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="viz-caption">
        Прогон эталона D04: модель {cfg.layers} слоя, d_model = {cfg.d_model}, {Number(cfg.params).toLocaleString("ru-RU")}{" "}
        параметров. «Предобучение» — {cfg.pretrain_steps} шагов на сплошном тексте со всеми фактами, SFT — {cfg.sft_steps}{" "}
        шагов на {cfg.sft_examples} диалогах, проверка — на {cfg.test_examples} отложенных примерах (слова, имена и пары чисел,
        которых в SFT не было). Первая строка таблицы показывает, что база умеет задачи, но только в своём формате: в
        chat-формате на шаге 0 она не отвечает ни на что. SFT учит формат за сотню шагов. Ответы на новые примеры зависят
        от того, что модель вынесла из предобучения: без него формат выучивается, а содержание — нет.
      </p>
    </div>
  );
}
