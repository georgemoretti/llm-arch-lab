// V-training-curve: предзаписанный прогон мини-GPT (A18) на корпусе курса — train/val-лосс по шагам и образцы
// генерации на каждой проверке. Данные: content/viz/training-curve.data.json (solution.py --record).
import { useState } from "react";
import run from "../../../content/viz/training-curve.data.json";
import { LineChart } from "./LineChart";
import { Slider } from "./common";

interface Run {
  steps: number[];
  train: number[];
  val: number[];
  samples: string[];
  best_step: number;
  init_loss: number;
  config: Record<string, number | boolean>;
}

const R = run as Run;

export default function TrainingCurve() {
  const best = R.steps.indexOf(R.best_step);
  const [i, setI] = useState(Math.max(0, best));
  const last = R.steps[R.steps.length - 1];
  const cfg = R.config;
  const gap = R.val[i] - R.train[i];

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>шаг: {R.steps[i]}</>} value={i} min={0} max={R.steps.length - 1} onChange={setI} width={260} />
        <button className="btn small" onClick={() => setI(best)}>
          к минимуму val
        </button>
      </div>
      <div className="kv-tiles">
        <div>
          <span className="muted">train-лосс</span>
          <b>{R.train[i].toFixed(3)}</b>
        </div>
        <div>
          <span className="muted">val-лосс</span>
          <b>{R.val[i].toFixed(3)}</b>
          <span className="muted">перплексия {Math.exp(R.val[i]).toFixed(1)}</span>
        </div>
        <div>
          <span className="muted">val − train</span>
          <b>{gap.toFixed(2)}</b>
          <span className="muted">{gap > 1 ? "заучивает текст" : "ещё обобщает"}</span>
        </div>
        <div>
          <span className="muted">старт (ln V)</span>
          <b>{R.init_loss.toFixed(2)}</b>
        </div>
      </div>
      <LineChart
        xs={R.steps}
        series={[
          { id: "train", label: "train", color: "var(--series-1)", ys: R.train },
          { id: "val", label: "val", color: "var(--series-2)", ys: R.val },
        ]}
        xTicks={[R.steps[0], Math.round(last / 4), Math.round(last / 2), Math.round((3 * last) / 4), last]}
        xFormat={(v) => String(Math.round(v))}
        yFormat={(v) => v.toFixed(1)}
        yLabel="лосс (cross-entropy)"
        xLabel="шаг обучения"
        markerX={R.steps[i]}
        height={220}
      />
      <div className="viz-subtitle">генерация на шаге {R.steps[i]} (начало «Модель », temperature 0.8)</div>
      <div className="mono" style={{ fontSize: 12.5, whiteSpace: "pre-wrap", padding: 8, background: "var(--panel-2)", borderRadius: 6 }}>
        {R.samples[i]}
      </div>
      <p className="viz-caption">
        Прогон эталона A18: {String(cfg.n_layer)} блока, n_embd = {String(cfg.n_embd)}, block_size = {String(cfg.block_size)},{" "}
        {Number(cfg.params).toLocaleString("ru-RU")} параметров, AdamW lr = {String(cfg.lr)}, батч {String(cfg.batch_size)},{" "}
        {last} шагов без ранней остановки. Сначала оба лосса падают от ln V: модель выучивает частые буквы, потом
        пробелы и слоги, потом слова. Минимум val — на шаге {R.best_step}. Дальше train продолжает падать, а val
        растёт: текста всего около 21 тысячи символов, и модель начинает заучивать его, а не обобщать. Сравните
        образцы до и после минимума: поздние сэмплы содержат куски обучающего текста, но на отложенном тексте модель
        ошибается всё сильнее. Скрипт задачи по умолчанию останавливается у минимума и берёт лучшие веса.
      </p>
    </div>
  );
}
