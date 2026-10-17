// V-mtp: что предсказывает каждая глубина Multi-Token Prediction и из чего.
import { useState } from "react";
import { Slider } from "./common";

type Cell = { row: number; i: number };

export default function Mtp() {
  const [T, setT] = useState(8);
  const [depth, setDepth] = useState(2);
  const [hover, setHover] = useState<Cell | null>({ row: 1, i: 2 });

  // Глубина k (0 = основная модель), позиция i: вход — токен i + k (для k ≥ 1) и скрытое состояние
  // предыдущей глубины в позиции i; цель — токен i + k + 1. Позиций с целью: T − k − 1.
  const rows = Array.from({ length: depth + 1 }, (_, k) => k);
  const target = hover ? hover.i + hover.row + 1 : null;
  const inputTok = hover && hover.row > 0 ? hover.i + hover.row : null;

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>T = {T}</>} value={T} min={5} max={12} onChange={setT} />
        <Slider label={<>глубин D = {depth}</>} value={depth} min={1} max={3} onChange={setDepth} />
      </div>
      <div className="mtp-grid" style={{ gridTemplateColumns: `120px repeat(${T}, minmax(34px, 1fr))` }}>
        <div className="muted small-text">токены</div>
        {Array.from({ length: T }, (_, j) => (
          <div key={j} className={`mtp-tok ${j === target ? "target" : ""} ${j === inputTok ? "input" : ""} ${hover && hover.row === 0 && j === hover.i ? "input" : ""}`}>
            t{j}
          </div>
        ))}
        {rows.map((k) => (
          <Row key={k} k={k} T={T} hover={hover} setHover={setHover} />
        ))}
      </div>
      <p className="viz-caption">
        {hover ? (
          hover.row === 0 ? (
            <>
              Основная модель в позиции {hover.i} видит t0…t{hover.i} и предсказывает t{hover.i + 1}.
            </>
          ) : (
            <>
              Глубина {hover.row} в позиции {hover.i}: берёт скрытое состояние {hover.row === 1 ? "основной модели" : `глубины ${hover.row - 1}`} в той же
              позиции (оно уже «знает» t0…t{hover.i + hover.row - 1}), добавляет эмбеддинг t{hover.i + hover.row} и предсказывает t{hover.i + hover.row + 1}.
            </>
          )
        ) : (
          "Наведите на ячейку."
        )}{" "}
        Каждая следующая глубина короче на одну позицию: на глубине k целей T − k − 1. Модули MTP нужны при обучении (плотнее
        сигнал) и для спекулятивного декодинга; обычная генерация использует только основную модель.
      </p>
    </div>
  );
}

function Row({ k, T, hover, setHover }: { k: number; T: number; hover: Cell | null; setHover(c: Cell | null): void }) {
  return (
    <>
      <div className="mtp-rowlabel">{k === 0 ? "основная модель" : `глубина ${k}`}</div>
      {Array.from({ length: T }, (_, i) => {
        const tgt = i + k + 1;
        const valid = tgt < T;
        const active = hover?.row === k && hover.i === i;
        const chain = hover && hover.i === i && k < hover.row; // откуда пришло скрытое состояние
        return (
          <div
            key={i}
            className={`mtp-cell ${valid ? "" : "empty"} ${active ? "active" : ""} ${chain ? "chain" : ""}`}
            onMouseEnter={() => valid && setHover({ row: k, i })}
          >
            {valid ? `→ t${tgt}` : ""}
          </div>
        );
      })}
    </>
  );
}
