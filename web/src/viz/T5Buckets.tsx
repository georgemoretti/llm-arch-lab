// V-t5-buckets: номер бакета относительной позиции T5 от сдвига key − query (двунаправленный и causal режим),
// матрица бакетов «запрос × ключ» и диапазоны расстояний каждого бакета. Формула — как в B13.
import { useMemo, useState } from "react";
import { LineChart } from "./LineChart";
import { Segmented, Slider } from "./common";

type Mode = "bi" | "causal";

function bucket(r: number, bidirectional: boolean, numBuckets: number, maxDistance: number): number {
  let base = 0;
  let n: number;
  let nb = numBuckets;
  if (bidirectional) {
    nb = Math.floor(nb / 2);
    base = r > 0 ? nb : 0;
    n = Math.abs(r);
  } else {
    n = Math.max(-r, 0);
  }
  const exact = Math.floor(nb / 2);
  if (n < exact) return base + n;
  const large = exact + Math.floor((Math.log(n / exact) / Math.log(maxDistance / exact)) * (nb - exact));
  return base + Math.min(large, nb - 1);
}

const R = 160;

export default function T5Buckets() {
  const [mode, setMode] = useState<Mode>("bi");
  const [nb, setNb] = useState(32);
  const [md, setMd] = useState(128);
  const [T, setT] = useState(16);
  const [hover, setHover] = useState<number | null>(null);
  const bi = mode === "bi";

  const xs = useMemo(() => Array.from({ length: 2 * R + 1 }, (_, i) => i - R), []);
  const ys = xs.map((r) => bucket(r, bi, nb, md));
  const half = bi ? Math.floor(nb / 2) : nb;
  const exact = Math.floor(half / 2);

  // диапазоны расстояний по бакетам (на сдвигах −2000 … 2000)
  const ranges = useMemo(() => {
    const m = new Map<number, [number, number]>();
    for (let r = -2000; r <= 2000; r++) {
      const b = bucket(r, bi, nb, md);
      const cur = m.get(b);
      m.set(b, cur ? [Math.min(cur[0], r), Math.max(cur[1], r)] : [r, r]);
    }
    return m;
  }, [bi, nb, md]);
  // где начинается последний бакет «слева»: дальше всё сливается
  const lastLeft = ranges.get(half - 1);
  const saturate = lastLeft ? -lastLeft[1] : 0;
  const unused = Array.from({ length: nb }, (_, b) => b).filter((b) => !ranges.has(b));

  const fmtRange = (b: number) => {
    const r = ranges.get(b);
    if (!r) return "не используется";
    const [lo, hi] = r;
    const show = (v: number) => (Math.abs(v) >= 2000 ? (v < 0 ? "−∞" : "+∞") : String(v).replace("-", "−"));
    return lo === hi ? show(lo) : `${show(lo)} … ${show(hi)}`;
  };

  const cell = T > 20 ? 18 : 24;
  const hue = (b: number) => Math.round((b / Math.max(1, nb - 1)) * 85);

  return (
    <div>
      <div className="viz-controls">
        <Segmented
          value={mode}
          options={[
            ["bi", "энкодер: двунаправленный"],
            ["causal", "декодер: causal"],
          ]}
          onChange={setMode}
        />
      </div>
      <div className="viz-controls">
        <Slider label={<>num_buckets = {nb}</>} value={nb} min={8} max={64} step={8} onChange={setNb} />
        <Slider label={<>max_distance = {md}</>} value={md} min={16} max={256} step={16} onChange={setMd} />
        <Slider label={<>длина T = {T}</>} value={T} min={6} max={24} onChange={setT} />
      </div>
      <LineChart
        xs={xs}
        series={[{ id: "b", label: "бакет", color: "var(--series-1)", ys }]}
        xTicks={[-160, -128, -64, -16, 0, 16, 64, 128, 160]}
        xFormat={(v) => String(Math.round(v)).replace("-", "−")}
        yFormat={(v) => String(Math.round(v))}
        yLabel="номер бакета"
        xLabel="сдвиг j − i (ключ минус запрос; < 0 — ключ слева)"
        markerX={Math.max(-R, -md)}
        height={210}
      />
      <div className="kv-tiles">
        <div>
          <span className="muted">точных расстояний</span>
          <b>0 … {exact - 1}</b>
          <span className="muted">каждое в своём бакете{bi ? ", слева и справа" : ""}</span>
        </div>
        <div>
          <span className="muted">всё дальше</span>
          <b>{saturate}</b>
          <span className="muted">в последнем бакете (max_distance = {md})</span>
        </div>
        <div>
          <span className="muted">не используются</span>
          <b>{unused.length ? unused.join(", ") : "—"}</b>
          <span className="muted">{bi ? "сдвиг 0 — в первой половине" : "будущее → бакет 0"}</span>
        </div>
      </div>
      <div className="viz-subtitle">
        Матрица бакетов: строка — запрос i, столбец — ключ j; цвет — номер бакета. Наведи на ячейку.
        {!bi && " Пунктир — будущее: бакет 0, ячейку всё равно закроет causal-маска."}
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `28px repeat(${T}, ${cell}px)` }}>
        <div />
        {Array.from({ length: T }, (_, j) => (
          <div key={j} className="viz-axis">
            {j}
          </div>
        ))}
        {Array.from({ length: T }, (_, i) => [
          <div key={`a${i}`} className="viz-axis">
            {i}
          </div>,
          ...Array.from({ length: T }, (_, j) => {
            const b = bucket(j - i, bi, nb, md);
            const future = !bi && j > i;
            return (
              <div
                key={`${i}-${j}`}
                className={`viz-cell ${hover === b ? "active" : ""}`}
                style={{
                  height: cell,
                  fontSize: 10,
                  background: future ? undefined : `color-mix(in srgb, var(--accent) ${12 + hue(b)}%, transparent)`,
                  ...(future ? { borderStyle: "dashed", color: "var(--muted)" } : {}),
                }}
                title={`запрос ${i}, ключ ${j}: сдвиг ${j - i} → бакет ${b} (сдвиги ${fmtRange(b)})`}
                onMouseEnter={() => setHover(b)}
                onMouseLeave={() => setHover(null)}
              >
                {b}
              </div>
            );
          }),
        ])}
      </div>
      {hover !== null && (
        <div className="viz-subtitle">
          бакет {hover}: сдвиги {fmtRange(hover)}
        </div>
      )}
      <p className="viz-caption">
        Близкие сдвиги различаются точно, дальние собираются в корзины, которые расширяются логарифмически. Вдоль
        каждой диагонали матрицы номер один и тот же: bias зависит только от j − i. В энкодере половины бакетов
        отделяют «слева» от «справа», в декодере справа ничего нет, и все {nb} бакетов отданы прошлому — точных
        расстояний вдвое больше. Последний бакет начинается раньше max_distance: формула упирается в номер{" "}
        {half - 1}, не дойдя до {md}.
      </p>
    </div>
  );
}
