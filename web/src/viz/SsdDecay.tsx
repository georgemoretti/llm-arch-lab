// V-ssd-decay: маска затухания Mamba-2 L[t, s] = exp(a_{s+1} + … + a_t), a_r = Δ_r · A.
import { useMemo, useState } from "react";
import { Slider } from "./common";

const T = 12;

export default function SsdDecay() {
  const [A, setA] = useState(-0.5);
  const [baseDt, setBaseDt] = useState(0.3);
  const [spikes, setSpikes] = useState<Set<number>>(() => new Set([7]));
  const [hover, setHover] = useState<[number, number] | null>(null);

  const dt = useMemo(() => Array.from({ length: T }, (_, r) => (spikes.has(r) ? 6 : baseDt)), [baseDt, spikes]);
  const L = useMemo(() => {
    const cs: number[] = [];
    dt.reduce((acc, d, r) => (cs[r] = acc + d * A), 0);
    return Array.from({ length: T }, (_, t) => Array.from({ length: T }, (_, s) => (s <= t ? Math.exp(cs[t] - cs[s]) : null)));
  }, [dt, A]);

  const toggle = (r: number) =>
    setSpikes((prev) => {
      const n = new Set(prev);
      if (n.has(r)) n.delete(r);
      else n.add(r);
      return n;
    });

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>A = {A.toFixed(2)}</>} value={A} min={-2} max={-0.02} step={0.02} onChange={setA} />
        <Slider label={<>Δ = {baseDt.toFixed(2)}</>} value={baseDt} min={0.02} max={1.5} step={0.02} onChange={setBaseDt} />
        <span className="muted">клик по номеру позиции — большой Δ («забудь прошлое»)</span>
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `28px repeat(${T}, 30px)` }}>
        <div />
        {Array.from({ length: T }, (_, s) => (
          <button key={s} className={`viz-axis ssd-pos ${spikes.has(s) ? "spike" : ""}`} onClick={() => toggle(s)} title={`Δ${s} = ${dt[s].toFixed(2)}`}>
            {s}
          </button>
        ))}
        {L.map((row, t) => (
          <SsdRow key={t} t={t} row={row} hover={hover} setHover={setHover} />
        ))}
      </div>
      <p className="viz-caption">
        Строка t — с каким весом выход в позиции t «помнит» вход позиции s. На диагонали 1, дальше вес затухает как
        exp(Δ · A · расстояние): A &lt; 0 задаёт скорость забывания, Δ — размер шага. Δ зависит от токена (селективность): большой
        Δ в позиции r почти обнуляет всё, что было до r — в матрице появляется «стена». Это та же матрица, что в
        ssd_quadratic умножается поэлементно на C·Bᵀ: SSM = attention с причинной затухающей маской.
        {hover && (
          <>
            {" "}
            L[{hover[0]}, {hover[1]}] = {L[hover[0]][hover[1]]?.toExponential(2)}.
          </>
        )}
      </p>
    </div>
  );
}

function SsdRow({ t, row, hover, setHover }: { t: number; row: (number | null)[]; hover: [number, number] | null; setHover(h: [number, number] | null): void }) {
  return (
    <>
      <div className={`viz-axis ${hover?.[0] === t ? "active" : ""}`}>{t}</div>
      {row.map((v, s) =>
        v === null ? (
          <div key={s} className="viz-cell blocked" style={{ height: 26 }} />
        ) : (
          <div
            key={s}
            className="viz-cell"
            style={{ height: 26, background: `color-mix(in srgb, var(--series-3) ${Math.round(v * 85)}%, transparent)` }}
            onMouseEnter={() => setHover([t, s])}
            onMouseLeave={() => setHover(null)}
          >
            {v >= 0.995 ? "1" : v < 0.005 ? "" : v.toFixed(2).replace(/^0/, "")}
          </div>
        ),
      )}
    </>
  );
}
