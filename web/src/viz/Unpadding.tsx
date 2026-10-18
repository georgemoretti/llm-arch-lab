// V-unpadding: батч с паддингом [B, T] против упаковки [N] (B17): cu_seqlens, позиции RoPE с нуля у каждого текста,
// блочно-диагональная маска упаковки (и окно ModernBERT внутри блока) и доля работы, которую съедает паддинг.
// Переключатель ошибки B17d: номер текста считается строгим сравнением, и первый токен текста уходит к соседу.
import { useState } from "react";
import { Slider } from "./common";

const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const tint = (b: number, pct = 30) => `color-mix(in srgb, ${COLORS[b]} ${pct}%, transparent)`;

export default function Unpadding() {
  const [lens, setLens] = useState([5, 2, 8]);
  const [win, setWin] = useState(false);
  const [W, setW] = useState(4);
  const [bug, setBug] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  const B = lens.length;
  const T = Math.max(...lens);
  const cu = lens.reduce<number[]>((acc, L) => [...acc, acc[acc.length - 1] + L], [0]);
  const N = cu[B];
  // номер текста токена упаковки: сколько концов текстов позади (с ошибкой — строгое сравнение)
  const seqOf = (n: number) => cu.slice(1).filter((c) => (bug ? n > c : n >= c)).length;
  const trueSeq = (n: number) => cu.slice(1).filter((c) => n >= c).length;
  const pos = (n: number) => n - cu[trueSeq(n)];
  const allowed = (i: number, j: number) => seqOf(i) === seqOf(j) && (!win || Math.abs(i - j) <= Math.floor(W / 2));
  const leak = (i: number, j: number) => allowed(i, j) && trueSeq(i) !== trueSeq(j);

  const sumL = N;
  const sumL2 = lens.reduce((s, L) => s + L * L, 0);
  const linear = 1 - sumL / (B * T);
  const attn = 1 - sumL2 / (B * T * T);
  const leaks = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => leak(i, j))).flat().filter(Boolean).length;
  const cell = N > 18 ? 16 : 22;

  return (
    <div>
      <div className="viz-controls">
        {lens.map((L, b) => (
          <Slider
            key={b}
            label={<span style={{ color: COLORS[b] }}>L{b} = {L}</span>}
            value={L}
            min={1}
            max={12}
            onChange={(v) => setLens(lens.map((x, i) => (i === b ? v : x)))}
          />
        ))}
      </div>
      <div className="viz-controls">
        <label className="check">
          <input type="checkbox" checked={win} onChange={(e) => setWin(e.target.checked)} /> локальный слой (окно)
        </label>
        <span style={{ opacity: win ? 1 : 0.4 }}>
          <Slider label={<>W = {W}</>} value={W} min={2} max={12} step={2} onChange={setW} />
        </span>
        <label className="check" style={bug ? { color: "var(--bad)" } : undefined}>
          <input type="checkbox" checked={bug} onChange={(e) => setBug(e.target.checked)} /> ошибка B17d: «&gt;» вместо «≥»
        </label>
      </div>

      <div className="viz-subtitle">Батч с паддингом [B, T] = [{B}, {T}]: заштриховано — PAD, на них тоже считаются все слои.</div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `28px repeat(${T}, ${cell}px)`, marginBottom: 10 }}>
        <div />
        {Array.from({ length: T }, (_, t) => (
          <div key={t} className="viz-axis">
            {t}
          </div>
        ))}
        {lens.map((L, b) => [
          <div key={`r${b}`} className="viz-axis" style={{ color: COLORS[b] }}>
            {b}
          </div>,
          ...Array.from({ length: T }, (_, t) => (
            <div
              key={`${b}-${t}`}
              className={`viz-cell ${t >= L ? "blocked" : ""}`}
              style={{ height: cell, ...(t < L ? { background: tint(b) } : {}) }}
              title={t < L ? `текст ${b}, позиция ${t}` : "PAD"}
            />
          )),
        ])}
      </div>

      <div className="viz-subtitle">
        Упаковка [N] = [{N}], cu_seqlens = [{cu.join(", ")}]; в ячейке — позиция RoPE (с нуля у каждого текста).
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `28px repeat(${N}, ${cell}px)`, marginBottom: 10 }}>
        <div className="viz-axis">n</div>
        {Array.from({ length: N }, (_, n) => (
          <div
            key={n}
            className={`viz-cell ${hover === n ? "active" : ""}`}
            style={{ height: cell, fontSize: 10, background: tint(trueSeq(n)), ...(cu.includes(n) && n > 0 ? { borderLeft: "2px solid var(--text)" } : {}) }}
            title={`токен упаковки ${n}: текст ${trueSeq(n)}, позиция ${pos(n)}`}
            onMouseEnter={() => setHover(n)}
            onMouseLeave={() => setHover(null)}
          >
            {pos(n)}
          </div>
        ))}
      </div>

      <div className="viz-subtitle">
        Маска упаковки [N, N]: строка — запрос, столбец — ключ. Цвет — текст, который «видит» маска
        {bug ? "; красным — чужой текст (утечка)" : ""}.
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `28px repeat(${N}, ${cell}px)` }} onMouseLeave={() => setHover(null)}>
        <div />
        {Array.from({ length: N }, (_, j) => (
          <div key={j} className={`viz-axis ${hover === j ? "active" : ""}`}>
            {N > 18 && j % 2 ? "" : j}
          </div>
        ))}
        {Array.from({ length: N }, (_, i) => [
          <div key={`a${i}`} className={`viz-axis ${hover === i ? "active" : ""}`}>
            {i}
          </div>,
          ...Array.from({ length: N }, (_, j) => {
            const ok = allowed(i, j);
            const bad = leak(i, j);
            return (
              <div
                key={`${i}-${j}`}
                className="viz-cell"
                style={{
                  height: cell,
                  background: bad ? "var(--bad)" : ok ? tint(trueSeq(i), 45) : undefined,
                  opacity: ok ? 1 : 0.35,
                }}
                title={`запрос ${i} (текст ${trueSeq(i)}) → ключ ${j} (текст ${trueSeq(j)}): ${bad ? "УТЕЧКА: чужой текст" : ok ? "можно" : "закрыто"}`}
                onMouseEnter={() => setHover(i)}
              />
            );
          }),
        ])}
      </div>

      <div className="kv-tiles">
        <div>
          <span className="muted">линейные слои, FFN</span>
          <b>{N} вместо {B * T}</b>
          <span className="muted">паддинг съедал {(linear * 100).toFixed(0)} % работы</span>
        </div>
        <div>
          <span className="muted">attention (varlen)</span>
          <b>
            {sumL2} вместо {B * T * T}
          </b>
          <span className="muted">пар; паддинг съедал {(attn * 100).toFixed(0)} %</span>
        </div>
        <div>
          <span className="muted">пар «чужих» текстов в маске</span>
          <b style={leaks ? { color: "var(--bad)" } : undefined}>{leaks}</b>
          <span className="muted">{leaks ? "соседи видят друг друга" : "тексты изолированы"}</span>
        </div>
      </div>
      <p className="viz-caption">
        Текст b занимает в упаковке токены cu[b] … cu[b + 1] − 1, и cu[b + 1] — граница «после последнего». Поэтому
        номер текста токена n — число концов, для которых n ≥ cu: со строгим «&gt;» первый токен каждого текста
        уходит к предыдущему, а формы и cu_seqlens остаются верными. Маска [N, N] здесь учебная: ядро Flash
        Attention с cu_seqlens считает только диагональные блоки и не тратит память на остальное.
      </p>
    </div>
  );
}
