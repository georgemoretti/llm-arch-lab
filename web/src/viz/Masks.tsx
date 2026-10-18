// V-masks: маски внимания — causal и sliding window, расписание sliding/full (и RoPE/NoPE) по слоям
// реальных моделей, рост поля зрения при наложении SWA-слоёв, sparse-маска top-k в стиле DSA (C18).
// Числа расписаний — из config.json на HF.
import { useState } from "react";

type Mode = "mask" | "layers" | "field" | "sparse";

export default function Masks() {
  const [mode, setMode] = useState<Mode>("mask");
  return (
    <div className="viz-masks">
      <div className="seg" role="tablist">
        {(
          [
            ["mask", "Маска слоя"],
            ["layers", "Расписание слоёв"],
            ["field", "Поле зрения стека"],
            ["sparse", "Sparse (DSA)"],
          ] as [Mode, string][]
        ).map(([id, label]) => (
          <button key={id} role="tab" aria-selected={mode === id} className={mode === id ? "active" : ""} onClick={() => setMode(id)}>
            {label}
          </button>
        ))}
      </div>
      {mode === "mask" && <MaskView />}
      {mode === "layers" && <LayersView />}
      {mode === "field" && <FieldView />}
      {mode === "sparse" && <SparseView />}
    </div>
  );
}

// ---------------------------------------------------------------- маска одного слоя

function MaskView() {
  const [T, setT] = useState(12);
  const [W, setW] = useState(4);
  const [swa, setSwa] = useState(true);
  const [bi, setBi] = useState(false);
  const [row, setRow] = useState<number | null>(null);
  const win = swa ? Math.min(W, T) : T;
  const half = Math.floor(W / 2);
  const cell = T > 16 ? 18 : T > 10 ? 24 : 30;

  // энкодер (ModernBERT): окно симметричное, |p − j| ≤ W/2; декодер: causal и окно назад
  const kind = (t: number, s: number) =>
    bi ? (swa && Math.abs(t - s) > half ? "outside" : "allowed") : s > t ? "future" : t - s >= win ? "outside" : "allowed";
  const seen = (t: number) => Array.from({ length: T }, (_, s) => kind(t, s)).filter((k) => k === "allowed").length;

  return (
    <>
      <div className="viz-controls">
        <label>
          T = {T}
          <input type="range" min={4} max={24} value={T} onChange={(e) => setT(+e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={swa} onChange={(e) => setSwa(e.target.checked)} /> скользящее окно
        </label>
        <label className="check">
          <input type="checkbox" checked={bi} onChange={(e) => setBi(e.target.checked)} /> энкодер (двунаправленное, ModernBERT)
        </label>
        <label style={{ opacity: swa ? 1 : 0.4 }}>
          W = {W}
          <input type="range" min={1} max={24} value={W} disabled={!swa} onChange={(e) => setW(+e.target.value)} />
        </label>
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `30px repeat(${T}, ${cell}px)` }} onMouseLeave={() => setRow(null)}>
        <div />
        {Array.from({ length: T }, (_, s) => (
          <div key={s} className="viz-axis">
            {T > 16 && s % 2 ? "" : `k${s}`}
          </div>
        ))}
        {Array.from({ length: T }, (_, t) => (
          <Row
            key={t}
            t={t}
            T={T}
            cell={cell}
            kind={kind}
            active={row === t}
            onHover={setRow}
            tip={bi ? (p, j, k) => `|p − j| = ${Math.abs(p - j)}${k === "outside" ? ` > W/2 = ${half}: за окном` : ": видно"}` : undefined}
          />
        ))}
      </div>
      <div className="mask-legend">
        <span>
          <i className="mask-swatch allowed" /> можно смотреть
        </span>
        <span>
          <i className="mask-swatch outside" /> {bi ? "за окном" : "прошлое за окном"}
        </span>
        {!bi && (
          <span>
            <i className="mask-swatch future" /> будущее (causal)
          </span>
        )}
      </div>
      <p className="viz-caption">
        {bi ? (
          swa ? (
            <>
              Энкодер смотрит в обе стороны, и окно симметричное: запрос p видит ключи p − {half} … p + {half}, условие{" "}
              <code>|p − j| ≤ W / 2</code> (у ModernBERT local_attention = 128 → ±64, 129 ключей). Разрешённых ячеек:{" "}
              {Array.from({ length: T }, (_, t) => seen(t)).reduce((a, b) => a + b, 0)} вместо {T * T}.
            </>
          ) : (
            <>Глобальный слой энкодера: каждый запрос видит все T ключей, слева и справа.</>
          )
        ) : swa ? (
          <>
            Окно из W = {W} позиций включает сам токен: запрос на позиции p видит ключи p − W + 1 … p, условие{" "}
            <code>0 ≤ p − j &lt; W</code>. При W ≥ T маска совпадает с обычной causal. Число разрешённых ячеек:{" "}
            {Array.from({ length: T }, (_, t) => Math.min(t + 1, win)).reduce((a, b) => a + b, 0)} вместо {(T * (T + 1)) / 2}.
          </>
        ) : (
          <>Обычная causal-маска: запрос на позиции p видит все ключи 0 … p. Число пар растёт как T²/2.</>
        )}
        {row !== null && (
          <>
            {" "}
            {bi ? (
              <>Сейчас: q<sub>{row}</sub> видит {seen(row)} из {T} ключей.</>
            ) : (
              <>
                Сейчас: q<sub>{row}</sub> видит {Math.min(row + 1, win)} из {row + 1} прошлых позиций
                {swa && row + 1 > win ? ` (ключи ${row - win + 1}…${row})` : ""}.
              </>
            )}
          </>
        )}
      </p>
    </>
  );
}

function Row(props: {
  t: number;
  T: number;
  cell: number;
  kind: (t: number, s: number) => string;
  active: boolean;
  onHover: (t: number | null) => void;
  tip?: (t: number, s: number, kind: string) => string;
}) {
  const { t, T, cell, kind, active, onHover, tip } = props;
  return (
    <>
      <div className={`viz-axis ${active ? "active" : ""}`}>q{t}</div>
      {Array.from({ length: T }, (_, s) => {
        const k = kind(t, s);
        return (
          <div
            key={s}
            className={`mask-cell ${k} ${active ? "active" : ""}`}
            style={{ height: cell }}
            onMouseEnter={() => onHover(t)}
            title={
              tip ? tip(t, s, k) : k === "future" ? `j=${s} > p=${t}: будущее` : k === "outside" ? `p − j = ${t - s} ≥ W: за окном` : `p − j = ${t - s}: в окне`
            }
          />
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------- расписание слоёв

interface Schedule {
  id: string;
  name: string;
  layers: number;
  window: number | null; // null — sliding-слоёв нет
  pattern: number; // каждый pattern-й слой (с 1) — full; 0 — все слои sliding; 1 — все full
  firstFull?: boolean; // full — слои 0, pattern, 2·pattern, … (ModernBERT)
  nopeInterval?: number; // каждый N-й слой (с 1) — без RoPE
  source: string;
  note: string;
}

const SCHEDULES: Schedule[] = [
  { id: "llama3", name: "Llama 3 8B", layers: 32, window: null, pattern: 1, source: "meta-llama/Meta-Llama-3-8B", note: "все слои — полное causal attention с RoPE" },
  { id: "mistral", name: "Mistral 7B v0.1", layers: 32, window: 4096, pattern: 0, source: "mistralai/Mistral-7B-v0.1", note: "sliding_window = 4096 во всех слоях" },
  { id: "gemma2", name: "Gemma 2 9B", layers: 42, window: 4096, pattern: 2, source: "google/gemma-2-9b", note: "sliding и full чередуются 1 : 1" },
  { id: "gemma3", name: "Gemma 3 27B", layers: 62, window: 1024, pattern: 6, source: "google/gemma-3-27b-pt", note: "sliding_window_pattern = 6 → 5 sliding : 1 full" },
  { id: "modernbert", name: "ModernBERT-base (энкодер)", layers: 22, window: 128, pattern: 3, firstFull: true, source: "answerdotai/ModernBERT-base", note: "global_attn_every_n_layers = 3 → full: 0, 3, …, 21; окно 128 = ±64 в обе стороны" },
  { id: "smollm3", name: "SmolLM3 3B", layers: 36, window: null, pattern: 1, nopeInterval: 4, source: "HuggingFaceTB/SmolLM3-3B", note: "все слои full, каждый 4-й — без RoPE (NoPE)" },
];

const isFull = (s: Schedule, i: number) => s.pattern !== 0 && (s.firstFull ? i % s.pattern === 0 : (i + 1) % s.pattern === 0);
const isNope = (s: Schedule, i: number) => !!s.nopeInterval && (i + 1) % s.nopeInterval === 0;

function LayersView() {
  const [sid, setSid] = useState("gemma3");
  const [hover, setHover] = useState<number | null>(null);
  const s = SCHEDULES.find((x) => x.id === sid)!;
  const nFull = Array.from({ length: s.layers }, (_, i) => isFull(s, i)).filter(Boolean).length;
  const nNope = Array.from({ length: s.layers }, (_, i) => isNope(s, i)).filter(Boolean).length;

  return (
    <>
      <div className="viz-controls">
        <label className="check">
          модель
          <select value={sid} onChange={(e) => setSid(e.target.value)}>
            {SCHEDULES.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">
          {s.note} ·{" "}
          <a href={`https://huggingface.co/${s.source}/blob/main/config.json`} target="_blank" rel="noreferrer">
            config.json
          </a>
        </span>
      </div>
      <div className="layer-strip" onMouseLeave={() => setHover(null)}>
        {Array.from({ length: s.layers }, (_, i) => {
          const full = isFull(s, i);
          return (
            <div
              key={i}
              className={`layer-cell ${full ? "full" : "sliding"} ${isNope(s, i) ? "nope" : ""}`}
              onMouseEnter={() => setHover(i)}
              title={`слой ${i}: ${full ? "full attention" : `sliding, окно ${s.window}`}${isNope(s, i) ? ", без RoPE" : ""}`}
            >
              {i}
            </div>
          );
        })}
      </div>
      <div className="mask-legend">
        {s.pattern !== 1 && (
          <span>
            <i className="mask-swatch layer-sliding" /> sliding (окно {s.window})
          </span>
        )}
        <span>
          <i className="mask-swatch layer-full" /> full
        </span>
        {s.nopeInterval && (
          <span>
            <i className="mask-swatch layer-nope" /> без RoPE (NoPE)
          </span>
        )}
      </div>
      <p className="viz-caption">
        {s.layers} слоёв: {s.layers - nFull} sliding, {nFull} full
        {nNope ? `; ${nNope} слоёв без RoPE (3, 7, 11, …)` : ""}. Индексы с 0; по правилу Gemma слой i — full, если (i + 1)
        делится на pattern, поэтому слой 0 всегда sliding. По 12 слоёв в строке — так повторяющийся паттерн выстраивается в
        столбцы.
        {hover !== null && (
          <>
            {" "}
            Слой {hover}: {isFull(s, hover) ? "видит весь контекст" : `видит последние ${s.window} токенов`}
            {isNope(s, hover) ? ", позиции знает только через маску" : ""}.
          </>
        )}
      </p>
    </>
  );
}

// ---------------------------------------------------------------- поле зрения стека

function FieldView() {
  const [W, setW] = useState(3);
  const [L, setL] = useState(4);
  const T = 24;
  const t = T - 1;
  return (
    <>
      <div className="viz-controls">
        <label>
          W = {W}
          <input type="range" min={1} max={8} value={W} onChange={(e) => setW(+e.target.value)} />
        </label>
        <label>
          слоёв = {L}
          <input type="range" min={1} max={6} value={L} onChange={(e) => setL(+e.target.value)} />
        </label>
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `58px repeat(${T}, 16px)` }}>
        <div />
        {Array.from({ length: T }, (_, s) => (
          <div key={s} className="viz-axis">
            {s % 4 === 3 ? s : ""}
          </div>
        ))}
        {Array.from({ length: L }, (_, li) => {
          const l = L - li; // сверху — выход последнего слоя
          const reach = l * (W - 1);
          return (
            <FieldRow key={l} label={`слой ${l}`} T={T} lo={Math.max(0, t - reach)} hi={t} />
          );
        })}
      </div>
      <p className="viz-caption">
        Строка «слой l» — от каких входных токенов зависит выход слоя l в последней позиции ({t}). Каждый слой с окном W
        добавляет W − 1 позиций назад: после {L} слоёв — {L * (W - 1)} позиций (тест{" "}
        <code>test_stacked_receptive_field</code> в C07). Так Mistral 7B с окном 4096 и 32 слоями формально «дотягивается» на
        ~131K токенов назад — но только через цепочку слоёв, а не прямым вниманием.
      </p>
    </>
  );
}

function FieldRow({ label, T, lo, hi }: { label: string; T: number; lo: number; hi: number }) {
  return (
    <>
      <div className="viz-axis">{label}</div>
      {Array.from({ length: T }, (_, s) => (
        <div key={s} className={`mask-cell ${s >= lo && s <= hi ? "allowed" : "outside"}`} style={{ height: 16 }} title={`токен ${s}`} />
      ))}
    </>
  );
}

// ---------------------------------------------------------------- sparse-маска top-k (DSA, C18)

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function SparseView() {
  const [T, setT] = useState(16);
  const [k, setK] = useState(4);
  const [loc, setLoc] = useState(1.5);
  const [seed, setSeed] = useState(7);
  const [row, setRow] = useState<number | null>(null);
  const cell = T > 16 ? 18 : 24;

  // Игрушечные оценки индексатора: «важность» ключа (несколько якорных токенов) + близость + шум пары.
  const rnd = mulberry32(seed);
  const anchors = new Set<number>([0, 2 + Math.floor(rnd() * (T - 4))]);
  const imp = Array.from({ length: T }, (_, s) => (anchors.has(s) ? 2.2 : 0.3 * rnd()));
  const score = Array.from({ length: T }, (_, t) =>
    Array.from({ length: T }, (_, s) => (s > t ? -Infinity : imp[s] + loc * Math.exp(-(t - s) / 3) + 0.6 * rnd())),
  );
  const keep = score.map((r) => {
    const order = r.map((v, s) => [v, s] as [number, number]).filter(([v]) => isFinite(v)).sort((a, b) => b[0] - a[0]);
    return new Set(order.slice(0, k).map(([, s]) => s));
  });
  const kind = (t: number, s: number) => (s > t ? "future" : keep[t].has(s) ? "allowed" : "outside");
  const farPicks = keep.reduce((acc, ks, t) => acc + [...ks].filter((s) => t - s >= k).length, 0);
  const total = keep.reduce((a, ks) => a + ks.size, 0);

  return (
    <>
      <div className="viz-controls">
        <label>
          T = {T}
          <input type="range" min={6} max={24} value={T} onChange={(e) => setT(+e.target.value)} />
        </label>
        <label>
          top-k = {k}
          <input type="range" min={1} max={12} value={k} onChange={(e) => setK(+e.target.value)} />
        </label>
        <label>
          вес близости = {loc.toFixed(1)}
          <input type="range" min={0} max={4} step={0.1} value={loc} onChange={(e) => setLoc(+e.target.value)} />
        </label>
        <button onClick={() => setSeed((x) => x + 1)}>новые оценки</button>
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `30px repeat(${T}, ${cell}px)` }} onMouseLeave={() => setRow(null)}>
        <div />
        {Array.from({ length: T }, (_, s) => (
          <div key={s} className="viz-axis" title={anchors.has(s) ? "якорный токен: высокая «важность» для всех запросов" : undefined}>
            {anchors.has(s) ? `★${s}` : T > 16 && s % 2 ? "" : `k${s}`}
          </div>
        ))}
        {Array.from({ length: T }, (_, t) => (
          <Row
            key={t}
            t={t}
            T={T}
            cell={cell}
            kind={kind}
            active={row === t}
            onHover={setRow}
            tip={(t, s, kd) => (kd === "future" ? `s=${s} > t=${t}: будущее` : `оценка ${score[t][s].toFixed(2)}: ${kd === "allowed" ? "выбран" : "не выбран"}`)}
          />
        ))}
      </div>
      <div className="mask-legend">
        <span>
          <i className="mask-swatch allowed" /> выбран индексатором (top-k)
        </span>
        <span>
          <i className="mask-swatch outside" /> прошлое, не выбрано
        </span>
        <span>
          <i className="mask-swatch future" /> будущее (−inf ещё до top-k)
        </span>
      </div>
      <p className="viz-caption">
        Каждый запрос выбирает свои k ключей по оценкам индексатора — маска разная в каждой строке, в отличие от окна. Оценки
        здесь игрушечные: «важность» ключа (★ — якорные токены) + бонус за близость + шум. В первых k строках допустимых ключей
        меньше k — выбираются все (topk_mask отбрасывает −inf). Пар в attention: {total} вместо {(T * (T + 1)) / 2}; из них{" "}
        {farPicks} — дальше, чем дотянулось бы окно W = k. Поставь вес близости 4 — маска почти превратится в sliding window;
        0 — в выбор по важности. У DeepSeek-V3.2-Exp k = 2048 (index_topk в config.json) при контексте в десятки тысяч токенов.
        {row !== null && (
          <>
            {" "}
            Сейчас: q<sub>{row}</sub> смотрит на ключи {[...keep[row]].sort((a, b) => a - b).join(", ")}.
          </>
        )}
      </p>
    </>
  );
}
