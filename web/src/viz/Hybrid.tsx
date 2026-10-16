// V-hybrid: схема стека слоёв (Llama 3 / Gemma 3 / Qwen3-Next) и память инференса по длине контекста
// для «все full», SWA 5 : 1 и гибрида 3 : 1 с Gated DeltaNet. Числа — из config.json на HF (см. C22).
import { useState } from "react";
import { LineChart, type Series } from "./LineChart";

type Kind = "full" | "sliding" | "linear";

interface Preset {
  id: string;
  name: string;
  layers: number;
  kind(i: number): Kind;
  note: string;
  source: string;
}

const PRESETS: Preset[] = [
  { id: "llama3", name: "Llama 3 8B", layers: 32, kind: () => "full", note: "все слои — full attention", source: "meta-llama/Meta-Llama-3-8B" },
  {
    id: "gemma3",
    name: "Gemma 3 27B",
    layers: 62,
    kind: (i) => ((i + 1) % 6 === 0 ? "full" : "sliding"),
    note: "5 sliding (окно 1024) : 1 full",
    source: "google/gemma-3-27b-pt",
  },
  {
    id: "qwen3next",
    name: "Qwen3-Next 80B-A3B",
    layers: 48,
    kind: (i) => ((i + 1) % 4 === 0 ? "full" : "linear"),
    note: "3 Gated DeltaNet : 1 gated attention, full_attention_interval = 4",
    source: "Qwen/Qwen3-Next-80B-A3B-Instruct",
  },
];

const KIND_LABEL: Record<Kind, string> = { full: "full attention", sliding: "sliding window", linear: "Gated DeltaNet (линейный)" };

// Размерности Qwen3-Next (config.json): 48 слоёв, KV-головы 2 × 256; linear: 32 головы × 128 × 128. bf16.
const L = 48;
const KV_PER_TOKEN = 2 * 2 * 256; // K и V одного full-слоя на токен
const STATE = 32 * 128 * 128; // состояние одного Gated DeltaNet-слоя
const BPE = 2;
const CTX = Array.from({ length: 12 }, (_, i) => 512 * 2 ** i); // 512 … 1M
const fmtCtx = (x: number) => (x >= 1 << 20 ? `${x / (1 << 20)}M` : x >= 1024 ? `${x / 1024}K` : String(x));
const fmtBytes = (b: number) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} ГиБ` : `${(b / 2 ** 20).toFixed(0)} МиБ`);

function memAllFull(s: number) {
  return L * KV_PER_TOKEN * s * BPE;
}
function memSwa(s: number, w: number) {
  const full = L / 6; // 8 full-слоёв при 5 : 1
  return (full * KV_PER_TOKEN * s + (L - full) * KV_PER_TOKEN * Math.min(s, w)) * BPE;
}
function memHybrid(s: number) {
  const full = L / 4; // 12 full-слоёв при 3 : 1
  return (full * KV_PER_TOKEN * s + (L - full) * STATE) * BPE;
}

export default function Hybrid() {
  const [pid, setPid] = useState("qwen3next");
  const [hover, setHover] = useState<number | null>(null);
  const [ctxIdx, setCtxIdx] = useState(9);
  const [wIdx, setWIdx] = useState(2);
  const windows = [256, 512, 1024, 2048, 4096];
  const W = windows[wIdx];
  const ctx = CTX[ctxIdx];
  const p = PRESETS.find((x) => x.id === pid)!;
  const kinds = Array.from({ length: p.layers }, (_, i) => p.kind(i));
  const count = (k: Kind) => kinds.filter((x) => x === k).length;

  const series: Series[] = [
    { id: "full", label: "все full", color: "var(--series-2)", ys: CTX.map((s) => memAllFull(s) / 2 ** 30) },
    { id: "swa", label: `SWA 5:1 (W=${W})`, color: "var(--series-1)", ys: CTX.map((s) => memSwa(s, W) / 2 ** 30) },
    { id: "hybrid", label: "гибрид 3:1", color: "var(--series-3)", ys: CTX.map((s) => memHybrid(s) / 2 ** 30) },
  ];

  return (
    <div className="viz-hybrid">
      <div className="viz-controls">
        <label className="check">
          модель
          <select value={pid} onChange={(e) => setPid(e.target.value)}>
            {PRESETS.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">
          {p.note} ·{" "}
          <a href={`https://huggingface.co/${p.source}/blob/main/config.json`} target="_blank" rel="noreferrer">
            config.json
          </a>
        </span>
      </div>
      <div className="layer-strip" onMouseLeave={() => setHover(null)}>
        {kinds.map((k, i) => (
          <div key={i} className={`layer-cell ${k}`} onMouseEnter={() => setHover(i)} title={`слой ${i}: ${KIND_LABEL[k]}`}>
            {i}
          </div>
        ))}
      </div>
      <div className="mask-legend">
        {(["full", "sliding", "linear"] as Kind[])
          .filter((k) => count(k) > 0)
          .map((k) => (
            <span key={k}>
              <i className={`mask-swatch layer-${k}`} /> {KIND_LABEL[k]}: {count(k)}
            </span>
          ))}
      </div>
      <p className="viz-caption">
        Слой i (с нуля) — full, если (i + 1) делится на период: у Gemma 3 — 6, у Qwen3-Next — 4, поэтому full-слой
        последний в группе. По 12 слоёв в строке.
        {hover !== null && (
          <>
            {" "}
            Слой {hover}: {KIND_LABEL[kinds[hover]]}
            {kinds[hover] === "full"
              ? " — KV-cache растёт с каждым токеном."
              : kinds[hover] === "sliding"
                ? " — кеш ограничен окном."
                : " — состояние фиксированного размера."}
          </>
        )}
      </p>

      <h4 className="viz-subtitle">Память инференса: одна и та же «геометрия» Qwen3-Next, три расписания слоёв</h4>
      <div className="viz-controls">
        <label>
          контекст = {fmtCtx(ctx)}
          <input type="range" min={0} max={CTX.length - 1} value={ctxIdx} onChange={(e) => setCtxIdx(+e.target.value)} />
        </label>
        <label>
          окно SWA = {W}
          <input type="range" min={0} max={windows.length - 1} value={wIdx} onChange={(e) => setWIdx(+e.target.value)} />
        </label>
      </div>
      <LineChart
        xs={CTX}
        series={series}
        logX
        xTicks={CTX.filter((_, i) => i % 2 === 0)}
        xFormat={fmtCtx}
        yFormat={(y) => (y >= 1 || y === 0 ? `${+y.toFixed(1)} ГиБ` : `${+(y * 1024).toFixed(0)} МиБ`)}
        yLabel="кеш"
        xLabel="контекст"
        markerX={ctx}
      />
      <p className="viz-caption">
        48 слоёв, у full-слоя 2 KV-головы × 256 ({KV_PER_TOKEN} чисел на токен), у Gated DeltaNet-слоя состояние 32 × 128 × 128 ={" "}
        {STATE.toLocaleString("ru")} чисел, bf16, batch 1. На {fmtCtx(ctx)}: все full — {fmtBytes(memAllFull(ctx))}, SWA 5 : 1 —{" "}
        {fmtBytes(memSwa(ctx, W))}, гибрид 3 : 1 — {fmtBytes(memHybrid(ctx))} ({((memHybrid(ctx) / memAllFull(ctx)) * 100).toFixed(0)} % от
        «все full»). Гибрид дороже на коротком контексте (состояния «весят» как KV-cache на 512 токенов) и стремится к ¼ на
        длинном. SWA в пределе ещё дешевле (⅙), но sliding-слои вообще не видят дальше окна, а Gated DeltaNet хранит сжатую
        память обо всём прошлом. Чистый линейный стек — константа ({fmtBytes(L * STATE * BPE)}), но хуже извлекает факты.
      </p>
    </div>
  );
}
