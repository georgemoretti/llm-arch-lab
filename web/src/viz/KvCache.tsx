// V-kvcache: калькулятор размера KV-cache для MHA / GQA / MQA с пресетами реальных моделей,
// с учётом sliding window (у sliding-слоёв кеш не больше окна — C11).
// Числа пресетов — из config.json на Hugging Face (ссылка у каждого пресета; конфиги Gemma
// закрыты лицензией и сверены по открытым зеркалам unsloth/*).
import { useMemo, useState } from "react";
import { LineChart, type Series } from "./LineChart";

interface Preset {
  id: string;
  name: string;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  ctx: number; // max_position_embeddings / n_positions
  source: string;
  window?: number; // sliding_window
  pattern?: number; // каждый pattern-й слой (с 1) — full; 0 — все слои sliding
  // MLA (C16b): на токен и слой в кеше kv_lora_rank + qk_rope_head_dim чисел. Чтобы формула
  // 2·kvHeads·headDim осталась общей, kvHeads у такого пресета — «эквивалент» (r + d_r) / (2·headDim).
  mla?: { rank: number; rope: number };
}

// head_dim = hidden_size / num_attention_heads, если head_dim не задан явно.
const PRESETS: Preset[] = [
  { id: "gpt2", name: "GPT-2 small", layers: 12, heads: 12, kvHeads: 12, headDim: 64, ctx: 1024, source: "openai-community/gpt2" },
  { id: "llama2-7b", name: "Llama 2 7B", layers: 32, heads: 32, kvHeads: 32, headDim: 128, ctx: 4096, source: "meta-llama/Llama-2-7b-hf" },
  { id: "llama3-8b", name: "Llama 3 8B", layers: 32, heads: 32, kvHeads: 8, headDim: 128, ctx: 8192, source: "meta-llama/Meta-Llama-3-8B" },
  { id: "llama3-70b", name: "Llama 3 70B", layers: 80, heads: 64, kvHeads: 8, headDim: 128, ctx: 8192, source: "meta-llama/Meta-Llama-3-70B" },
  { id: "falcon-7b", name: "Falcon-7B (MQA)", layers: 32, heads: 71, kvHeads: 1, headDim: 64, ctx: 2048, source: "tiiuae/falcon-7b" },
  { id: "qwen3-0.6b", name: "Qwen3-0.6B", layers: 28, heads: 16, kvHeads: 8, headDim: 128, ctx: 40960, source: "Qwen/Qwen3-0.6B" },
  { id: "mistral-7b", name: "Mistral 7B v0.1 (SWA)", layers: 32, heads: 32, kvHeads: 8, headDim: 128, ctx: 32768, source: "mistralai/Mistral-7B-v0.1", window: 4096, pattern: 0 },
  { id: "gemma2-9b", name: "Gemma 2 9B (1:1)", layers: 42, heads: 16, kvHeads: 8, headDim: 256, ctx: 8192, source: "google/gemma-2-9b", window: 4096, pattern: 2 },
  { id: "gemma3-4b", name: "Gemma 3 4B (5:1)", layers: 34, heads: 8, kvHeads: 4, headDim: 256, ctx: 131072, source: "google/gemma-3-4b-pt", window: 1024, pattern: 6 },
  // DeepSeek-V3: MHA-эквивалент — ключ 128 + 64, значение 128 на голову, то есть в среднем headDim = 160.
  { id: "deepseek-v3", name: "DeepSeek-V3 (MLA)", layers: 61, heads: 128, kvHeads: 576 / 320, headDim: 160, ctx: 163840, source: "deepseek-ai/DeepSeek-V3", mla: { rank: 512, rope: 64 } },
  { id: "gemma3-27b", name: "Gemma 3 27B (5:1)", layers: 62, heads: 32, kvHeads: 16, headDim: 128, ctx: 131072, source: "google/gemma-3-27b-pt", window: 1024, pattern: 6 },
];

const DTYPES: [string, number][] = [
  ["fp32", 4],
  ["bf16 / fp16", 2],
  ["fp8 / int8", 1],
];

const CTX_STEPS = Array.from({ length: 9 }, (_, i) => 512 * 2 ** i); // 512 … 131 072

// Число full- и sliding-слоёв по правилу Gemma: слой i — full, если (i + 1) % pattern == 0.
function layerSplit(p: Preset, swa: boolean) {
  if (!swa || !p.window) return { full: p.layers, sliding: 0 };
  const full = p.pattern ? Math.floor(p.layers / p.pattern) : 0;
  return { full, sliding: p.layers - full };
}

function bytes(p: Preset, kvHeads: number, seq: number, batch: number, bpe: number, swa = false) {
  const { full, sliding } = layerSplit(p, swa);
  const tokens = full * seq + sliding * Math.min(seq, p.window ?? seq);
  return 2 * tokens * kvHeads * p.headDim * batch * bpe;
}

export function fmtBytes(b: number): string {
  const units = ["Б", "КиБ", "МиБ", "ГиБ", "ТиБ"];
  let i = 0;
  while (b >= 1024 && i < units.length - 1) {
    b /= 1024;
    i++;
  }
  return `${b >= 100 || i === 0 ? b.toFixed(0) : b >= 10 ? b.toFixed(1) : b.toFixed(2)} ${units[i]}`;
}

const fmtCtx = (x: number) => (x >= 1024 ? `${x / 1024}K` : String(x));

export default function KvCache() {
  const [pid, setPid] = useState("llama3-8b");
  const [ctxIdx, setCtxIdx] = useState(4); // 8K
  const [batch, setBatch] = useState(1);
  const [bpe, setBpe] = useState(2);
  const [swaOn, setSwaOn] = useState(true);
  const p = PRESETS.find((x) => x.id === pid)!;
  const swa = swaOn && !!p.window;
  const split = layerSplit(p, true);
  const ctx = CTX_STEPS[ctxIdx];

  // Три режима для тех же слоёв и head_dim: у каждой головы свои K/V, как у модели, одна общая.
  const variants = useMemo(() => {
    const v: { id: string; label: string; kv: number }[] = [{ id: "mha", label: `MHA (${p.heads})`, kv: p.heads }];
    if (p.mla) v.push({ id: "mla", label: `MLA (${p.mla.rank} + ${p.mla.rope})`, kv: p.kvHeads });
    else if (p.kvHeads !== p.heads && p.kvHeads !== 1) v.push({ id: "gqa", label: `GQA (${p.kvHeads})`, kv: p.kvHeads });
    if (!p.mla) v.push({ id: "mqa", label: "MQA (1)", kv: 1 }); // у MLA сравниваем с MHA тех же голов
    return v;
  }, [p]);
  const own = variants.find((v) => v.kv === p.kvHeads)!;

  const series: Series[] = variants.map((v, i) => ({
    id: v.id,
    label: v.id === own.id ? `${v.label} ★` : v.label,
    color: `var(--series-${i + 1})`,
    ys: CTX_STEPS.map((s) => bytes(p, v.kv, s, batch, bpe, swa) / 2 ** 30),
  }));

  const perToken = bytes(p, p.kvHeads, 1, 1, bpe);
  const total = bytes(p, p.kvHeads, ctx, batch, bpe, swa);
  const noWindow = bytes(p, p.kvHeads, ctx, batch, bpe, false);

  return (
    <div className="viz-kv">
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
        <label>
          контекст = {fmtCtx(ctx)}
          <input type="range" min={0} max={CTX_STEPS.length - 1} value={ctxIdx} onChange={(e) => setCtxIdx(+e.target.value)} />
        </label>
        <label>
          batch = {batch}
          <input type="range" min={1} max={64} value={batch} onChange={(e) => setBatch(+e.target.value)} />
        </label>
        {p.window && (
          <label className="check">
            <input type="checkbox" checked={swaOn} onChange={(e) => setSwaOn(e.target.checked)} /> учитывать окно SWA
          </label>
        )}
        <label className="check">
          тип
          <select value={bpe} onChange={(e) => setBpe(+e.target.value)}>
            {DTYPES.map(([name, b]) => (
              <option key={b} value={b}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="kv-config mono">
        {p.layers} слоёв · {p.heads} голов запросов ·{" "}
        {p.mla ? (
          <b>
            MLA: латент {p.mla.rank} + RoPE-ключ {p.mla.rope} на токен
          </b>
        ) : (
          <>
            <b>{p.kvHeads} KV-голов</b> · head_dim {p.headDim}
          </>
        )}{" "}
        · обучен на{" "}
        {fmtCtx(p.ctx)} ·{" "}
        {p.window && (
          <>
            <b>
              {split.sliding} sliding (окно {fmtCtx(p.window)}) + {split.full} full
            </b>{" "}
            ·{" "}
          </>
        )}
        <a href={`https://huggingface.co/${p.source}/blob/main/config.json`} target="_blank" rel="noreferrer">
          config.json
        </a>
      </div>

      <div className="kv-tiles">
        <div>
          <span className="muted">на один токен{swa ? " (до заполнения окна)" : ""}</span>
          <b>{fmtBytes(perToken)}</b>
        </div>
        <div>
          <span className="muted">
            {fmtCtx(ctx)} токенов × batch {batch}
          </span>
          <b>{fmtBytes(total)}</b>
        </div>
        {swa && (
          <div>
            <span className="muted">без окна (все слои full)</span>
            <b>{fmtBytes(noWindow)}</b>
            <span className="muted">{noWindow > total ? `×${(noWindow / total).toFixed(1)}` : "столько же"}</span>
          </div>
        )}
        {variants
          .filter((v) => v.id !== own.id)
          .map((v) => {
            const b = bytes(p, v.kv, ctx, batch, bpe, swa);
            return (
              <div key={v.id}>
                <span className="muted">если бы {v.label}</span>
                <b>{fmtBytes(b)}</b>
                <span className="muted">{b > total ? `×${(b / total).toFixed(0)}` : `÷${(total / b).toFixed(0)}`}</span>
              </div>
            );
          })}
      </div>

      <LineChart
        xs={CTX_STEPS}
        series={series}
        logX
        xTicks={CTX_STEPS.filter((_, i) => i % 2 === 0)}
        xFormat={fmtCtx}
        yFormat={(y) => (y >= 1 || y === 0 ? `${+y.toFixed(1)} ГиБ` : `${+(y * 1024).toFixed(0)} МиБ`)}
        yLabel="KV-cache"
        xLabel="контекст"
        markerX={ctx}
      />

      <p className="viz-caption">
        Размер = 2 (K и V) × слои × KV-головы × head_dim × токены × batch × байт на число. Он линеен по длине
        контекста, поэтому на длинных контекстах кеш легко обгоняет сами веса. ★ — конфигурация выбранной
        модели; остальные линии — те же слои и head_dim, но с другим числом KV-голов. Шкала по x логарифмическая.
        У моделей со sliding window кеш sliding-слоя не больше окна (кольцевой буфер, C11): у Mistral 7B кривая
        выходит на плато после 4K, у Gemma 3 растут только full-слои. У DeepSeek-V3 (MLA) в кеше латент и общий RoPE-ключ — 576 чисел
        на токен и слой (C16b); «MHA» для неё посчитан с ключом 192 и значением 128 на голову.
      </p>
    </div>
  );
}
