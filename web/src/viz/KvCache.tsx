// V-kvcache: калькулятор размера KV-cache для MHA / GQA / MQA с пресетами реальных моделей.
// Числа пресетов — из config.json на Hugging Face (ссылка у каждого пресета).
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
}

// head_dim = hidden_size / num_attention_heads, если head_dim не задан явно.
const PRESETS: Preset[] = [
  { id: "gpt2", name: "GPT-2 small", layers: 12, heads: 12, kvHeads: 12, headDim: 64, ctx: 1024, source: "openai-community/gpt2" },
  { id: "llama2-7b", name: "Llama 2 7B", layers: 32, heads: 32, kvHeads: 32, headDim: 128, ctx: 4096, source: "meta-llama/Llama-2-7b-hf" },
  { id: "llama3-8b", name: "Llama 3 8B", layers: 32, heads: 32, kvHeads: 8, headDim: 128, ctx: 8192, source: "meta-llama/Meta-Llama-3-8B" },
  { id: "llama3-70b", name: "Llama 3 70B", layers: 80, heads: 64, kvHeads: 8, headDim: 128, ctx: 8192, source: "meta-llama/Meta-Llama-3-70B" },
  { id: "falcon-7b", name: "Falcon-7B (MQA)", layers: 32, heads: 71, kvHeads: 1, headDim: 64, ctx: 2048, source: "tiiuae/falcon-7b" },
  { id: "qwen3-0.6b", name: "Qwen3-0.6B", layers: 28, heads: 16, kvHeads: 8, headDim: 128, ctx: 40960, source: "Qwen/Qwen3-0.6B" },
];

const DTYPES: [string, number][] = [
  ["fp32", 4],
  ["bf16 / fp16", 2],
  ["fp8 / int8", 1],
];

const CTX_STEPS = Array.from({ length: 9 }, (_, i) => 512 * 2 ** i); // 512 … 131 072

function bytes(p: Preset, kvHeads: number, seq: number, batch: number, bpe: number) {
  return 2 * p.layers * kvHeads * p.headDim * seq * batch * bpe;
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
  const p = PRESETS.find((x) => x.id === pid)!;
  const ctx = CTX_STEPS[ctxIdx];

  // Три режима для тех же слоёв и head_dim: у каждой головы свои K/V, как у модели, одна общая.
  const variants = useMemo(() => {
    const v: { id: string; label: string; kv: number }[] = [{ id: "mha", label: `MHA (${p.heads})`, kv: p.heads }];
    if (p.kvHeads !== p.heads && p.kvHeads !== 1) v.push({ id: "gqa", label: `GQA (${p.kvHeads})`, kv: p.kvHeads });
    v.push({ id: "mqa", label: "MQA (1)", kv: 1 });
    return v;
  }, [p]);
  const own = variants.find((v) => v.kv === p.kvHeads)!;

  const series: Series[] = variants.map((v, i) => ({
    id: v.id,
    label: v.id === own.id ? `${v.label} ★` : v.label,
    color: `var(--series-${i + 1})`,
    ys: CTX_STEPS.map((s) => bytes(p, v.kv, s, batch, bpe) / 2 ** 30),
  }));

  const perToken = bytes(p, p.kvHeads, 1, 1, bpe);
  const total = bytes(p, p.kvHeads, ctx, batch, bpe);

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
        {p.layers} слоёв · {p.heads} голов запросов · <b>{p.kvHeads} KV-голов</b> · head_dim {p.headDim} · обучен на{" "}
        {fmtCtx(p.ctx)} ·{" "}
        <a href={`https://huggingface.co/${p.source}/blob/main/config.json`} target="_blank" rel="noreferrer">
          config.json
        </a>
      </div>

      <div className="kv-tiles">
        <div>
          <span className="muted">на один токен</span>
          <b>{fmtBytes(perToken)}</b>
        </div>
        <div>
          <span className="muted">
            {fmtCtx(ctx)} токенов × batch {batch}
          </span>
          <b>{fmtBytes(total)}</b>
        </div>
        {variants
          .filter((v) => v.id !== own.id)
          .map((v) => {
            const b = bytes(p, v.kv, ctx, batch, bpe);
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
        Sliding window и сжатие кеша (MLA) здесь не учтены — они появятся в модулях 2 и 3.
      </p>
    </div>
  );
}
