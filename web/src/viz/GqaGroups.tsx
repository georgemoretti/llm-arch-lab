// V-gqa-groups: какие головы запросов делят одну KV-голову (MHA → GQA → MQA).
import { useState } from "react";
import { plural, Segmented } from "./common";

// num_attention_heads / num_key_value_heads из config.json (те же модели, что в задачах курса).
const PRESETS = {
  toy: { label: "игрушка 8 голов", H: 8, Hkv: 2 },
  llama2: { label: "Llama 2 7B", H: 32, Hkv: 32 },
  llama3: { label: "Llama 3 8B", H: 32, Hkv: 8 },
  qwen3: { label: "Qwen3-0.6B", H: 16, Hkv: 8 },
} as const;
type Preset = keyof typeof PRESETS;

const divisors = (n: number) => Array.from({ length: n }, (_, i) => i + 1).filter((d) => n % d === 0);
const GROUP_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--ax-model)", "var(--ax-wide)", "var(--ax-other)", "var(--ax-batch)", "var(--ax-heads)"];

export default function GqaGroups() {
  const [preset, setPreset] = useState<Preset>("toy");
  const H = PRESETS[preset].H;
  const [Hkv, setHkv] = useState<number>(PRESETS.toy.Hkv);
  const [hover, setHover] = useState<number | null>(null);
  const choose = (p: Preset) => {
    setPreset(p);
    setHkv(PRESETS[p].Hkv);
  };

  const r = H / Hkv;
  const W = 560;
  const qw = Math.min(40, (W - 8) / H - 3);
  const kw = Math.min(64, (W - 8) / Hkv - 6);
  const qx = (i: number) => 4 + i * ((W - 8) / H) + ((W - 8) / H - qw) / 2;
  const kx = (j: number) => 4 + j * ((W - 8) / Hkv) + ((W - 8) / Hkv - kw) / 2;
  const kind = Hkv === H ? "MHA" : Hkv === 1 ? "MQA" : "GQA";
  const color = (j: number) => GROUP_COLORS[j % GROUP_COLORS.length];

  return (
    <div>
      <div className="viz-controls">
        <Segmented value={preset} options={Object.entries(PRESETS).map(([k, v]) => [k as Preset, v.label])} onChange={choose} />
        <label>
          H_kv
          <select value={Hkv} onChange={(e) => setHkv(+e.target.value)}>
            {divisors(H).map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
      </div>
      <svg width={W} height={150} viewBox={`0 0 ${W} 150`} style={{ maxWidth: "100%", height: "auto" }} role="img" aria-label="Связь голов запросов и KV-голов">
        <text x={4} y={12} className="viz-svg-label">
          головы запросов, H = {H}
        </text>
        <text x={4} y={146} className="viz-svg-label">
          KV-головы, H_kv = {Hkv} — только они лежат в KV-cache
        </text>
        {Array.from({ length: H }, (_, i) => {
          const j = Math.floor(i / r);
          const on = hover === null || hover === j;
          return (
            <g key={i} opacity={on ? 1 : 0.2}>
              <line x1={qx(i) + qw / 2} y1={44} x2={kx(j) + kw / 2} y2={106} stroke={color(j)} strokeWidth={1.2} />
              <rect x={qx(i)} y={20} width={qw} height={24} rx={3} fill={color(j)} onMouseEnter={() => setHover(j)} onMouseLeave={() => setHover(null)}>
                <title>{`q-голова ${i} → KV-голова ${j}`}</title>
              </rect>
            </g>
          );
        })}
        {Array.from({ length: Hkv }, (_, j) => (
          <g key={j} opacity={hover === null || hover === j ? 1 : 0.2} onMouseEnter={() => setHover(j)} onMouseLeave={() => setHover(null)}>
            <rect x={kx(j)} y={106} width={kw} height={24} rx={3} fill="var(--panel)" stroke={color(j)} strokeWidth={2} />
            {kw > 18 && (
              <text x={kx(j) + kw / 2} y={122} textAnchor="middle" className="viz-svg-label">
                {j}
              </text>
            )}
            <title>{`KV-голова ${j}: её читают q-головы ${j * r}…${(j + 1) * r - 1}`}</title>
          </g>
        ))}
      </svg>
      <p className="viz-caption">
        <b>{kind}</b>: r = H / H_kv = {r} {plural(r, ["голова", "головы", "голов"])} запросов на одну KV-голову, KV-cache в{" "}
        {r} {plural(r, ["раз", "раза", "раз"])} меньше, чем у MHA. Головы группы идут подряд: q-голова i читает KV-голову ⌊i / r⌋ —
        поэтому repeat_kv повторяет каждую KV-голову r раз подряд. Наведите на группу, чтобы увидеть её целиком.
      </p>
    </div>
  );
}
