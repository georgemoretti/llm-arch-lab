// V-param-budget: куда уходят параметры Qwen3-0.6B — посчитано из config.json, как в задаче про чекпойнт.
import { useState } from "react";

// Qwen/Qwen3-0.6B config.json: vocab_size 151936, hidden_size 1024, num_hidden_layers 28,
// num_attention_heads 16, num_key_value_heads 8, head_dim 128, intermediate_size 3072, tie_word_embeddings true.
const CFG = { V: 151936, d: 1024, L: 28, H: 16, Hkv: 8, hd: 128, F: 3072 };

const PARTS = (() => {
  const { V, d, L, H, Hkv, hd, F } = CFG;
  const attn = L * (d * H * hd + 2 * d * Hkv * hd + H * hd * d);
  const mlp = L * 3 * d * F;
  const norms = L * (2 * d + 2 * hd) + d; // input/post_attention_layernorm, q_norm/k_norm, финальная norm
  return [
    { id: "embed", label: "эмбеддинги (= lm_head)", n: V * d, color: "var(--series-2)" },
    { id: "attn", label: "attention: q/k/v/o_proj", n: attn, color: "var(--series-1)" },
    { id: "mlp", label: "MLP: gate/up/down_proj", n: mlp, color: "var(--series-3)" },
    { id: "norm", label: "нормы", n: norms, color: "var(--ax-batch)" },
  ];
})();

const M = (n: number) => `${(n / 1e6).toFixed(1)} M`;

export default function ParamBudget() {
  const [tied, setTied] = useState(true);
  const parts = tied ? PARTS : [...PARTS, { id: "head", label: "отдельный lm_head", n: CFG.V * CFG.d, color: "var(--ax-wide)" }];
  const total = parts.reduce((s, p) => s + p.n, 0);

  return (
    <div>
      <div className="viz-controls">
        <label className="check">
          <input type="checkbox" checked={tied} onChange={(e) => setTied(e.target.checked)} /> lm_head = embed_tokens (tie_word_embeddings)
        </label>
      </div>
      <div className="budget-bar" role="img" aria-label="Доли параметров">
        {parts.map((p) => (
          <div key={p.id} style={{ flexGrow: p.n, background: p.color }} title={`${p.label}: ${p.n.toLocaleString("ru-RU")}`} />
        ))}
      </div>
      <table className="budget-table">
        <tbody>
          {parts.map((p) => (
            <tr key={p.id}>
              <td>
                <span className="budget-dot" style={{ background: p.color }} /> {p.label}
              </td>
              <td className="mono">{p.n.toLocaleString("ru-RU")}</td>
              <td className="mono muted">{((100 * p.n) / total).toFixed(p.n / total < 0.01 ? 2 : 1)}%</td>
            </tr>
          ))}
          <tr className="budget-total">
            <td>всего</td>
            <td className="mono">{total.toLocaleString("ru-RU")}</td>
            <td className="mono muted">
              fp32 ≈ {((total * 4) / 1e9).toFixed(2)} ГБ · bf16 ≈ {((total * 2) / 1e9).toFixed(2)} ГБ
            </td>
          </tr>
        </tbody>
      </table>
      <p className="viz-caption">
        Qwen3-0.6B: {CFG.L} слоёв, d = {CFG.d}, {CFG.H} / {CFG.Hkv} голов × {CFG.hd}, FFN {CFG.F}, словарь {CFG.V.toLocaleString("ru-RU")}.
        У маленькой модели с большим словарём четверть весов — таблица эмбеддингов ({M(CFG.V * CFG.d)}), поэтому её и делят с
        lm_head. Внутри слоя MLP весит в 1.5 раза больше attention: GQA и узкий FFN 3d. При загрузке это сходится до
        параметра: {total.toLocaleString("ru-RU")}{tied ? " — то же число, что в задаче-проекте" : ""}.
      </p>
    </div>
  );
}
