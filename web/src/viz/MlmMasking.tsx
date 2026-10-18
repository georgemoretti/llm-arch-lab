// V-mlm-masking: маскирование BERT на тексте (игрушечный WordPiece) — какие токены выбраны и чем заменены,
// whole word masking, и доли 80 / 10 / 10 на большой выборке. Всё считается в браузере.
import { useMemo, useState } from "react";
import { Bars, BarRow, Slider, rng } from "./common";
import { TOY_VOCAB, tokenizeWithWords } from "./wordpieceToy";

const TEXT =
  "Модель учится восстанавливать закрытые слова по контексту с обеих сторон: предобучение токенизатора, " +
  "unaffable tokenization и transformers.";

const VOCAB_LIST = Array.from(TOY_VOCAB).filter((t) => !t.startsWith("["));

type Kind = "keep" | "mask" | "random" | "special";

interface Cell {
  orig: string;
  shown: string;
  chosen: boolean;
  kind: Kind;
}

function mask(tokens: string[], wordIds: number[], prob: number, wholeWord: boolean, bugSpecials: boolean, seed: number): Cell[] {
  const next = rng(seed);
  const special = (t: string) => t === "[CLS]" || t === "[SEP]";
  const wordPick = new Map<number, boolean>();
  return tokens.map((t, i) => {
    let chosen: boolean;
    if (wholeWord && wordIds[i] >= 0) {
      if (!wordPick.has(wordIds[i])) wordPick.set(wordIds[i], next() < prob);
      chosen = wordPick.get(wordIds[i])!;
    } else chosen = next() < prob;
    if (special(t) && !bugSpecials) chosen = false;
    const u = next();
    const rnd = VOCAB_LIST[Math.floor(next() * VOCAB_LIST.length)];
    if (!chosen) return { orig: t, shown: t, chosen, kind: special(t) ? "special" : "keep" };
    if (u < 0.8) return { orig: t, shown: "[MASK]", chosen, kind: "mask" };
    if (u < 0.9) return { orig: t, shown: rnd, chosen, kind: "random" };
    return { orig: t, shown: t, chosen, kind: "keep" };
  });
}

const STYLE: Record<Kind, React.CSSProperties> = {
  keep: {},
  special: { color: "var(--muted)" },
  mask: { background: "color-mix(in srgb, var(--accent) 30%, transparent)" },
  random: { background: "color-mix(in srgb, var(--series-2) 30%, transparent)" },
};

export default function MlmMasking() {
  const [prob, setProb] = useState(0.15);
  const [whole, setWhole] = useState(false);
  const [bug, setBug] = useState(false);
  const [seed, setSeed] = useState(3);

  const { tokens, wordIds } = useMemo(() => {
    const r = tokenizeWithWords(TEXT);
    return { tokens: ["[CLS]", ...r.tokens, "[SEP]"], wordIds: [-1, ...r.wordIds, -1] };
  }, []);
  const cells = useMemo(() => mask(tokens, wordIds, prob, whole, bug, seed), [tokens, wordIds, prob, whole, bug, seed]);

  // доли на большой выборке: 200 прогонов того же текста
  const stats = useMemo(() => {
    let n = 0, chosen = 0, m = 0, r = 0, k = 0, specials = 0, split = 0, words = 0;
    for (let s = 0; s < 200; s++) {
      const c = mask(tokens, wordIds, prob, whole, bug, 1000 + s);
      const byWord = new Map<number, boolean[]>();
      c.forEach((x, i) => {
        if (x.orig === "[CLS]" || x.orig === "[SEP]") {
          if (x.chosen) specials++;
          return;
        }
        n++;
        if (!byWord.has(wordIds[i])) byWord.set(wordIds[i], []);
        byWord.get(wordIds[i])!.push(x.chosen);
        if (!x.chosen) return;
        chosen++;
        if (x.kind === "mask") m++;
        else if (x.kind === "random") r++;
        else k++;
      });
      for (const v of byWord.values()) {
        words++;
        if (v.some(Boolean) && !v.every(Boolean)) split++;
      }
    }
    return { rate: chosen / n, m: m / chosen, r: r / chosen, k: k / chosen, specials, split: split / words };
  }, [tokens, wordIds, prob, whole, bug]);

  const nChosen = cells.filter((c) => c.chosen).length;

  return (
    <div>
      <div className="viz-controls" style={{ flexWrap: "wrap" }}>
        <Slider label={<>prob = {prob.toFixed(2)}</>} value={prob} min={0.05} max={0.5} step={0.01} onChange={setProb} />
        <label className="check">
          <input type="checkbox" checked={whole} onChange={(e) => setWhole(e.target.checked)} /> whole word masking
        </label>
        <label className="check">
          <input type="checkbox" checked={bug} onChange={(e) => setBug(e.target.checked)} /> ошибка: спецтокены тоже выбираются
        </label>
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новая маска
        </button>
      </div>
      <div className="viz-subtitle">вход модели (сверху) и labels (снизу: исходный токен у выбранных, · у остальных = −100)</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {cells.map((c, i) => (
          <div
            key={i}
            className="mono"
            style={{
              display: "grid",
              fontSize: 12,
              textAlign: "center",
              border: `1px solid ${c.chosen ? "var(--accent)" : "var(--border)"}`,
              borderRadius: 5,
              overflow: "hidden",
            }}
            title={c.chosen ? `выбран: ${c.kind === "mask" ? "[MASK]" : c.kind === "random" ? "случайный токен" : "оставлен как есть"}` : "не выбран"}
          >
            <span style={{ padding: "1px 5px", ...STYLE[c.kind] }}>{c.shown}</span>
            <span style={{ padding: "0 5px", fontSize: 11, color: c.chosen ? "var(--text)" : "var(--muted)", borderTop: "1px dashed var(--border)" }}>
              {c.chosen ? c.orig : "·"}
            </span>
          </div>
        ))}
      </div>
      <p className="viz-caption" style={{ marginTop: 6 }}>
        В этой маске выбрано {nChosen} из {cells.length} токенов. Синий — [MASK], оранжевый — случайный токен, рамка без
        заливки — выбран, но оставлен как есть (модель всё равно должна его «восстановить»).
      </p>
      <div className="viz-subtitle">200 масок того же текста</div>
      <BarRow title="из выбранных" stat={`${(stats.rate * 100).toFixed(1)} % токенов выбрано`}>
        <Bars
          values={[stats.m, stats.r, stats.k]}
          max={1}
          labels={["[MASK]", "случайный", "как есть"]}
          colors={["var(--accent)", "var(--series-2)", "var(--muted)"]}
          titles={[stats.m, stats.r, stats.k].map((v) => `${(v * 100).toFixed(1)} %`)}
          barWidth={56}
        />
      </BarRow>
      <div className="kv-tiles">
        <div>
          <span className="muted">[CLS] / [SEP] в labels</span>
          <b style={stats.specials ? { color: "var(--bad)" } : undefined}>{stats.specials}</b>
        </div>
        <div>
          <span className="muted">слов, выбранных частично</span>
          <b>{(stats.split * 100).toFixed(1)} %</b>
          <span className="muted">{whole ? "целые слова: 0" : "токены выбираются независимо"}</span>
        </div>
      </div>
      <p className="viz-caption">
        Токенизация — игрушечный WordPiece из V-wordpiece. Без whole word masking часть слов закрыта наполовину: «token
        [MASK] ##ization» легко угадать по соседним кускам. Включите ошибку — в labels появятся служебные токены, и часть
        лосса уйдёт на их тривиальное предсказание.
      </p>
    </div>
  );
}
