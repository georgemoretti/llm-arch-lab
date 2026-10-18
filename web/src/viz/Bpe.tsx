// V-bpe: byte-level BPE обучается в браузере на введённом тексте (с пре-токенизацией GPT-2 или без неё).
// Ползунок — число слияний: текст с подсветкой токенов, последние выученные слияния и кривые
// «размер словаря → число токенов» для обоих режимов. Алгоритм — тот же, что в A11–A13.
import { useMemo, useState } from "react";
import { LineChart } from "./LineChart";
import { Slider } from "./common";

const DEFAULT = `Токенизатор режет текст на куски, которые модель видит как отдельные токены. Сначала каждый байт — свой токен, потом самые частые пары склеиваются: буквы из двух байтов, частые слоги, целые слова с пробелом в начале. Чем больше слияний, тем короче последовательность и тем больше словарь. Emoji 🙂 и редкие символы всё равно кодируются: в худшем случае по байтам.`;
const MAX_MERGES = 400;
// та же регулярка, что GPT2_PATTERN в A13 (в JS классы Unicode — с флагом u)
const GPT2_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

type Pair = [number, number];
const utf8 = new TextEncoder();
const strict = new TextDecoder("utf-8", { fatal: true });

function mergePair(seq: number[], a: number, b: number, id: number) {
  const out: number[] = [];
  for (let i = 0; i < seq.length; i++) {
    if (i + 1 < seq.length && seq[i] === a && seq[i + 1] === b) {
      out.push(id);
      i++;
    } else out.push(seq[i]);
  }
  return out;
}

interface Run {
  merges: Pair[];
  vocab: Uint8Array[];
  lengths: number[]; // число токенов после k слияний, k = 0..merges.length
}

function train(chunks: number[][], maxMerges: number): Run {
  let seqs = chunks.map((c) => c.slice());
  const merges: Pair[] = [];
  const vocab = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i));
  const lengths = [seqs.reduce((s, q) => s + q.length, 0)];
  for (let k = 0; k < maxMerges; k++) {
    const counts = new Map<number, number>();
    for (const s of seqs) for (let i = 0; i + 1 < s.length; i++) {
      const key = s[i] * 65536 + s[i + 1];
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    if (counts.size === 0) break;
    let best = -1;
    let bestCount = 0;
    for (const [key, c] of counts) if (c > bestCount || (c === bestCount && key < best)) [best, bestCount] = [key, c];
    const a = Math.floor(best / 65536);
    const b = best % 65536;
    const id = 256 + k;
    merges.push([a, b]);
    const tok = new Uint8Array(vocab[a].length + vocab[b].length);
    tok.set(vocab[a]);
    tok.set(vocab[b], vocab[a].length);
    vocab.push(tok);
    seqs = seqs.map((s) => mergePair(s, a, b, id));
    lengths.push(seqs.reduce((s, q) => s + q.length, 0));
  }
  return { merges, vocab, lengths };
}

/** Токен как текст; половинки символов UTF-8 — шестнадцатеричными байтами ‹D0›. */
function show(bytes: Uint8Array) {
  try {
    return strict.decode(bytes).replace(/ /g, "␣").replace(/\n/g, "↵");
  } catch {
    // склеиваем то, что декодируется, остальное — байтами
    let out = "";
    let i = 0;
    while (i < bytes.length) {
      let done = false;
      for (let n = Math.min(4, bytes.length - i); n >= 1 && !done; n--) {
        try {
          out += strict.decode(bytes.subarray(i, i + n)).replace(/ /g, "␣").replace(/\n/g, "↵");
          i += n;
          done = true;
        } catch {
          /* пробуем короче */
        }
      }
      if (!done) out += `‹${bytes[i++].toString(16).toUpperCase().padStart(2, "0")}›`;
    }
    return out;
  }
}

export default function Bpe() {
  const [text, setText] = useState(DEFAULT);
  const [regex, setRegex] = useState(true);
  const [k, setK] = useState(120);

  const runs = useMemo(() => {
    const bytes = Array.from(utf8.encode(text));
    const chunks = (text.match(GPT2_RE) ?? []).map((c) => Array.from(utf8.encode(c)));
    return { plain: train([bytes], MAX_MERGES), regex: train(chunks, MAX_MERGES), nBytes: bytes.length, chunks };
  }, [text]);

  const run = regex ? runs.regex : runs.plain;
  const steps = run.merges.length;
  const kk = Math.min(k, steps);

  // разбиение текста после kk слияний (по кускам или целиком — как при обучении)
  const tokens = useMemo(() => {
    let seqs = regex ? runs.chunks.map((c) => c.slice()) : [Array.from(utf8.encode(text))];
    run.merges.slice(0, kk).forEach(([a, b], i) => {
      seqs = seqs.map((s) => mergePair(s, a, b, 256 + i));
    });
    return seqs.flat();
  }, [run, runs, kk, regex, text]);

  const n = Math.max(runs.plain.lengths.length, runs.regex.lengths.length);
  const xs = Array.from({ length: n }, (_, i) => 256 + i);
  const pad = (ys: number[]) => xs.map((_, i) => ys[Math.min(i, ys.length - 1)]);
  const recent = run.merges.slice(Math.max(0, kk - 10), kk).map((m, j) => ({ m, id: Math.max(0, kk - 10) + j + 256 })).reverse();

  return (
    <div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} style={{ width: "100%", font: "inherit", fontSize: 13 }} aria-label="Текст для обучения BPE" />
      <div className="viz-controls">
        <Slider label={<>слияний: {kk}</>} value={kk} min={0} max={Math.max(1, steps)} onChange={setK} width={220} />
        <label className="check">
          <input type="checkbox" checked={regex} onChange={(e) => setRegex(e.target.checked)} /> пре-токенизация GPT-2
        </label>
      </div>
      <div className="kv-tiles">
        <div>
          <span className="muted">байтов UTF-8</span>
          <b>{runs.nBytes}</b>
        </div>
        <div>
          <span className="muted">словарь</span>
          <b>{256 + kk}</b>
        </div>
        <div>
          <span className="muted">токенов</span>
          <b>{tokens.length}</b>
        </div>
        <div>
          <span className="muted">байтов на токен</span>
          <b>{(runs.nBytes / Math.max(1, tokens.length)).toFixed(2)}</b>
        </div>
      </div>
      <div className="bpe-text mono" aria-label="Текст, разбитый на токены">
        {tokens.map((t, i) => (
          <span key={i} className={`bpe-tok bpe-tok-${i % 3}${kk > 0 && t === 255 + kk ? " bpe-new" : ""}`} title={`id ${t}`}>
            {show(run.vocab[t])}
          </span>
        ))}
      </div>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start", marginTop: 8 }}>
        <div style={{ flex: "1 1 520px", minWidth: 0 }}>
          <LineChart
            xs={xs}
            series={[
              { id: "plain", label: "без регулярки", color: "var(--series-1)", ys: pad(runs.plain.lengths) },
              { id: "regex", label: "с регуляркой", color: "var(--series-2)", ys: pad(runs.regex.lengths) },
            ]}
            xTicks={[256, 256 + Math.round(n / 4), 256 + Math.round(n / 2), 256 + Math.round((3 * n) / 4), 255 + n]}
            xFormat={(v) => String(Math.round(v))}
            yFormat={(v) => String(Math.round(v))}
            yLabel="токенов в тексте"
            xLabel="размер словаря (256 + слияний)"
            markerX={256 + kk}
            height={200}
          />
        </div>
        <table className="bpe-merges mono">
          <thead>
            <tr>
              <th>id</th>
              <th>пара</th>
              <th>токен</th>
            </tr>
          </thead>
          <tbody>
            {recent.map(({ m: [a, b], id }) => (
              <tr key={id}>
                <td>{id}</td>
                <td>
                  {show(run.vocab[a])} + {show(run.vocab[b])}
                </td>
                <td>
                  <b>{show(run.vocab[id])}</b>
                </td>
              </tr>
            ))}
            {recent.length === 0 && (
              <tr>
                <td colSpan={3} className="muted">
                  пока только байты
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="viz-caption">
        Цветные плашки — токены после выбранного числа слияний (наведите, чтобы увидеть id), ␣ — пробел, ‹D0› — байт,
        который пока не собран в символ: кириллическая буква занимает в UTF-8 два байта. Таблица справа — последние
        выученные слияния, новое — сверху. Без пре-токенизации слияния идут через пробелы и пунктуацию: появляются токены
        вроде «а␣» или «,␣т». С регуляркой GPT-2 пробел прилипает только к началу слова, и токены похожи на слова и их
        части. На графике видно цену этого ограничения: на обучающем тексте последовательность чуть длиннее, зато
        словарь не тратится на случайные сочетания соседей. Вставьте свой текст, например английский или код, и
        сравните, что выучится первым.
      </p>
    </div>
  );
}
