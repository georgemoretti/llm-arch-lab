// V-encdec-masks: три маски энкодер-декодера на одном батче из двух пар разной длины. Самостоятельно считает
// веса softmax(q·k/√D + M) для случайных q, k и показывает, какие ячейки запрещены и почему (PAD или будущее).
import { useMemo, useState } from "react";
import { Segmented, Slider, gaussian, rng, softmax } from "./common";

type Kind = "enc" | "dec" | "cross";

const D = 16;

export default function EncdecMasks() {
  const [kind, setKind] = useState<Kind>("enc");
  const [srcLens, setSrcLens] = useState<[number, number]>([4, 7]);
  const [tgtLens, setTgtLens] = useState<[number, number]>([3, 6]);
  const [row, setRow] = useState<0 | 1>(0);
  const [wrongAxisRaw, setWrongAxis] = useState(false);
  const [seed, setSeed] = useState(1);
  const [hover, setHover] = useState<number | null>(null);

  const wrongAxis = wrongAxisRaw && kind === "enc"; // ошибку показываем на self-attention энкодера
  const S = Math.max(...srcLens);
  const T = Math.max(...tgtLens);
  const nq = kind === "enc" ? S : T; // запросы
  const nk = kind === "dec" ? T : S; // ключи
  const qLen = kind === "enc" ? srcLens[row] : tgtLens[row];
  const kLen = kind === "dec" ? tgtLens[row] : srcLens[row];

  // причина запрета ячейки: "pad" — PAD-ключ (или PAD-запрос при ошибке), "future" — causal
  const reason = (t: number, s: number): "pad" | "future" | null => {
    if (wrongAxis) {
      if (t >= qLen) return "pad";
    } else if (s >= kLen) return "pad";
    if (kind === "dec" && s > t) return "future";
    return null;
  };

  const weights = useMemo(() => {
    const next = rng(seed * 31 + row);
    return Array.from({ length: nq }, (_, t) => {
      const logits = Array.from({ length: nk }, (_, s) => {
        let v = 0;
        for (let i = 0; i < D; i++) v += gaussian(next) * gaussian(next);
        return reason(t, s) ? -Infinity : v / Math.sqrt(D);
      });
      return logits.every((v) => v === -Infinity) ? logits.map(() => NaN) : softmax(logits);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, row, nq, nk, qLen, kLen, kind, wrongAxis]);

  const maxW = Math.max(1e-9, ...weights.flat().filter((v) => !Number.isNaN(v)));
  const cell = Math.max(nq, nk) > 8 ? 30 : 36;
  const qName = kind === "enc" ? "src" : "tgt";
  const kName = kind === "dec" ? "tgt" : "src";
  const leaks = !wrongAxis ? 0 : weights.slice(0, qLen).reduce((a, r) => a + r.slice(kLen).reduce((x, y) => x + y, 0), 0);

  const setLen = (which: "src" | "tgt", i: 0 | 1, v: number) =>
    which === "src"
      ? setSrcLens((l) => (i === 0 ? [v, l[1]] : [l[0], v]))
      : setTgtLens((l) => (i === 0 ? [v, l[1]] : [l[0], v]));

  return (
    <div className="viz-causal">
      <div className="viz-controls">
        <Segmented
          value={kind}
          options={[
            ["enc", "self-attention энкодера"],
            ["dec", "self-attention декодера"],
            ["cross", "cross-attention"],
          ]}
          onChange={setKind}
        />
      </div>
      <div className="viz-controls">
        <Slider label={<>источник 1: {srcLens[0]}</>} value={srcLens[0]} min={1} max={9} onChange={(v) => setLen("src", 0, v)} />
        <Slider label={<>источник 2: {srcLens[1]}</>} value={srcLens[1]} min={1} max={9} onChange={(v) => setLen("src", 1, v)} />
        <Slider label={<>цель 1: {tgtLens[0]}</>} value={tgtLens[0]} min={1} max={9} onChange={(v) => setLen("tgt", 0, v)} />
        <Slider label={<>цель 2: {tgtLens[1]}</>} value={tgtLens[1]} min={1} max={9} onChange={(v) => setLen("tgt", 1, v)} />
      </div>
      <div className="viz-controls">
        <Segmented
          value={String(row) as "0" | "1"}
          options={[
            ["0", "пара 1 батча"],
            ["1", "пара 2 батча"],
          ]}
          onChange={(v) => setRow(v === "0" ? 0 : 1)}
        />
        {kind === "enc" && (
          <label className="check">
            <input type="checkbox" checked={wrongAxisRaw} onChange={(e) => setWrongAxis(e.target.checked)} /> ошибка:
            PAD запрещён по оси запросов
          </label>
        )}
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые q, k
        </button>
      </div>
      <div className="viz-matrix" style={{ gridTemplateColumns: `52px repeat(${nk}, ${cell}px) 52px` }}>
        <div />
        {Array.from({ length: nk }, (_, s) => (
          <div key={s} className="viz-axis" style={s >= kLen ? { opacity: 0.45 } : undefined}>
            {s >= kLen ? "PAD" : `${kName}${s}`}
          </div>
        ))}
        <div className="viz-axis">Σ</div>
        {weights.map((r, t) => {
          const padQuery = t >= qLen;
          const active = hover === t;
          const sum = r.reduce((a, b) => a + b, 0);
          return [
            <div key={`a${t}`} className={`viz-axis ${active ? "active" : ""}`} style={padQuery ? { opacity: 0.45 } : undefined}>
              {padQuery ? "PAD" : `${qName}${t}`}
            </div>,
            ...r.map((w, s) => {
              const why = reason(t, s);
              const title = Number.isNaN(w)
                ? "вся строка −inf → softmax 0/0 = nan"
                : why === "pad"
                  ? wrongAxis
                    ? "строка PAD-запроса запрещена целиком"
                    : `ключ ${s} — PAD: вес 0`
                  : why === "future"
                    ? `ключ ${s} > запрос ${t}: будущее, вес 0`
                    : `w[${t},${s}] = ${w.toFixed(4)}${padQuery ? " (выход PAD-позиции потом игнорируют)" : ""}`;
              return (
                <div
                  key={`${t}-${s}`}
                  className={`viz-cell ${why && !Number.isNaN(w) ? "blocked" : ""} ${active ? "active" : ""}`}
                  style={{
                    ...(why || Number.isNaN(w)
                      ? {}
                      : { background: `color-mix(in srgb, var(--accent) ${Math.round((w / maxW) * 85)}%, transparent)` }),
                    ...(why === "future" ? { borderStyle: "dashed" } : {}),
                    ...(Number.isNaN(w) ? { color: "var(--bad)" } : {}),
                    opacity: padQuery && !wrongAxis ? 0.5 : 1,
                  }}
                  title={title}
                  onMouseEnter={() => setHover(t)}
                  onMouseLeave={() => setHover(null)}
                >
                  {Number.isNaN(w) ? "nan" : why ? "0" : w.toFixed(2).replace(/^0/, "")}
                </div>
              );
            }),
            <div key={`z${t}`} className="viz-sum" style={Number.isNaN(sum) ? { color: "var(--bad)" } : undefined}>
              {Number.isNaN(sum) ? "nan" : sum.toFixed(2)}
            </div>,
          ];
        })}
      </div>
      <p className="viz-caption">
        {kind === "enc" &&
          "Энкодер: каждый токен источника видит весь источник, кроме PAD. Маска [B, 1, 1, S] одинакова для всех строк."}
        {kind === "dec" &&
          "Декодер: causal (пунктир — будущее) И padding цели. Маска [B, 1, T, T]. Без causal-части позиция t при teacher forcing увидела бы правильный ответ."}
        {kind === "cross" &&
          "Cross-attention: строки — позиции цели, столбцы — токены источника, матрица T × S. Запрещён только PAD источника: начало перевода имеет право видеть конец фразы."}{" "}
        {wrongAxis
          ? `С ошибкой строки PAD-запросов целиком −inf → nan, а настоящие запросы смотрят на PAD-ключи (их суммарный вес сейчас ${leaks.toFixed(2)}).`
          : kind === "enc"
            ? "Строки PAD-запросов (бледные) считаются, но их выходы потом игнорируют. Включите «ошибку», чтобы увидеть, что будет, если запретить строки, а не столбцы."
            : "Строки PAD-запросов (бледные) считаются, но их выходы потом игнорируют."}
      </p>
    </div>
  );
}
