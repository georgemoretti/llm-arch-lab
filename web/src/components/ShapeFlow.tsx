// Схема форм тензоров из блока ```shapes в README. Формат (docs/TASK_FORMAT.md §«Схема форм»):
//
//   x [B, T, C] — заметка          тензор: имя, форма (одна или несколько [..]), заметка после « — »
//   > c_attn: Linear C → 3C — …    операция: стрелка вниз с подписью
//   ## forward_absorbed            подзаголовок группы
//   - текст                         свободная строка (пояснение без стрелки)
//   fork / branch / join ⊙ текст    параллельные ветки и их слияние
//   axis r = ранг латента           подпись оси в легенде (переопределяет стандартную);
//   axis N:hdim = размер состояния   …и её цвет: batch | seq | model | heads | hdim | wide | other
//   (пустая строка)                 разрыв: следующий тензор без стрелки
//
// Каждая ось окрашена по смыслу (батч, длина, d_model, головы, размер головы…), наведение на ось
// подсвечивает её во всей схеме — видно, как T или H «проходит» через вычисление.
import { useState, type ReactNode } from "react";

type Item =
  | { kind: "tensor"; name: string; shape: string; note: string }
  | { kind: "op"; text: string; note: string }
  | { kind: "text"; text: string }
  | { kind: "section"; title: string }
  | { kind: "gap" }
  | { kind: "fork"; branches: Item[][]; join: string | null };

const NOTE_SEP = " — ";

function splitNote(s: string): [string, string] {
  const i = s.indexOf(NOTE_SEP);
  return i < 0 ? [s.trim(), ""] : [s.slice(0, i).trim(), s.slice(i + NOTE_SEP.length).trim()];
}

export function parseShapes(source: string): { items: Item[]; axes: Record<string, string>; kinds: Record<string, AxisKind> } {
  const root: Item[] = [];
  const axes: Record<string, string> = {};
  const kinds: Record<string, AxisKind> = {};
  let target = root;
  let fork: Extract<Item, { kind: "fork" }> | null = null;
  for (const raw of source.split("\n")) {
    const line = raw.trim();
    if (line.startsWith("//")) continue;
    if (!line) {
      if (target.length && target[target.length - 1].kind !== "gap") target.push({ kind: "gap" });
      continue;
    }
    const axis = line.match(/^axis\s+([^\s:=]+)(?::(\w+))?\s*=\s*(.+)$/);
    if (axis) {
      axes[axis[1]] = axis[3];
      if (axis[2]) kinds[axis[1]] = axis[2] as AxisKind;
      continue;
    }
    if (line === "fork" || line === "branch") {
      if (line === "fork" || !fork) {
        fork = { kind: "fork", branches: [], join: null };
        root.push(fork);
      }
      target = [];
      fork.branches.push(target);
      continue;
    }
    if (line === "join" || line.startsWith("join ")) {
      if (fork) fork.join = line.slice(4).trim() || null;
      fork = null;
      target = root;
      continue;
    }
    if (line.startsWith("## ")) target.push({ kind: "section", title: line.slice(3) });
    else if (line.startsWith("> ")) {
      const [text, note] = splitNote(line.slice(2));
      target.push({ kind: "op", text, note });
    } else if (line.startsWith("- ")) target.push({ kind: "text", text: line.slice(2) });
    else {
      const [body, note] = splitNote(line);
      const at = body.search(/(^|\s)\[/);
      target.push(
        at < 0 ? { kind: "tensor", name: body, shape: "", note } : { kind: "tensor", name: body.slice(0, at).trim(), shape: body.slice(at).trim(), note },
      );
    }
  }
  return { items: root, axes, kinds };
}

// ---------------------------------------------------------------- оси

type AxisKind = "batch" | "seq" | "model" | "heads" | "hdim" | "wide" | "other" | "num";

const SEQ = new Set(["T", "S", "N", "W", "L"]);
const HEADS = new Set(["H", "H_kv", "H_I", "H_lin"]);
const HDIM = new Set(["D", "d", "dk", "dv", "d_I", "dn", "dr", "P", "R", "dc"]);
const WIDE = new Set(["F", "V", "E", "k", "F_e"]);

function axisKind(dim: string, kinds: Record<string, AxisKind> = {}): AxisKind {
  if (kinds[normDim(dim)]) return kinds[normDim(dim)];
  const ids: string[] = dim.match(/[A-Za-z_]+/g) ?? [];
  if (!ids.length) return "num";
  if (ids.includes("C")) return "model";
  const first = ids[0] ?? "";
  if (first === "B" || first === "b") return "batch";
  if (SEQ.has(first)) return "seq";
  if (HEADS.has(first)) return ids.length === 1 ? "heads" : "model"; // H·D — сплющенные головы
  if (HDIM.has(first)) return "hdim";
  if (WIDE.has(first)) return "wide";
  return "other";
}

const AXIS_MEANING: Record<string, string> = {
  B: "батч",
  b: "батч",
  T: "токены в этом прогоне",
  S: "все позиции с учётом кеша",
  C: "d_model — ширина модели",
  H: "головы запросов",
  H_kv: "головы ключей/значений",
  D: "размер головы",
  d: "размер головы",
  F: "скрытый размер FFN",
  V: "размер словаря",
  E: "число экспертов",
  k: "сколько экспертов выбирает токен (top-k)",
  N: "все токены батча, N = B·T",
  W: "размер окна",
};

const normDim = (d: string) => d.replace(/\s+/g, "");

// Квадратные скобки с перечнем осей; индексация вроде x[:, t] или h[tok] — не форма.
const SHAPE_RE = /(^|[\s(=,;→←:|])\[([^[\]]*)\]/g;
const isShape = (inner: string) =>
  inner.trim() === "" || (!inner.includes(":") && !/["']/.test(inner) && inner.split(",").every((d) => /^[\w·+\-−/ .()²…]+$/.test(d.trim())));

// ---------------------------------------------------------------- рендер

type Hover = { axis: string | null; set(a: string | null): void; kinds: Record<string, AxisKind> };

function Shape({ inner, hover }: { inner: string; hover: Hover }) {
  const dims = inner.trim() ? inner.split(",").map((d) => d.trim()) : [];
  if (!dims.length)
    return (
      <span className="sf-shape">
        <span className="sf-dim ax-num">скаляр</span>
      </span>
    );
  return (
    <span className="sf-shape">
      {dims.map((d, i) => {
        const key = normDim(d);
        const hl = hover.axis === null ? "" : hover.axis === key ? " hl" : " fade";
        return (
          <span key={i} className={`sf-dim ax-${axisKind(d, hover.kinds)}${hl}`} onMouseEnter={() => hover.set(key)} onMouseLeave={() => hover.set(null)}>
            {d}
          </span>
        );
      })}
    </span>
  );
}

/** Текст, в котором формы [..] заменены на цветные оси. */
function Rich({ text, hover }: { text: string; hover: Hover }) {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(SHAPE_RE)) {
    if (!isShape(m[2])) continue;
    const start = m.index! + m[1].length;
    if (start > last) out.push(text.slice(last, start));
    out.push(<Shape key={start} inner={m[2]} hover={hover} />);
    last = start + m[2].length + 2;
  }
  if (last < text.length) out.push(text.slice(last));
  return <>{out}</>;
}

function Flow({ items, hover }: { items: Item[]; hover: Hover }) {
  return (
    <div className="sf-flow">
      {items.map((it, i) => {
        switch (it.kind) {
          case "tensor":
            return (
              <div key={i} className="sf-node">
                {it.name && <span className="sf-name">{it.name}</span>}
                {it.shape && (
                  <span className="sf-shapes">
                    <Rich text={it.shape} hover={hover} />
                  </span>
                )}
                {it.note && (
                  <span className="sf-note">
                    <Rich text={it.note} hover={hover} />
                  </span>
                )}
              </div>
            );
          case "op":
            return (
              <div key={i} className="sf-op">
                <span className="sf-op-text">
                  <Rich text={it.text} hover={hover} />
                </span>
                {it.note && (
                  <span className="sf-note">
                    <Rich text={it.note} hover={hover} />
                  </span>
                )}
              </div>
            );
          case "text":
            return (
              <div key={i} className="sf-text">
                <Rich text={it.text} hover={hover} />
              </div>
            );
          case "section":
            return (
              <div key={i} className="sf-section">
                {it.title}
              </div>
            );
          case "gap":
            return <div key={i} className="sf-gap" />;
          case "fork":
            return (
              <div key={i} className="sf-fork-wrap">
                <div className={`sf-fork ${it.join ? "joined" : ""}`}>
                  {it.branches.map((b, j) => (
                    <div key={j} className="sf-branch">
                      <Flow items={b} hover={hover} />
                      {it.join && <div className="sf-tail" />}
                    </div>
                  ))}
                </div>
                {it.join && (
                  <div className="sf-op sf-join">
                    <span className="sf-op-text">
                      <Rich text={it.join} hover={hover} />
                    </span>
                  </div>
                )}
              </div>
            );
        }
      })}
    </div>
  );
}

function collectAxes(items: Item[], into: Set<string>) {
  for (const it of items) {
    const texts = it.kind === "tensor" ? [it.shape, it.note] : it.kind === "op" ? [it.text, it.note] : it.kind === "text" ? [it.text] : it.kind === "fork" ? [it.join ?? ""] : [];
    for (const t of texts)
      for (const m of t.matchAll(SHAPE_RE)) if (isShape(m[2])) m[2].split(",").forEach((d) => d.trim() && into.add(normDim(d)));
    if (it.kind === "fork") it.branches.forEach((b) => collectAxes(b, into));
  }
}

export function ShapeFlow({ source }: { source: string }) {
  const [axis, setAxis] = useState<string | null>(null);
  const { items, axes, kinds } = parseShapes(source);
  const present = new Set<string>();
  collectAxes(items, present);
  const legend = [...present].filter((a) => axes[a] ?? AXIS_MEANING[a]);
  // Порядок легенды — как оси обычно идут в тензоре.
  const rank = (a: string) => ["batch", "heads", "seq", "hdim", "model", "wide", "other", "num"].indexOf(axisKind(a, kinds));
  legend.sort((a, b) => rank(a) - rank(b));
  const hover = { axis, set: setAxis, kinds };

  return (
    <figure className="sf" aria-label="Схема форм тензоров">
      <Flow items={items} hover={hover} />
      {legend.length > 0 && (
        <figcaption className="sf-legend">
          {legend.map((a) => (
            <span key={a} className="sf-legend-item" onMouseEnter={() => setAxis(a)} onMouseLeave={() => setAxis(null)}>
              <span className={`sf-dim ax-${axisKind(a, kinds)}${axis === a ? " hl" : ""}`}>{a}</span> {axes[a] ?? AXIS_MEANING[a]}
            </span>
          ))}
        </figcaption>
      )}
    </figure>
  );
}
