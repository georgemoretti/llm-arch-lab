// Markdown задачи: GFM-таблицы, формулы KaTeX и директива `:::viz <id>:::` на отдельной строке.
// NoteMarkdown — то же для конспектов + синтаксис Obsidian: [[wiki-ссылки]] и callout-блоки
// `> [!type]- Заголовок` (знак «-» — свёрнут по умолчанию).
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { VIZ_REGISTRY } from "../viz/registry";
import type { VizInfo } from "../api";
import { noteHref } from "../router";

const VIZ_RE = /^:::viz\s+([a-z0-9][a-z0-9-]*)\s*:::\s*$/gm;

export function Markdown({ text, viz = [] }: { text: string; viz?: VizInfo[] }) {
  // split с группой захвата: [md, id, md, id, md, ...]
  const parts = text.split(VIZ_RE);
  return (
    <div className="md">
      {parts.map((part, i) =>
        i % 2 === 0 ? (
          <ReactMarkdown
            key={i}
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              a: ({ node: _node, ...props }) =>
                props.href?.startsWith("#") ? <a {...props} /> : <a {...props} target="_blank" rel="noreferrer" />,
            }}
          >
            {part}
          </ReactMarkdown>
        ) : (
          <VizBlock key={i} id={part} info={viz.find((v) => v.id === part)} />
        ),
      )}
    </div>
  );
}

export function VizBlock({ id, info }: { id: string; info?: VizInfo }) {
  const Component = VIZ_REGISTRY[id];
  return (
    <figure className="viz">
      <figcaption>
        <span className="viz-badge">viz</span> {info?.title ?? id}
      </figcaption>
      {Component ? <Component /> : <div className="viz-missing">Визуализация «{id}» не зарегистрирована в web/src/viz/registry.ts</div>}
    </figure>
  );
}

type Segment = { kind: "md"; text: string } | { kind: "callout"; type: string; folded: boolean; title: string; body: string };

const CALLOUT_HEAD = /^>\s*\[!([\w-]+)\]([-+]?)\s*(.*)$/;

function splitCallouts(text: string): Segment[] {
  const out: Segment[] = [];
  let md: string[] = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const head = lines[i].match(CALLOUT_HEAD);
    if (!head) {
      md.push(lines[i]);
      continue;
    }
    const body: string[] = [];
    while (i + 1 < lines.length && lines[i + 1].startsWith(">")) body.push(lines[++i].replace(/^> ?/, ""));
    if (md.length) out.push({ kind: "md", text: md.join("\n") });
    md = [];
    out.push({ kind: "callout", type: head[1].toLowerCase(), folded: head[2] === "-", title: head[3], body: body.join("\n") });
  }
  if (md.length) out.push({ kind: "md", text: md.join("\n") });
  return out;
}

// [[id]] / [[id|подпись]] → ссылка на конспект внутри приложения.
const wikiToLinks = (text: string) =>
  text.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_m, id: string, label?: string) => `[${label ?? id}](${noteHref(id.trim())})`);

export function NoteMarkdown({ text }: { text: string }) {
  return (
    <div className="note-md">
      {splitCallouts(text).map((seg, i) =>
        seg.kind === "md" ? (
          <Markdown key={i} text={wikiToLinks(seg.text)} />
        ) : (
          <details key={i} className={`callout callout-${seg.type}`} open={!seg.folded}>
            <summary>{seg.title || seg.type}</summary>
            <Markdown text={wikiToLinks(seg.body)} />
          </details>
        ),
      )}
    </div>
  );
}

// Сниппет поиска: сервер обрамляет совпадения управляющими символами \u0002…\u0003.
export function Highlighted({ text }: { text: string }) {
  const plain = text
    .replace(/>\s*\[![\w-]+\][-+]?/g, "") // заголовок callout
    .replace(/^#{1,6}\s+|\s#{1,6}\s+|(^|\s)>\s?/gm, " ") // заголовки и цитаты
    .replace(/\*\*|`{1,3}/g, "")
    .replace(/\s+/g, " ");
  const parts = plain.split(/\u0002|\u0003/);
  return (
    <>
      {parts.map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>))}
    </>
  );
}
