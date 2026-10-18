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
import { noteHref, taskHref } from "../router";
import { codeLookup, labelById, useCourseIndex, type CourseIndex } from "../courseIndex";
import { ShapeFlow } from "./ShapeFlow";

const VIZ_RE = /^:::viz\s+([a-z0-9][a-z0-9-]*)\s*:::\s*$/gm;

export function Markdown({ text, viz = [], linkTasks = false }: { text: string; viz?: VizInfo[]; linkTasks?: boolean }) {
  const index = useCourseIndex();
  // split с группой захвата: [md, id, md, id, md, ...]
  const parts = text.split(VIZ_RE);
  const remarkPlugins = linkTasks && index ? [remarkGfm, remarkMath, remarkTaskRefs(index)] : [remarkGfm, remarkMath];
  return (
    <div className="md">
      {parts.map((part, i) =>
        i % 2 === 0 ? (
          <ReactMarkdown
            key={i}
            remarkPlugins={remarkPlugins}
            rehypePlugins={[rehypeKatex]}
            components={{
              code: ({ node: _node, className, children, ...props }) =>
                className === "language-shapes" ? (
                  <ShapeFlow source={String(children)} />
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                ),
              pre: ({ node, children, ...props }) => {
                // ```shapes рисуется компонентом, обёртка <pre> ему не нужна
                const first = node?.children[0];
                const cls = first && "properties" in first ? first.properties?.className : undefined;
                return Array.isArray(cls) && cls.includes("language-shapes") ? <>{children}</> : <pre {...props}>{children}</pre>;
              },
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

// Упоминания задач в тексте («см. C05», «A01d-…-debug») → ссылка «C1.5 · KV-cache». Ссылкой становится только код
// существующей задачи, так что «A100» или «B2» в тексте не трогаются.
// Код (`…`, ```…```) и существующие ссылки не трогаем: плагин видит только узлы text.
const TASK_REF_RE = /(?<![\w/.-])([A-Z]-[A-Z]\d+[a-z]?|[A-Z]\d{2}[a-z]?)((?:-[a-z0-9]+)*)(?![\w])/g;

type MdNode = { type: string; value?: string; children?: MdNode[]; url?: string; title?: string; data?: unknown };

function remarkTaskRefs(index: CourseIndex) {
  const lookup = codeLookup(index);
  const split = (value: string): MdNode[] => {
    const out: MdNode[] = [];
    let last = 0;
    for (const m of value.matchAll(TASK_REF_RE)) {
      const id = lookup.get(m[0]) ?? (m[2] ? undefined : lookup.get(m[1]));
      if (!id) continue;
      if (m.index! > last) out.push({ type: "text", value: value.slice(last, m.index) });
      out.push({
        type: "link",
        url: taskHref(id),
        title: index.get(id)?.title,
        data: { hProperties: { className: "task-ref" } },
        children: [{ type: "text", value: labelById(index, id) }],
      });
      last = m.index! + m[0].length;
    }
    if (!out.length) return [{ type: "text", value }];
    if (last < value.length) out.push({ type: "text", value: value.slice(last) });
    return out;
  };
  const walk = (node: MdNode) => {
    if (!node.children || node.type === "link") return;
    node.children = node.children.flatMap((c) => (c.type === "text" && c.value ? split(c.value) : (walk(c), [c])));
  };
  return () => (tree: MdNode) => walk(tree);
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
          <Markdown key={i} text={wikiToLinks(seg.text)} linkTasks />
        ) : (
          <details key={i} className={`callout callout-${seg.type}`} open={!seg.folded}>
            <summary>{seg.title || seg.type}</summary>
            <Markdown text={wikiToLinks(seg.body)} linkTasks />
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
