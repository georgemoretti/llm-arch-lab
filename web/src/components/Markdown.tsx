// Markdown задачи: GFM-таблицы, формулы KaTeX и директива `:::viz <id>:::` на отдельной строке.
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { VIZ_REGISTRY } from "../viz/registry";
import type { VizInfo } from "../api";

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
              a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
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
