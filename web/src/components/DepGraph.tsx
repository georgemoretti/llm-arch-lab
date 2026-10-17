// Граф зависимостей задач блока: слои по длине самой длинной цепочки depends_on.
import type { TaskSummary } from "../api";
import { taskHref } from "../router";
import { TYPE_ICON } from "../courseIndex";

const NODE_W = 168;
const NODE_H = 30;
const GAP_X = 52;
const GAP_Y = 12;

export function DepGraph({
  tasks,
  hrefFor = taskHref,
  classFor = () => "",
  label = "Граф зависимостей задач",
}: {
  tasks: TaskSummary[];
  hrefFor?: (id: string) => string;
  classFor?: (t: TaskSummary) => string;
  label?: string;
}) {
  const byId = new Map(tasks.map((t) => [t.id, t]));
  const depth = new Map<string, number>();
  const level = (id: string, seen = new Set<string>()): number => {
    if (depth.has(id)) return depth.get(id)!;
    if (seen.has(id)) return 0; // цикл — загрузчик о нём уже сообщил
    seen.add(id);
    const deps = byId.get(id)?.depends_on.filter((d) => byId.has(d)) ?? [];
    const d = deps.length ? 1 + Math.max(...deps.map((x) => level(x, seen))) : 0;
    depth.set(id, d);
    return d;
  };
  tasks.forEach((t) => level(t.id));

  const columns: TaskSummary[][] = [];
  for (const t of tasks) (columns[depth.get(t.id)!] ??= []).push(t);
  // Внутри столбца — по среднему положению родителей (меньше пересечений рёбер), затем по id.
  const pos = new Map<string, { x: number; y: number }>();
  columns.forEach((col, i) => {
    const bary = (t: TaskSummary) => {
      const ys = t.depends_on.filter((d) => pos.has(d)).map((d) => pos.get(d)!.y);
      return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
    };
    col.sort((a, b) => bary(a) - bary(b) || a.id.localeCompare(b.id));
    col.forEach((t, j) => pos.set(t.id, { x: 4 + i * (NODE_W + GAP_X), y: 4 + j * (NODE_H + GAP_Y) }));
  });
  const width = 8 + columns.length * (NODE_W + GAP_X) - GAP_X;
  const height = 8 + Math.max(1, ...columns.map((c) => c.length)) * (NODE_H + GAP_Y) - GAP_Y;

  return (
    <div className="depgraph">
      {/* viewBox: широкий граф сжимается под ширину контейнера, а не уезжает за край */}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ maxWidth: "100%", height: "auto" }} role="img" aria-label={label}>
        {tasks.flatMap((t) =>
          t.depends_on
            .filter((d) => pos.has(d))
            .map((d) => {
              const a = pos.get(d)!;
              const b = pos.get(t.id)!;
              const x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2, x2 = b.x, y2 = b.y + NODE_H / 2;
              const mx = (x1 + x2) / 2;
              return <path key={`${d}-${t.id}`} className="edge" d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2 - 4},${y2}`} markerEnd="url(#arrow)" />;
            }),
        )}
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="arrowhead" />
          </marker>
        </defs>
        {tasks.map((t) => {
          const p = pos.get(t.id)!;
          return (
            <a key={t.id} href={hrefFor(t.id)}>
              <g className={`node ${t.progress.status} type-${t.type} ${classFor(t)}`} transform={`translate(${p.x},${p.y})`}>
                <title>{t.title}</title>
                <rect width={NODE_W} height={NODE_H} rx={5} />
                <text x={10} y={NODE_H / 2 + 4}>
                  {t.progress.status === "solved" ? "✓ " : ""}
                  {nodeLabel(t)}
                </text>
              </g>
            </a>
          );
        })}
      </svg>
    </div>
  );
}

// Узел узкий: номер + короткое имя, у отладки — 🐞 и номер (имя совпало бы с родителем).
function nodeLabel(t: TaskSummary): string {
  const text = t.parent ? `${TYPE_ICON.debug} ${t.number ?? ""} баг` : t.number ? `${t.number} ${t.short_title}` : `${TYPE_ICON[t.type]} ${t.short_title.replace(/^(Статьи?|Квиз):\s*/, "")}`;
  return text.length > 23 ? text.slice(0, 22) + "…" : text;
}
