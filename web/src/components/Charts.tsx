import type { Point } from "../types";

export function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined) return "–";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return v.toLocaleString();
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 1e6)) return v.toExponential(2);
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? "–" : `${(v * 100).toFixed(v > 0 && v < 0.001 ? 3 : 1)}%`;

export function Bars({ items, max, format = fmt, labelWidth = 150 }: {
  items: { label: string; value: number; alt?: boolean }[];
  max?: number;
  format?: (v: number) => string;
  labelWidth?: number;
}) {
  const w = 520;
  const rowH = 20;
  const top = max ?? Math.max(1e-12, ...items.map((d) => d.value));
  return (
    <svg viewBox={`0 0 ${w} ${items.length * rowH + 6}`} className="w-full" role="img">
      {items.map((d, i) => {
        const bw = Math.max(1, (d.value / top) * (w - labelWidth - 70));
        const y = i * rowH + 3;
        return (
          <g key={`${d.label}-${i}`}>
            <text x={labelWidth - 6} y={y + 13} textAnchor="end" className="fill-slate-500 text-[11px]">
              {d.label.length > 24 ? `${d.label.slice(0, 23)}…` : d.label}
            </text>
            <rect x={labelWidth} y={y + 3} width={bw} height={rowH - 7} rx={2} className={d.alt ? "fill-amber-500" : "fill-indigo-500"}>
              <title>{`${d.label}: ${format(d.value)}`}</title>
            </rect>
            <text x={labelWidth + bw + 4} y={y + 13} className="fill-slate-500 text-[11px]">
              {format(d.value)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function Histogram({ edges, counts }: { edges: number[]; counts: number[] }) {
  if (!counts.length) return <p className="text-sm text-slate-500">No data</p>;
  const w = 340;
  const h = 110;
  const top = Math.max(...counts);
  const bw = (w - 20) / counts.length;
  return (
    <svg viewBox={`0 0 ${w} ${h + 18}`} className="w-full" role="img">
      {counts.map((c, i) => {
        const bh = top ? (c / top) * h : 0;
        return (
          <rect key={i} x={10 + i * bw} y={h - bh} width={Math.max(bw - 1, 1)} height={bh} className="fill-indigo-500/80">
            <title>{`${fmt(edges[i])} – ${fmt(edges[i + 1])}: ${fmt(c)}`}</title>
          </rect>
        );
      })}
      <text x={10} y={h + 14} className="fill-slate-500 text-[10px]">{fmt(edges[0])}</text>
      <text x={w - 10} y={h + 14} textAnchor="end" className="fill-slate-500 text-[10px]">{fmt(edges[edges.length - 1])}</text>
    </svg>
  );
}

export function Line({ points, marker, zero = false }: { points: Point[]; marker?: string; zero?: boolean }) {
  const vals = points.filter((p) => p.v !== null).map((p) => p.v as number);
  if (vals.length < 2) return <p className="text-sm text-slate-500">Not enough points</p>;
  const w = 560;
  const h = 140;
  const pad = 30;
  let min = Math.min(...vals, ...(zero ? [0] : []));
  let max = Math.max(...vals);
  if (max === min) max = min + 1;
  if (min === max) min -= 1;
  const x = (i: number) => pad + (i / Math.max(points.length - 1, 1)) * (w - 2 * pad);
  const y = (v: number) => h - ((v - min) / (max - min)) * (h - 20);
  let d = "";
  points.forEach((p, i) => {
    if (p.v === null) return;
    d += `${d ? " L" : "M"} ${x(i).toFixed(1)} ${y(p.v).toFixed(1)}`;
  });
  const mi = marker ? points.findIndex((p) => marker.startsWith(p.t)) : -1;
  return (
    <svg viewBox={`0 0 ${w} ${h + 18}`} className="w-full" role="img">
      <line x1={pad} x2={w - pad} y1={h} y2={h} className="stroke-slate-300 dark:stroke-slate-700" />
      <path d={d} fill="none" className="stroke-indigo-500" strokeWidth={1.6} />
      {mi >= 0 && <line x1={x(mi)} x2={x(mi)} y1={6} y2={h} className="stroke-orange-500" strokeDasharray="4 3" />}
      <text x={pad} y={h + 14} className="fill-slate-500 text-[10px]">{points[0].t}</text>
      <text x={w - pad} y={h + 14} textAnchor="end" className="fill-slate-500 text-[10px]">{points[points.length - 1].t}</text>
      <text x={2} y={12} className="fill-slate-500 text-[10px]">{fmt(max)}</text>
      <text x={2} y={h} className="fill-slate-500 text-[10px]">{fmt(min)}</text>
    </svg>
  );
}

export function Heatmap({ labels, matrix }: { labels: string[]; matrix: (number | null)[][] }) {
  const n = labels.length;
  const cell = Math.max(10, Math.min(28, Math.floor(520 / Math.max(n, 1))));
  const lab = 120;
  const size = lab + n * cell;
  return (
    <svg viewBox={`0 0 ${size + 90} ${size}`} className="w-full" style={{ maxWidth: size + 90 }} role="img">
      {labels.map((l, i) => (
        <g key={l}>
          <text x={lab - 4} y={lab + i * cell + cell * 0.7} textAnchor="end" className="fill-slate-500 text-[10px]">{l.slice(0, 18)}</text>
          <text
            x={lab + i * cell + cell * 0.7}
            y={lab - 4}
            transform={`rotate(-60 ${lab + i * cell + cell * 0.7} ${lab - 4})`}
            className="fill-slate-500 text-[10px]"
          >
            {l.slice(0, 18)}
          </text>
          {labels.map((m, j) => {
            const v = matrix[i][j];
            const color = v === null ? "#999" : v >= 0 ? `rgba(79,70,229,${Math.abs(v)})` : `rgba(234,88,12,${Math.abs(v)})`;
            return (
              <rect key={m} x={lab + j * cell} y={lab + i * cell} width={cell - 1} height={cell - 1} fill={color}>
                <title>{`${l} × ${m}: ${fmt(v, 3)}`}</title>
              </rect>
            );
          })}
        </g>
      ))}
    </svg>
  );
}
