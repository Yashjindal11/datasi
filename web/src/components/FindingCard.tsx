import type { DetectorDoc, Finding, Point, Severity } from "../types";
import { Line } from "./Charts";

const SEV_STYLE: Record<Severity, string> = {
  critical: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  high: "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300",
  warning: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  info: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
};
const SEV_BORDER: Record<Severity, string> = {
  critical: "border-l-red-600",
  high: "border-l-orange-500",
  warning: "border-l-amber-500",
  info: "border-l-slate-400",
};

export function SeverityBadge({ severity }: { severity: Severity }) {
  return <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase ${SEV_STYLE[severity]}`}>{severity}</span>;
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="mt-3 text-[11px] font-semibold uppercase tracking-wide text-slate-500">{children}</div>;
}

export default function FindingCard({ finding: f, doc }: { finding: Finding; doc?: DetectorDoc }) {
  const { series, ...evidence } = f.evidence as { series?: Point[] } & Record<string, unknown>;
  return (
    <details className={`mb-2 rounded-md border border-l-4 border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 ${SEV_BORDER[f.severity]}`}>
      <summary className="flex cursor-pointer list-none items-baseline gap-3 px-3 py-2">
        <SeverityBadge severity={f.severity} />
        <span className="flex-1 font-medium">{f.title}</span>
        <span className="text-xs text-slate-500">{f.columns.join(", ")}</span>
      </summary>
      <div className="px-3 pb-3 text-sm">
        <Label>Observation</Label>
        <p>{f.observation}</p>
        {f.interpretation && (
          <>
            <Label>Possible interpretation{f.confidence !== null ? ` · confidence ${f.confidence.toFixed(2)}` : ""}</Label>
            <p>{f.interpretation}</p>
          </>
        )}
        {f.suggestion && (
          <>
            <Label>Suggested investigation</Label>
            <p>{f.suggestion}</p>
          </>
        )}
        {f.rule && (
          <>
            <Label>Why this severity</Label>
            <p className="text-xs text-slate-600 dark:text-slate-400">{f.rule}</p>
          </>
        )}
        {series && (
          <>
            <Label>Over time</Label>
            <Line points={series} marker={typeof evidence.change_at === "string" ? evidence.change_at : undefined} />
          </>
        )}
        <Label>Evidence</Label>
        <pre className="mt-1 max-h-72 overflow-auto rounded bg-slate-50 p-2 text-xs dark:bg-slate-950">{JSON.stringify(evidence, null, 2)}</pre>
        {doc && (
          <>
            <Label>About the {f.detector} detector</Label>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              {doc.description} {doc.limitations && <em>Limitations: {doc.limitations}</em>}
            </p>
          </>
        )}
        <p className="mt-2 text-[11px] text-slate-400">code {f.code} · id {f.id}</p>
      </div>
    </details>
  );
}
