import { useMemo, useState } from "react";
import { download, renderReport } from "../api";
import { CATEGORY_TITLES, SEVERITIES, type Report, type SchemaField, type Severity } from "../types";
import { Bars, Heatmap, Histogram, Line, fmt, pct } from "./Charts";
import FindingCard, { SeverityBadge } from "./FindingCard";

const TABS = ["Overview", "Findings", "Schema", "Distributions", "Temporal", "Relationships", "Drift", "Target", "Recommendations"] as const;
type Tab = (typeof TABS)[number];

function Card({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900">
      <div className="text-xs text-slate-500">{k}</div>
      <div className="text-2xl font-semibold">{v}</div>
    </div>
  );
}

function Panel({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div className="mb-4 rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      {title && <h3 className="mb-2 text-sm font-semibold">{title}</h3>}
      {children}
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <div className="mb-3 rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-600 dark:bg-slate-800 dark:text-slate-300">{children}</div>;
}

function bytes(n: number): string {
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

function FindingsList({ report, category }: { report: Report; category?: string }) {
  const fs = report.findings.filter((f) => !category || f.category === category);
  if (!fs.length) return <p className="text-sm italic text-slate-500">No findings in this section.</p>;
  return (
    <>
      {fs.map((f) => (
        <FindingCard key={f.id + f.code} finding={f} doc={report.detector_docs?.[f.detector]} />
      ))}
    </>
  );
}

function FindingsTab({ report }: { report: Report }) {
  const [sev, setSev] = useState<Set<Severity>>(new Set());
  const [cat, setCat] = useState("");
  const [q, setQ] = useState("");
  const shown = report.findings.filter((f) => {
    if (sev.size && !sev.has(f.severity)) return false;
    if (cat && f.category !== cat) return false;
    if (q) {
      const hay = `${f.title} ${f.code} ${f.columns.join(" ")} ${f.detector}`.toLowerCase();
      if (!hay.includes(q.toLowerCase())) return false;
    }
    return true;
  });
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {SEVERITIES.map((s) => (
          <button
            key={s}
            onClick={() => setSev((prev) => {
              const next = new Set(prev);
              if (next.has(s)) next.delete(s);
              else next.add(s);
              return next;
            })}
            className={`rounded-full border px-3 py-1 text-sm ${sev.has(s) ? "border-indigo-600 bg-indigo-600 text-white" : "border-slate-300 dark:border-slate-700"}`}
          >
            {s} ({report.summary.by_severity[s] ?? 0})
          </button>
        ))}
        <select value={cat} onChange={(e) => setCat(e.target.value)} className="rounded-md border border-slate-300 bg-transparent px-2 py-1 text-sm dark:border-slate-700">
          <option value="">All sections</option>
          {Object.entries(CATEGORY_TITLES).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search title, column, code…"
          className="min-w-56 flex-1 rounded-md border border-slate-300 bg-transparent px-2 py-1 text-sm dark:border-slate-700"
        />
      </div>
      <p className="mb-2 text-xs text-slate-500">{shown.length} of {report.findings.length} findings</p>
      {shown.map((f) => (
        <FindingCard key={f.id + f.code} finding={f} doc={report.detector_docs?.[f.detector]} />
      ))}
    </div>
  );
}

type SortKey = keyof SchemaField;

function SchemaTab({ report }: { report: Report }) {
  const [key, setKey] = useState<SortKey>("position");
  const [asc, setAsc] = useState(true);
  const rows = useMemo(() => {
    const r = [...(report.schema.fields ?? [])];
    r.sort((a, b) => {
      const x = a[key];
      const y = b[key];
      if (x === y) return 0;
      return (x < y ? -1 : 1) * (asc ? 1 : -1);
    });
    return r;
  }, [report, key, asc]);
  const cols: [SortKey, string][] = [["name", "Column"], ["type", "Type"], ["dtype", "dtype"], ["missing_ratio", "Missing"], ["unique", "Unique"], ["unique_ratio", "Unique %"], ["memory_bytes", "Memory"]];
  return (
    <Panel>
      <p className="mb-2 text-xs text-slate-500">Types are inferred from values. Click a header to sort.</p>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500 dark:border-slate-800">
              {cols.map(([k, label]) => (
                <th key={k} className="cursor-pointer whitespace-nowrap px-2 py-1" onClick={() => (k === key ? setAsc(!asc) : (setKey(k), setAsc(true)))}>
                  {label}{k === key ? (asc ? " ▴" : " ▾") : ""}
                </th>
              ))}
              <th className="px-2 py-1">Hints</th>
              <th className="px-2 py-1">Why</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.name} className="border-b border-slate-100 align-top dark:border-slate-800">
                <td className="px-2 py-1 font-medium">{c.name}</td>
                <td className="px-2 py-1">{c.type}</td>
                <td className="px-2 py-1 text-slate-500">{c.dtype}</td>
                <td className="px-2 py-1 text-right tabular-nums">{pct(c.missing_ratio)}</td>
                <td className="px-2 py-1 text-right tabular-nums">{fmt(c.unique)}</td>
                <td className="px-2 py-1 text-right tabular-nums">{pct(c.unique_ratio)}</td>
                <td className="px-2 py-1 text-right tabular-nums">{bytes(c.memory_bytes)}</td>
                <td className="px-2 py-1 text-xs">{c.hints.join(", ")}</td>
                <td className="px-2 py-1 text-xs text-slate-500">{c.reasons[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function DistributionsTab({ report }: { report: Report }) {
  const miss = report.sections.missingness?.columns ?? {};
  const missItems = Object.entries(miss).filter(([, v]) => v.ratio > 0).sort((a, b) => b[1].ratio - a[1].ratio).map(([k, v]) => ({ label: k, value: v.ratio }));
  return (
    <div>
      <Panel title="Missing values per column">
        {missItems.length ? <Bars items={missItems} max={1} format={pct} /> : <p className="text-sm text-slate-500">No missing values.</p>}
      </Panel>
      {report.sections.missingness?.matrix && report.sections.missingness.matrix.columns.length >= 2 && (
        <Panel title="Missingness co-occurrence (phi between missing indicators)">
          <Heatmap labels={report.sections.missingness.matrix.columns} matrix={report.sections.missingness.matrix.phi} />
        </Panel>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        {Object.entries(report.profiles).map(([col, p]) => {
          if (p.numeric?.count && p.histogram) {
            const s = p.numeric;
            return (
              <Panel key={col} title={`${col} (numeric)`}>
                <Histogram edges={p.histogram.edges} counts={p.histogram.counts} />
                <p className="mt-1 text-xs text-slate-500">
                  n={fmt(s.count)} · mean {fmt(s.mean)} · median {fmt(s.median)} · sd {fmt(s.std)} · IQR {fmt(s.iqr)} · skew {fmt(s.skewness, 2)}
                </p>
              </Panel>
            );
          }
          if (p.categorical) {
            const c = p.categorical;
            return (
              <Panel key={col} title={`${col} (categorical)`}>
                <Bars items={c.top.map((t) => ({ label: t.label, value: t.share }))} max={1} format={pct} labelWidth={130} />
                <p className="mt-1 text-xs text-slate-500">
                  {fmt(c.distinct)} distinct · entropy {fmt(c.entropy_bits, 2)} bits · {fmt(c.rare_levels)} rare levels
                </p>
              </Panel>
            );
          }
          return null;
        })}
      </div>
    </div>
  );
}

function TemporalTab({ report }: { report: Report }) {
  const t = report.sections.temporal;
  const fs = report.findings.filter((f) => f.category === "temporal" || f.code === "missing.over_time");
  return (
    <div>
      {t ? (
        <Panel title={`Rows per ${t.period} by '${t.time_column}'`}>
          <Line points={t.volume} zero />
        </Panel>
      ) : (
        <Note>{report.time_column ? `The time range of '${report.time_column}' is too short for period analysis.` : "No suitable time column found. Choose one before running the investigation."}</Note>
      )}
      {fs.map((f) => <FindingCard key={f.id + f.code} finding={f} doc={report.detector_docs?.[f.detector]} />)}
    </div>
  );
}

function RelationshipsTab({ report }: { report: Report }) {
  const c = report.sections.correlation;
  const [method, setMethod] = useState<"pearson" | "spearman">("pearson");
  const matrix = c?.[method];
  return (
    <div>
      {c?.numeric_columns && matrix && c.numeric_columns.length >= 2 && (
        <Panel title="Correlation heatmap">
          <div className="mb-2 flex gap-2">
            {(["pearson", "spearman"] as const).map((m) => (
              <button key={m} onClick={() => setMethod(m)} className={`rounded-full border px-3 py-0.5 text-sm ${m === method ? "border-indigo-600 bg-indigo-600 text-white" : "border-slate-300 dark:border-slate-700"}`}>
                {m}
              </button>
            ))}
          </div>
          <Heatmap labels={c.numeric_columns} matrix={matrix} />
          <p className="text-xs text-slate-500">Indigo = positive, orange = negative. Correlation does not imply causation.</p>
        </Panel>
      )}
      <FindingsList report={report} category="relationships" />
    </div>
  );
}

function DriftTab({ report }: { report: Report }) {
  const d = report.sections.drift;
  if (!d) return <Note>Use the Compare tab to compare a reference dataset (e.g. train) with a current one (e.g. test).</Note>;
  const keys = ["column", "kind", "ks_statistic", "psi", "jsd", "wasserstein_iqr", "missing_reference", "missing_current", "severity"];
  return (
    <div>
      {d.domain_classifier_auc !== null && (
        <Note>Domain classifier AUC: {fmt(d.domain_classifier_auc, 3)} (0.5 = the datasets are indistinguishable).</Note>
      )}
      <Panel>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 dark:border-slate-800">{keys.map((k) => <th key={k} className="px-2 py-1">{k}</th>)}</tr>
            </thead>
            <tbody>
              {d.columns.map((r, i) => (
                <tr key={i} className="border-b border-slate-100 dark:border-slate-800">
                  {keys.map((k) => (
                    <td key={k} className="px-2 py-1 tabular-nums">{k.startsWith("missing") ? pct(r[k] as number) : fmt(r[k], 3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <FindingsList report={report} category="distribution" />
    </div>
  );
}

function TargetTab({ report }: { report: Report }) {
  const t = report.sections.target;
  if (!t) return <Note>No target configured. Pick a target column before running the investigation.</Note>;
  return (
    <div>
      <p className="mb-3 text-sm">Target <strong>{t.column}</strong> ({t.task}).</p>
      {t.classes && (
        <Panel title="Class distribution">
          <Bars items={t.classes.map((c) => ({ label: c.label, value: c.share }))} max={1} format={pct} />
        </Panel>
      )}
      {t.feature_scores && (
        <Panel title={`Single-feature predictive power (held-out ${t.task === "regression" ? "R²" : "AUC"})`}>
          <Bars
            items={t.feature_scores.filter((s) => s.score !== null).map((s) => ({ label: s.column, value: s.score as number, alt: (s.score as number) >= 0.95 }))}
            max={1}
            format={(v) => fmt(v, 3)}
          />
        </Panel>
      )}
      <FindingsList report={report} category="target" />
    </div>
  );
}

function Overview({ report }: { report: Report }) {
  const s = report.summary;
  const d = report.dataset;
  const top = report.findings.filter((f) => f.severity === "critical" || f.severity === "high");
  const warnings = report.findings.filter((f) => f.severity === "warning");
  const cats = Object.entries(s.by_category);
  return (
    <div>
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
        <Card k="Rows" v={fmt(d.rows)} />
        <Card k="Columns" v={fmt(d.columns)} />
        <Card k="Memory" v={bytes(d.memory_bytes)} />
        <Card k="Findings" v={fmt(s.findings)} />
        {SEVERITIES.map((sv) => <Card key={sv} k={sv[0].toUpperCase() + sv.slice(1)} v={fmt(s.by_severity[sv] ?? 0)} />)}
      </div>
      {report.reference && <Note>Comparing current “{d.name}” ({fmt(d.rows)} rows) with reference “{report.reference.name}” ({fmt(report.reference.rows)} rows).</Note>}
      {d.analysis_sample_rows && <Note>Pairwise and multivariate analyses used a {fmt(d.analysis_sample_rows)}-row sample; counts use all rows.</Note>}
      <h2 className="mb-2 mt-4 font-semibold">Critical and high findings</h2>
      {top.length ? top.map((f) => <FindingCard key={f.id + f.code} finding={f} doc={report.detector_docs?.[f.detector]} />) : <p className="text-sm italic text-slate-500">None.</p>}
      <h2 className="mb-2 mt-4 font-semibold">Warnings</h2>
      {warnings.length ? warnings.map((f) => <FindingCard key={f.id + f.code} finding={f} doc={report.detector_docs?.[f.detector]} />) : <p className="text-sm italic text-slate-500">None.</p>}
      {cats.length > 0 && (
        <Panel title="Findings by section">
          <Bars items={cats.map(([k, v]) => ({ label: CATEGORY_TITLES[k] ?? k, value: v }))} />
        </Panel>
      )}
    </div>
  );
}

function Recommendations({ report }: { report: Report }) {
  if (!report.recommendations.length) return <p className="text-sm italic text-slate-500">No warning-level findings, so no recommendations.</p>;
  return (
    <ol className="list-decimal space-y-2 pl-6">
      {report.recommendations.map((r) => (
        <li key={r.finding_id + r.suggestion}>
          <SeverityBadge severity={r.severity} /> <strong>{r.suggestion}</strong>
          <div className="text-xs text-slate-500">Because: {r.because}{r.columns.length ? ` (${r.columns.join(", ")})` : ""}</div>
        </li>
      ))}
    </ol>
  );
}

export default function ReportView({ report, canExport }: { report: Report; canExport: boolean }) {
  const [tab, setTab] = useState<Tab>("Overview");
  const [error, setError] = useState<string | null>(null);
  const base = report.dataset.name.replace(/\.[^.]+$/, "") || "report";
  const exportAs = async (format: "json" | "html" | "md") => {
    setError(null);
    try {
      if (format === "json") download(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }), `datasi-${base}.json`);
      else download(await renderReport(report, format), `datasi-${base}.${format}`);
    } catch (e) {
      setError(String(e));
    }
  };
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold">{report.kind === "comparison" ? "Comparison" : "Investigation"}: {report.dataset.name}</h2>
          <p className="text-xs text-slate-500">
            {fmt(report.dataset.rows)} rows × {report.dataset.columns} columns · DataSI {report.datasi_version} · {report.created_at}
          </p>
        </div>
        <div className="flex gap-2 text-sm">
          <button onClick={() => exportAs("json")} className="rounded-md border border-slate-300 px-3 py-1 dark:border-slate-700">Export JSON</button>
          <button disabled={!canExport} title={canExport ? "" : "Needs the local DataSI server"} onClick={() => exportAs("html")} className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40 dark:border-slate-700">Export HTML</button>
          <button disabled={!canExport} onClick={() => exportAs("md")} className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40 dark:border-slate-700">Export Markdown</button>
        </div>
      </div>
      {error && <Note>{error}</Note>}
      <nav className="mb-4 flex flex-wrap gap-1 border-b border-slate-200 dark:border-slate-800">
        {TABS.map((t) => (
          <button key={t} onClick={() => setTab(t)} className={`-mb-px border-b-2 px-3 py-2 text-sm ${t === tab ? "border-indigo-600 font-semibold" : "border-transparent text-slate-500"}`}>
            {t}
          </button>
        ))}
      </nav>
      {tab === "Overview" && <Overview report={report} />}
      {tab === "Findings" && <FindingsTab report={report} />}
      {tab === "Schema" && <SchemaTab report={report} />}
      {tab === "Distributions" && <DistributionsTab report={report} />}
      {tab === "Temporal" && <TemporalTab report={report} />}
      {tab === "Relationships" && <RelationshipsTab report={report} />}
      {tab === "Drift" && <DriftTab report={report} />}
      {tab === "Target" && <TargetTab report={report} />}
      {tab === "Recommendations" && <Recommendations report={report} />}
    </div>
  );
}
