import { useEffect, useState } from "react";
import { compareFiles, inspectFile, serverAvailable } from "./api";
import ReportView from "./components/ReportView";
import { previewFile, type Preview } from "./preview";
import type { Report } from "./types";

const DETECTORS = [
  "missingness", "duplicates", "constant", "identifier", "types", "numeric", "outliers",
  "categorical", "strings", "datetime", "temporal", "correlation", "redundancy", "target",
];
const ACCEPT = ".csv,.tsv,.txt,.json,.jsonl,.ndjson,.parquet,.gz";

function FilePicker({ label, file, onFile }: { label: string; file: File | null; onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
      }}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center text-sm ${over ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950" : "border-slate-300 dark:border-slate-700"}`}
    >
      <input type="file" accept={ACCEPT} className="hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
      <span className="font-medium">{file ? file.name : label}</span>
      <span className="mt-1 text-xs text-slate-500">{file ? `${(file.size / 1e6).toFixed(2)} MB` : "CSV, TSV, JSON / JSONL or Parquet — drop or click"}</span>
    </label>
  );
}

function PreviewTable({ preview }: { preview: Preview }) {
  return (
    <div className="mt-3 overflow-x-auto rounded-md border border-slate-200 dark:border-slate-800">
      <p className="px-2 pt-1 text-[11px] text-slate-500">Preview parsed in your browser (first rows).</p>
      <table className="w-full text-left text-xs">
        <thead><tr>{preview.columns.map((c, i) => <th key={i} className="whitespace-nowrap px-2 py-1">{c}</th>)}</tr></thead>
        <tbody>
          {preview.rows.map((r, i) => (
            <tr key={i} className="border-t border-slate-100 dark:border-slate-800">{r.map((v, j) => <td key={j} className="whitespace-nowrap px-2 py-0.5 text-slate-600 dark:text-slate-400">{v}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InspectForm({ onReport }: { onReport: (r: Report) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [target, setTarget] = useState("");
  const [timeColumn, setTimeColumn] = useState("");
  const [disabled, setDisabled] = useState<Set<string>>(new Set());
  const [iforest, setIforest] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = async (f: File) => {
    setFile(f);
    setTarget("");
    setTimeColumn("");
    setPreview(await previewFile(f));
  };
  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      onReport(await inspectFile(file, {
        target: target || undefined,
        timeColumn: timeColumn || undefined,
        disabled: [...disabled],
        outlierMethods: iforest ? ["iqr", "modified_zscore", "isolation_forest"] : undefined,
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const cols = preview?.columns ?? [];
  return (
    <div>
      <FilePicker label="Choose a dataset" file={file} onFile={pick} />
      {preview && <PreviewTable preview={preview} />}
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <label className="text-sm">Target column (optional)
          {cols.length ? (
            <select value={target} onChange={(e) => setTarget(e.target.value)} className="mt-1 w-full rounded-md border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700">
              <option value="">None</option>
              {cols.map((c) => <option key={c}>{c}</option>)}
            </select>
          ) : <input value={target} onChange={(e) => setTarget(e.target.value)} className="mt-1 w-full rounded-md border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700" />}
        </label>
        <label className="text-sm">Time column (auto-detected if empty)
          {cols.length ? (
            <select value={timeColumn} onChange={(e) => setTimeColumn(e.target.value)} className="mt-1 w-full rounded-md border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700">
              <option value="">Auto</option>
              {cols.map((c) => <option key={c}>{c}</option>)}
            </select>
          ) : <input value={timeColumn} onChange={(e) => setTimeColumn(e.target.value)} className="mt-1 w-full rounded-md border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700" />}
        </label>
      </div>
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-slate-600 dark:text-slate-400">Detectors</summary>
        <div className="mt-2 flex flex-wrap gap-2">
          {DETECTORS.map((d) => (
            <label key={d} className="flex items-center gap-1 rounded-md border border-slate-200 px-2 py-0.5 dark:border-slate-800">
              <input type="checkbox" checked={!disabled.has(d)} onChange={() => setDisabled((prev) => {
                const next = new Set(prev);
                if (next.has(d)) next.delete(d);
                else next.add(d);
                return next;
              })} />
              {d}
            </label>
          ))}
          <label className="flex items-center gap-1 rounded-md border border-slate-200 px-2 py-0.5 dark:border-slate-800">
            <input type="checkbox" checked={iforest} onChange={() => setIforest(!iforest)} /> isolation forest
          </label>
        </div>
      </details>
      <button disabled={!file || busy} onClick={run} className="mt-4 rounded-md bg-indigo-600 px-4 py-2 font-medium text-white disabled:opacity-50">
        {busy ? "Investigating…" : "Run investigation"}
      </button>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}

function CompareForm({ onReport }: { onReport: (r: Report) => void }) {
  const [ref, setRef] = useState<File | null>(null);
  const [cur, setCur] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async () => {
    if (!ref || !cur) return;
    setBusy(true);
    setError(null);
    try {
      onReport(await compareFiles(ref, cur));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div>
      <div className="grid gap-3 sm:grid-cols-2">
        <FilePicker label="Reference (e.g. train)" file={ref} onFile={setRef} />
        <FilePicker label="Current (e.g. test)" file={cur} onFile={setCur} />
      </div>
      <button disabled={!ref || !cur || busy} onClick={run} className="mt-4 rounded-md bg-indigo-600 px-4 py-2 font-medium text-white disabled:opacity-50">
        {busy ? "Comparing…" : "Compare datasets"}
      </button>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}

function OpenReport({ onReport }: { onReport: (r: Report) => void }) {
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="text-sm">
      <p className="mb-2 text-slate-600 dark:text-slate-400">Open a JSON report produced by <code>datasi inspect data.csv -o report.json</code>. It is read in your browser; nothing is uploaded.</p>
      <input type="file" accept=".json" onChange={async (e) => {
        const f = e.target.files?.[0];
        if (!f) return;
        try {
          const data = JSON.parse(await f.text()) as Report;
          if (!data.findings || !data.dataset) throw new Error("not a DataSI report");
          onReport(data);
          setError(null);
        } catch (err) {
          setError(String(err));
        }
      }} />
      {error && <p className="mt-2 text-red-600">{error}</p>}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<"inspect" | "compare" | "open">("inspect");
  const [report, setReport] = useState<Report | null>(null);
  const [server, setServer] = useState<boolean | null>(null);
  useEffect(() => {
    serverAvailable().then(setServer);
  }, []);
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold">DataSI</h1>
          <p className="text-sm text-slate-500">Find what is wrong with your data before your model does.</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs ${server ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300" : "bg-slate-200 text-slate-600 dark:bg-slate-800"}`}>
          {server === null ? "checking…" : server ? "local engine connected · data stays on this machine" : "viewer only · start `datasi serve` to analyse files"}
        </span>
      </header>
      {!report && (
        <div className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900">
          <div className="mb-4 flex gap-2">
            {([["inspect", "Investigate"], ["compare", "Compare"], ["open", "Open report"]] as const).map(([k, label]) => (
              <button key={k} onClick={() => setMode(k)} className={`rounded-md px-3 py-1 text-sm ${mode === k ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" : "text-slate-600 dark:text-slate-400"}`}>
                {label}
              </button>
            ))}
          </div>
          {mode === "inspect" && <InspectForm onReport={setReport} />}
          {mode === "compare" && <CompareForm onReport={setReport} />}
          {mode === "open" && <OpenReport onReport={setReport} />}
        </div>
      )}
      {report && (
        <div>
          <button onClick={() => setReport(null)} className="mb-4 text-sm text-indigo-600">← New analysis</button>
          <ReportView report={report} canExport={!!server} />
        </div>
      )}
    </div>
  );
}
