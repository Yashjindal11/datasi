import type { Report } from "./types";

async function postForm(path: string, form: FormData): Promise<Report> {
  const resp = await fetch(path, { method: "POST", body: form });
  const data = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }));
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data as Report;
}

export interface InspectOptions {
  target?: string;
  timeColumn?: string;
  disabled?: string[];
  outlierMethods?: string[];
}

function configJson(opts: InspectOptions): string {
  const config: Record<string, unknown> = {};
  if (opts.disabled?.length) config.detectors = Object.fromEntries(opts.disabled.map((d) => [d, false]));
  if (opts.outlierMethods?.length) config.outlier_methods = opts.outlierMethods;
  return JSON.stringify(config);
}

export function inspectFile(file: File, opts: InspectOptions): Promise<Report> {
  const form = new FormData();
  form.append("file", file, file.name);
  if (opts.target) form.append("target", opts.target);
  if (opts.timeColumn) form.append("time_column", opts.timeColumn);
  form.append("config", configJson(opts));
  return postForm("/api/inspect", form);
}

export function compareFiles(reference: File, current: File, target?: string): Promise<Report> {
  const form = new FormData();
  form.append("reference", reference, reference.name);
  form.append("current", current, current.name);
  if (target) form.append("target", target);
  return postForm("/api/compare", form);
}

export async function renderReport(report: Report, format: "html" | "md"): Promise<Blob> {
  const resp = await fetch(`/api/render?format=${format}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(report),
  });
  if (!resp.ok) throw new Error(`export failed: HTTP ${resp.status}`);
  return resp.blob();
}

export async function serverAvailable(): Promise<boolean> {
  try {
    const resp = await fetch("/api/health");
    return resp.ok;
  } catch {
    return false;
  }
}

export function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
