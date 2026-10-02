/** Minimal in-browser preview of delimited files: header + first rows, nothing uploaded. */
export interface Preview {
  columns: string[];
  rows: string[][];
}

function splitLine(line: string, sep: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === sep) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out;
}

export async function previewFile(file: File, maxRows = 8): Promise<Preview | null> {
  const name = file.name.toLowerCase();
  const sep = name.endsWith(".tsv") ? "\t" : name.endsWith(".csv") || name.endsWith(".txt") ? "," : null;
  if (name.endsWith(".jsonl") || name.endsWith(".ndjson")) {
    const text = await file.slice(0, 256 * 1024).text();
    const objs = text.split("\n").filter(Boolean).slice(0, maxRows).map((l) => {
      try {
        return JSON.parse(l) as Record<string, unknown>;
      } catch {
        return null;
      }
    }).filter((o): o is Record<string, unknown> => o !== null);
    if (!objs.length) return null;
    const columns = Object.keys(objs[0]);
    return { columns, rows: objs.map((o) => columns.map((c) => String(o[c] ?? ""))) };
  }
  if (!sep) return null;
  const text = await file.slice(0, 256 * 1024).text();
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (!lines.length) return null;
  return { columns: splitLine(lines[0], sep), rows: lines.slice(1, maxRows + 1).map((l) => splitLine(l, sep)) };
}
