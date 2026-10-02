# Web UI

```bash
pip install "datasi @ git+https://github.com/Yashjindal11/datasi"   # or a release wheel
datasi serve            # opens http://127.0.0.1:8765
```

Installing from a git checkout does not include the built UI; build it once with
`npm --prefix web ci && npm --prefix web run build` (release wheels include it).

The UI talks only to the local DataSI process. It can:

1. upload a dataset (drag and drop; CSV/TSV/JSON(L) are previewed **in the browser**
   before anything is sent, and target/time column pickers are filled from the header);
2. inspect the inferred schema;
3. run an investigation with chosen detectors (and optionally the isolation forest);
4. browse findings, filter by severity/section/text and expand evidence;
5. view distributions, missingness, correlation heatmaps and temporal charts;
6. compare a reference and a current dataset (drift table, domain classifier);
7. export the report as JSON (in-browser) or HTML/Markdown (rendered locally).

Without the server (e.g. `npm run dev` alone, or a static copy of the build) the UI
works as an offline **report viewer**: "Open report" reads a JSON report in the browser.

## Development

```bash
cd web
npm ci
npm run dev       # Vite on :5173, proxies /api to datasi serve on :8765
npm run build     # type-checks and writes ../src/datasi/server/static
```

The built assets are git-ignored and shipped inside the wheel.

## Server API

| method | path | body | returns |
|---|---|---|---|
| GET | `/api/health` | – | `{status, version}` |
| GET | `/api/detectors` | – | detector docs + default config |
| POST | `/api/inspect` | multipart: `file`, optional `target`, `time_column`, `config` (JSON) | report JSON |
| POST | `/api/compare` | multipart: `reference`, `current`, optional `target`, `config` | comparison JSON |
| POST | `/api/render?format=html\|md` | report JSON | rendered report |

Security: binds to 127.0.0.1 by default, rejects foreign `Host` headers, sends no CORS
headers, limits upload size (`DATASI_MAX_UPLOAD_MB`, default 1024), accepts only data
file extensions and deletes uploads immediately after analysis.
