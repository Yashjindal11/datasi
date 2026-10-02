# Command line

```
datasi inspect DATA [-o report.html] [--target COL] [--time-column COL] [--config FILE]
                    [--detectors a,b] [--disable a,b] [--outlier-methods iqr,zscore,...]
                    [--nrows N] [--include-examples] [--reference-time ISO]
                    [--fail-on SEVERITY] [--min-severity SEVERITY] [--json] [--limit N]
datasi report  DATA [-o report.html -o report.md -o report.json] [...same options]
datasi compare REFERENCE CURRENT [-o drift.html] [--target COL] [--fail-on SEVERITY] [--json]
datasi schema  DATA [--json] [--nrows N] [--chunksize N]
datasi config  validate FILE | init [-o FILE]
datasi detectors
datasi serve   [--port 8765] [--host 127.0.0.1] [--no-browser]
```

* `DATA` is a `.csv`, `.tsv`, `.csv.gz`, `.parquet`, `.json` or `.jsonl` file.
* `-o` is repeatable; the format follows the extension (`.html`, `.md`, `.json`).
* Exit codes: `0` success, `1` a finding at or above `--fail-on` exists (use in CI), `2`
  usage, configuration or loading error.
* `schema --chunksize N` streams a CSV in chunks and reports exact missing counts,
  exact moments and approximate distinct counts in bounded memory.

## Data quality gate in CI

```yaml
- run: pip install "datasi @ git+https://github.com/Yashjindal11/datasi@v0.1.0"
- run: datasi inspect data/train.parquet --config datasi.yaml --fail-on high -o datasi-report.html
- run: datasi compare data/train.parquet data/test.parquet --fail-on high
```
