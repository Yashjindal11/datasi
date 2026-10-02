# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue. Report privately through
[GitHub security advisories](https://github.com/Yashjindal11/datasi/security/advisories/new).
Include the affected version (`datasi --version`), a description and a minimal
reproduction (a schema or synthetic data, never your real data). You can expect an
acknowledgement within a week. Fixes are released as patch versions and credited in the
changelog unless you prefer otherwise.

Supported versions: the latest minor release.

## DataSI's threat model

DataSI reads data that may be sensitive and may be adversarial (crafted CSV/JSON).

| Concern | Behaviour |
|---|---|
| Data exfiltration | DataSI makes no network requests and has no telemetry. Analysis is local. |
| Data in reports | Reports hold aggregates, column names, category labels and row positions. Raw values only with `privacy.include_examples: true`; labels can be hashed with `privacy.redact_labels: true`. Treat reports as derived from your data anyway. |
| Malicious values in HTML reports | Report data is embedded as JSON with `<`, `>` and `&` escaped and rendered with text-only DOM APIs; a strict Content-Security-Policy blocks network access from the report. |
| Input mutation | The input DataFrame is never modified. |
| Web server exposure | `datasi serve` binds to `127.0.0.1`, rejects requests whose `Host` header is not the loopback address (DNS rebinding), sends no CORS headers, limits upload size (`DATASI_MAX_UPLOAD_MB`), accepts only data file extensions, serves static files only from its own directory, and deletes uploads after analysis. Binding to another host prints a warning; do not expose it publicly without authentication in front of it. |
| Code execution | DataSI never executes code or formulas from data. Configuration is parsed with `yaml.safe_load`. Regex `pattern` rules come from your own configuration; do not load configuration from untrusted sources. |
| Resource exhaustion | Very large or pathological files can exhaust memory. Use `--nrows`, `performance.sample_rows` or `datasi schema --chunksize` for untrusted large inputs. |
