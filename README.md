# DataSI

**Find what is wrong with your data before your model does.**

DataSI (Data Intelligence) is a local-first, model-free data quality **investigation**
toolkit for Python. Give it a dataset; it systematically investigates missingness,
duplicates, types, outliers, categories, text hygiene, time, relationships, drift and
target problems, and returns structured findings that separate **what was measured**
from **what it might mean**.

No LLMs. No API keys. No cloud. No telemetry. Your data never leaves your machine.

[![CI](https://github.com/Yashjindal11/datasi/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashjindal11/datasi/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Why another data quality tool?

Profilers tell you *that* a column is 13 % missing. Validation frameworks tell you
*whether* data meets rules you already wrote. DataSI asks the next questions
automatically:

```
[WARNING] Missingness in 'quantity' depends on 'region'
Observation: 'quantity' is missing in 70.1% of rows where region = 'East', versus 12.8% overall (Cramér's V = 0.81).
Why flagged: A group's missing rate differs from the rest of the data by at least missing_group_min_gap (0.2) ...
Possible interpretation: Missingness is not uniform across groups, so the data are probably not missing
completely at random. The collection process may differ between groups.
Confidence: 0.90
Suggested investigation: Check how 'quantity' is collected for each 'region' value before imputing with a
single global statistic.

[WARNING] Missingness in 'score' increased around 2025-10-06
Observation: The share of missing 'score' values increased from 0.4% to 100.0% per week, with the change
starting at 2025-10-06 (by 'order_ts').
Why flagged: Significant CUSUM change point (1% level, independence assumed) with an absolute shift >= 0.15.
Possible interpretation: A sudden change in missingness often coincides with a change in the upstream source,
form, or pipeline. It could also reflect a genuine change in the population.
Suggested investigation: Check whether the upstream source for 'score' changed around that date.
```

(Real output on [`tests/data/dataset_missing.csv`](tests/data/dataset_missing.csv).)

Design principles:

* **Observation ≠ interpretation.** Facts, evidence and the severity rule are separate
  from possible explanations, which are always phrased as possibilities with a confidence.
* **Explicit, configurable severity.** Every `info / warning / high / critical` comes
  from a documented threshold ([thresholds](docs/thresholds.md)).
* **Effect sizes before p-values.** Large datasets make everything "significant"; DataSI
  decides on KS D, PSI, JSD, Cramér's V and rate gaps.
* **Unusual is not invalid.** Statistical outliers are never called errors; only your
  domain rules produce *confirmed invalid* findings.
* **Measured, not claimed.** Detector precision/recall is measured on synthetic data
  with known ground truth ([research](docs/research.md)).

## Install

DataSI is not published on PyPI yet. Install from GitHub (Python 3.11+):

```bash
pip install "datasi @ git+https://github.com/Yashjindal11/datasi"             # core
pip install "datasi[parquet,plot] @ git+https://github.com/Yashjindal11/datasi"  # + pyarrow, matplotlib
```

Core dependencies: numpy, pandas, scipy, scikit-learn, pydantic, pyyaml. Wheels attached
to [GitHub releases](https://github.com/Yashjindal11/datasi/releases) include the
prebuilt web UI. For development: `git clone https://github.com/Yashjindal11/datasi &&
cd datasi && pip install -e ".[dev]"`.

## Quickstart

```python
from datasi import inspect

report = inspect(df)                    # or inspect("data.parquet"), inspect("data.csv")
print(report.summary)

report = inspect(
    "orders.parquet",
    target="churned",                    # enables target analysis
    detectors=["missingness", "duplicates", "outliers", "temporal"],  # optional subset
    config={"thresholds": {"missing_warning": 0.05}},
)

for f in report.filter(min_severity="warning"):
    print(f.explain())

report.save("report.html")              # also .md and .json
```

Train/test comparison:

```python
from datasi import inspect_train_test

drift = inspect_train_test(train_df, test_df, target="label")
```

Command line:

```bash
datasi inspect data.parquet --output report.html
datasi inspect data.csv --target churned --fail-on high      # exit 1 if high/critical findings (CI gate)
datasi compare train.csv test.csv -o drift.html
datasi schema data.csv
datasi report data.csv -o report.html -o report.md -o report.json
datasi config validate datasi.yaml
datasi serve                                                  # local web UI
```

## What DataSI investigates

| area | examples |
|---|---|
| **Schema** | inferred semantic types (numeric, categorical, boolean, datetime, text, identifier, constant), storage hints, duplicate/colliding names, exported index columns |
| **Missingness** | per column and row; concentration in groups; change over time; columns missing together; association with numeric variables |
| **Duplicates** | exact rows, rows identical except for IDs, repeated keys, repeated entities with conflicting attributes |
| **Types** | numbers/currency/percentages/dates/booleans stored as text, mixed types, inconsistent date formats, mixed timezone notation |
| **Numeric** | moments, quantiles, skew, heavy tails, zero inflation, rare negatives, value spikes, sentinel codes (−999, 9999) |
| **Outliers** | IQR, z-score, modified z-score, isolation forest; log scale for skewed data; sample-size-adjusted severity |
| **Categorical** | cardinality, entropy, rare labels, category explosion, case/whitespace variants, typo-level spelling variants |
| **Text** | empty/blank strings, placeholder tokens (`N/A`, `unknown`), padding, invisible/control characters, format breaks |
| **Time** | unparseable/future/placeholder dates, granularity, gaps, irregular sampling, volume shifts and spikes, distribution shifts, new/vanished categories |
| **Relationships** | Pearson, Spearman, Cramér's V, correlation ratio, mutual information; identical columns, unit conversions (`height_m = 0.01 × height_cm`), totals (`total = x + y`) and the rows that break them; code↔label bijections |
| **Target** | missing labels, imbalance, tiny classes, single features that predict the target suspiciously well out of sample, predictive identifiers and missingness |
| **Drift** | schema changes, missingness changes, KS / Wasserstein / PSI / JSD per column, unseen categories, out-of-range values, domain-classifier AUC |
| **Rules** | your `min` / `max` / `allowed` / `pattern` / `nullable` / `unique` constraints → confirmed invalid values |

Full reference with assumptions and limitations: [docs/detectors.md](docs/detectors.md).

## Reports

* **HTML** – a single self-contained file that works offline from `file://` (no CDN, a
  strict Content-Security-Policy, data rendered as text only): summary cards, filterable
  findings with expandable evidence, sortable schema table, missingness bars,
  histograms and box plots, category frequencies, correlation heatmaps, time-series
  charts with change points, drift tables, target diagnostics, recommendations and
  detector explanations.
* **Markdown** – Executive Summary, Dataset Overview, Schema, Critical/High findings,
  one section per area, Recommendations.
* **JSON** – the full structured report (`Report.from_json` round-trips).
* **Figures** – `datasi.visualization` draws matplotlib figures from a report.

Recommendations are drawn only from warning-or-worse findings and cite the finding
that motivates them.

## Web UI

`datasi serve` starts a local React + TypeScript + Tailwind interface on
`127.0.0.1`: upload a dataset (previewed in the browser), pick target/time columns and
detectors, run the investigation, filter findings, inspect evidence, view distributions
and temporal changes, compare datasets and export reports. See [docs/web.md](docs/web.md).

## Research: how reliable is it?

`benchmarks/experiments.py` plants known problems with `datasi.synthetic` and scores
detectors with `datasi.evaluation` (precision, recall, false positive/negative rates).
Highlights from the [recorded run](docs/research.md) (20 datasets × 5,000 rows):

* Every planted issue type was found (recall 1.0 at warning level or above), except
  numbers-stored-as-text, which is deliberately **info** severity.
* Outlier findings at **any** severity had precision 0.455 (heavy natural tails are
  reported as info by design); at **warning** severity precision was 0.976.
* On clean data, warning-level false positives were 0.1 per 1,000-row dataset and 0 per
  10,000- and 100,000-row dataset; info-level outlier findings remain (~2–3 per dataset).
* Drift: no false alarms at shift 0; a 0.25 log-SD shift was detected in 70–80 % of
  pairs and 0.5 log-SD in 100 %. Detection does not grow with sample size because
  thresholds are effect sizes, not p-values.
* Isolation forest with `contamination="auto"` had precision 0.06–0.07 on single
  columns; with the true contamination rate it reached 0.95–0.99.

These are synthetic, relatively clean problems; treat them as upper bounds for messy
real data.

## Performance

`benchmarks/bench_inspect.py` (all detectors plus target analysis on a 14-column table;
Apple M4, Python 3.12, pandas 3.0; results in
[`benchmarks/results/performance.json`](benchmarks/results/performance.json)):

| rows | DataFrame size | runtime | peak RSS (process) |
|---:|---:|---:|---:|
| 10,100 | 1.1 MB | 0.66 s | 197 MB |
| 101,000 | 11.5 MB | 3.35 s | 305 MB |
| 1,010,000 | 114.8 MB | 11.26 s | 1,229 MB |

Counts and per-column statistics use every row; pairwise and multivariate analyses use
a deterministic sample (`performance.sample_rows`, default 200,000) that the report
discloses. `datasi schema --chunksize N` profiles CSVs that do not fit in memory using
mergeable streaming statistics.

## Privacy

* Runs entirely locally; no network access, telemetry or external services.
* Never modifies the input (tests assert this).
* Reports contain aggregates, column names, category labels and **row positions**, not
  raw rows. Raw example values appear only with `privacy.include_examples: true`;
  `privacy.redact_labels: true` hashes category labels.

## Configuration

```yaml
detectors: {correlation: true, temporal: true}
thresholds: {missing_warning: 0.1, missing_high: 0.5, correlation_warning: 0.9}
outlier_methods: [iqr, modified_zscore]
target: churned
rules:
  age: {min: 0, max: 120}
  status: {allowed: [active, closed]}
```

See [docs/configuration.md](docs/configuration.md) and [docs/thresholds.md](docs/thresholds.md).

## Extending

Detectors are plugins: any object with a `name` and `analyze(ctx) -> list[Finding]`.
New input sources register with `datasi.loaders.register_adapter`. See
[docs/detectors.md](docs/detectors.md#writing-your-own-detector) and
[docs/architecture.md](docs/architecture.md).

## Documentation

[Detectors](docs/detectors.md) · [Thresholds](docs/thresholds.md) ·
[Methodology](docs/methodology.md) · [Configuration](docs/configuration.md) ·
[CLI](docs/cli.md) · [Web UI](docs/web.md) · [Architecture](docs/architecture.md) ·
[Research](docs/research.md) · [Examples](examples/)

## Roadmap

* Chunked execution of all detectors for out-of-core data; SQL and native Polars adapters.
* Seasonality-aware change detection.
* LeakSI: a dedicated target-leakage investigator building on the target detector.

## License

MIT – see [LICENSE](LICENSE). Contributions welcome: [CONTRIBUTING.md](CONTRIBUTING.md).
