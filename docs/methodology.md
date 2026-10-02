# Methodology

DataSI is a statistical investigator, not an oracle. This page explains how it reasons
and where that reasoning stops.

## Observation vs interpretation

Every `Finding` separates:

| field | contains | phrasing |
|---|---|---|
| `observation` | what was measured | factual: "18.4 % of values are missing" |
| `evidence` | the numbers behind it (JSON) | counts, ratios, statistics, row positions |
| `rule` | the explicit rule that set the severity | "missing ratio ≥ missing_warning (0.1) → warning" |
| `interpretation` | what it *might* mean | always a possibility: "may indicate", "could also be" |
| `confidence` | how much to trust the interpretation | `None` for pure measurements |
| `suggestion` | what to check next | an investigation, never an automatic fix |

DataSI never modifies data and never states speculation as fact. A statistical outlier
is "unusual", not "invalid"; a predictive feature is "suspicious", not "leaking";
correlated columns are "associated", not "causal".

## Severity

| level | meaning |
|---|---|
| `critical` | analysis cannot proceed meaningfully (no rows, target absent or single-class) |
| `high` | likely to corrupt results if ignored (majority missing, ≥ 5 % duplicates, declared rule violations, suspiciously predictive features, large drift) |
| `warning` | worth investigating before use |
| `info` | descriptive facts and weak signals |

All levels come from the documented thresholds in [thresholds.md](thresholds.md).

## Confidence

Confidence is a **heuristic in [0, 1]**, not a probability. It is lower when a finding
rests on assumptions the data may violate (independence for change points, symmetry for
robust z-scores) and is capped below 1 except for user-declared rules (1.0). Examples:

* change points: 0.5, +0.2 if the CUSUM statistic ≥ 2.5, +0.1 with ≥ 20 periods, max 0.85;
* missingness by group: 0.5 + V/2 (max 0.9);
* outliers: share of flagged values on which all configured methods agree;
* target "suspicious feature": fixed 0.5, because data alone cannot distinguish a strong
  predictor from leakage.

## Effect sizes before p-values

With 1M rows every trivial difference is "statistically significant"; with 200 rows
real problems are not. DataSI therefore decides on **effect sizes** (KS D, PSI, JSD,
Cramér's V, absolute rate gaps) and reports p-values only as context. The one place a
null distribution sets a threshold is the CUSUM change-point test, where it controls
false alarms on stationary series (verified in `tests/statistical`).

## Change points

For a per-period series x₁…xₙ, the standardised CUSUM statistic is

    T = max_k |Σ_{i≤k}(xᵢ − x̄)| / (σ̂ √n)

Under i.i.d. noise T converges to the supremum of a Brownian bridge (Kolmogorov
distribution); DataSI uses the 1 % critical value 1.628. σ̂ is the MAD of first
differences divided by 0.6745·√2, so the level shift being tested does not inflate it.
Binary segmentation finds up to three change points; a change is only reported when its
size also passes an effect-size threshold. Autocorrelation and seasonality inflate T –
this is the main known source of false temporal alarms.

## Outliers and sample size

A fixed cut-off (z > 3, 3 × IQR) flags *something* in almost every large dataset because
extremes grow with n. DataSI keeps the configured methods for *listing* outliers (info)
but only raises a **warning** when a value exceeds the sample-size-adjusted robust
threshold k_n = Φ⁻¹(1 − 0.005/n) on the (log-transformed, if right-skewed) scale. This
change was made after research experiment E2 measured warning-level false positives on
clean data.

## Leakage-adjacent checks

Single-feature predictive power is measured **out of sample** (random 50/50 split) with
a per-level / per-quantile-bin target-mean model, so unique identifiers cannot score by
memorisation. High scores are reported as suspicious with confidence 0.5. The dedicated
leakage project (LeakSI) is out of scope here.

## Privacy

* Data never leaves the machine: no network calls, no telemetry, no API keys, no LLM.
* Reports contain column names, aggregate statistics, category labels of categorical
  columns (needed to explain findings) and **row positions** – never raw rows.
* `privacy.include_examples: true` adds up to `privacy.max_examples` sampled raw values
  to evidence. `privacy.redact_labels: true` replaces category labels with stable hashes.
* Evidence about unparseable text uses character-class shapes (`A/A`, `99-AA`).
* The web server binds to 127.0.0.1, checks the Host header (DNS-rebinding protection),
  caps uploads and deletes uploaded files after analysis.

## Scalability

* Counts, missingness, duplicates and per-column statistics use **all rows**.
* Pairwise and multivariate analyses (correlation, mutual information, isolation forest,
  target scoring) use a deterministic sample capped at `performance.sample_rows`; the
  report discloses when this happens. Research E4 measures the effect on detection.
* String analyses work on distinct values (`value_counts`), so their cost scales with
  cardinality rather than row count.
* `datasi.statistics.streaming` provides mergeable statistics (Welford/Chan moments,
  k-minimum-values distinct counts) and `datasi schema --chunksize N` profiles CSVs in
  bounded memory. Running all detectors on chunked data is future work.

## Known limitations

* Synthetic research results (`docs/research.md`) use relatively clean, strong planted
  problems; real data is messier, so treat those numbers as upper bounds.
* Inference of semantic types is heuristic; declare `id_columns`, `time_column` and
  `rules` when you know them.
* Univariate drift metrics miss joint shifts; the domain classifier only partly covers
  this.
* English-centric name heuristics (identifiers, future-date names).
