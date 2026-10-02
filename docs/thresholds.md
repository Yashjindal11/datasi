# Thresholds

Severity is always the result of an explicit rule over one of these thresholds. Each
finding's `rule` field quotes the rule and the value used. Override any of them in
configuration:

```yaml
thresholds:
  missing_warning: 0.05
  correlation_warning: 0.85
```

Configuration is validated: unknown keys, out-of-range values and inconsistent pairs
(e.g. `missing_warning > missing_high`) are rejected.

| threshold | default | used by | meaning and rationale |
|---|---|---|---|
| `missing_warning` | 0.10 | missingness | Missing share that makes a column a warning. Below it, columns are listed in one info finding. 10 % is where naive complete-case analysis usually starts to lose material power. |
| `missing_high` | 0.50 | missingness | Majority missing: the column describes the minority of rows. |
| `missing_group_min_gap` | 0.20 | missingness | Minimum absolute gap between a group's missing rate and the rest of the data. An effect size, so it does not fire on trivial differences in large data. |
| `missing_cooccurrence_phi` | 0.5 | missingness | φ between missing indicators for "missing together". 0.5 is conventionally a strong association for binary variables. |
| `duplicate_warning` | 0.001 | duplicates | Duplicate-row share for a warning (any duplicate is at least info). |
| `duplicate_high` | 0.05 | duplicates | Share at which duplicates materially change counts and model weights. |
| `near_constant_ratio` | 0.99 | constant, numeric | Top-value share for "nearly constant". |
| `dominant_category_ratio` | 0.95 | constant | Top-label share reported as dominant (info). |
| `identifier_unique_ratio` | 0.95 | schema, identifier, duplicates | Uniqueness that, with another signal, marks an identifier; also the bar for "key-like". |
| `high_cardinality_ratio` | 0.5 | categorical | Unique ratio for category explosion (with `high_cardinality_min`). |
| `high_cardinality_min` | 100 | categorical | Minimum distinct labels for category explosion. |
| `rare_category_ratio` | 0.005 | categorical | Label share below which a label is rare. |
| `type_parse_ratio` | 0.95 | schema, types | Share of values that must parse for text to be treated as numbers/dates. |
| `skew_warning` | 2.0 | numeric | \|skewness\| reported as strong skew. ±2 is a common rule of thumb for "substantially non-normal". |
| `excess_kurtosis_warning` | 10.0 | numeric | Excess kurtosis reported as heavy tails (normal = 0, Laplace = 3, t₅ = 6). |
| `zero_inflation_ratio` | 0.5 | numeric | Share of exact zeros for zero inflation. |
| `value_spike_ratio` | 0.10 | numeric | Share of one non-zero value in a column with ≥ 20 distinct values. |
| `iqr_k` | 1.5 | outliers | Tukey's inner fences. |
| `iqr_extreme_k` | 3.0 | outliers, numeric | Tukey's outer fences ("far out"); reported in evidence and used for sentinels. |
| `zscore` | 3.0 | outliers | Classic z cut-off (0.27 % of a normal sample). |
| `modified_zscore` | 3.5 | outliers | Iglewicz & Hoaglin's recommended cut-off. |
| `isolation_contamination` | `"auto"` | outliers | Passed to `IsolationForest`. Research E3 shows `auto` flags ~13–16 % of rows; set a rate if you know it. |
| `outlier_warning_ratio` | 0.05 | outliers | If more than this share is flagged, the column is heavy-tailed rather than containing isolated anomalies → info. |
| `correlation_warning` | 0.90 | correlation | max(\|r\|, \|ρ\|) reported (info). |
| `correlation_high` | 0.98 | correlation | Near-redundant correlation (warning); also η threshold. |
| `cramers_v_warning` | 0.90 | correlation | Bias-corrected Cramér's V for strongly associated categoricals. |
| `psi_warning` / `psi_high` | 0.10 / 0.25 | drift | Industry-standard PSI bands (< 0.1 stable, 0.1–0.25 moderate, > 0.25 significant). |
| `ks_warning` / `ks_high` | 0.10 / 0.25 | drift, temporal | KS D is the maximum CDF gap; 0.1 ≈ a 0.25 SD mean shift of a normal. |
| `jsd_warning` / `jsd_high` | 0.05 / 0.15 | drift | Jensen–Shannon divergence in bits (0–1). |
| `temporal_min_periods` | 8 | temporal | Fewer periods give change detection too little power. |
| `temporal_shift_gap` | 0.15 | missingness, drift | Minimum absolute change in a rate across a change point / between datasets. |
| `volume_change_ratio` | 0.5 | temporal | Relative change in rows per period across a change point. |
| `gap_factor` | 5.0 | datetime | An interval this many times the usual one is a gap (regular series only). |
| `imbalance_warning` / `imbalance_high` | 0.10 / 0.01 | target | Minority-class share. |
| `target_association_warning` | 0.95 | target | Held-out single-feature AUC/R² considered "too good". |

Other configuration (not thresholds) that affects results:

* `outlier_methods` – default `["iqr", "modified_zscore"]`.
* `performance.sample_rows` (default 200,000) – row cap for pairwise and multivariate
  analyses. Counts and per-column statistics always use every row. Research E4 measures
  the effect of this cap.
* `performance.max_pairwise_columns` (60), `performance.max_group_levels` (50).
* `reference_time` – the "now" for future-date checks (default: run time). Set it for
  reproducible reports.
