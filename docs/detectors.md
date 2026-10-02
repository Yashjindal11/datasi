# Detector reference

Every detector documents **what it measures**, its **assumptions**, the **threshold**
that sets severity, how to **interpret** a finding, and its **limitations**. The same
text is embedded in every HTML report next to the findings a detector produces, and
`datasi detectors` lists them.

Detectors run in this order (later detectors may reuse cached work from earlier ones):
`schema → missingness → duplicates → constant → identifier → types → rules → numeric →
outliers → categorical → strings → datetime → temporal → correlation → redundancy →
target`. Comparison (`datasi compare`) uses the separate `drift` detector.

Threshold names refer to [`thresholds.md`](thresholds.md); every one is configurable.

---

## schema

| code | severity | rule |
|---|---|---|
| `schema.empty` | critical | dataset has 0 rows |
| `schema.duplicate_name` | warning | two columns share a name (analysed as `name__2`, …) |
| `schema.name_collision` | warning | names equal after case-folding / stripping |
| `schema.name_whitespace`, `schema.blank_name` | info / warning | surrounding whitespace / empty name |
| `schema.exported_index` | info | `Unnamed: 0`, `index`, `level_0` … |
| `schema.config_column_missing` | critical (target) / warning | configuration names an absent column |

**Limitations.** No external schema contract is checked; that is a different tool's job.

## missingness

**Measures.** Missing count/ratio per column and per row, then *structure* in the
missingness:

* `missing.by_group` – a categorical group whose missing rate differs from the rest
  of the data by at least `missing_group_min_gap` (groups with ≥ 1 % of rows). Effect
  size: bias-corrected Cramér's V between the missing indicator and the group.
* `missing.over_time` – CUSUM change point in the per-period missing rate with an
  absolute shift ≥ `temporal_shift_gap` (see [methodology](methodology.md#change-points)).
* `missing.co_occurrence` – clusters of columns whose missing indicators have
  φ ≥ `missing_cooccurrence_phi`.
* `missing.related_numeric` – a numeric column whose distribution differs (KS D ≥ 0.3)
  between rows where the column is missing and rows where it is present.

**Severity.** `missing.high`: ratio ≥ `missing_high` → high, ≥ `missing_warning` →
warning. `missing.complete` (100 % missing) → high. Below `missing_warning` a single
`missing.low` info lists the affected columns. Structural findings are warnings/info.

**Interpretation.** Structure means the data are probably **not missing completely at
random**; imputing with one global statistic may bias groups or periods.

**Limitations.** Missingness mechanisms cannot be proven from data. Associations may be
confounded. Placeholder strings (`"N/A"`, `"-"`) are reported by `strings`, not here.

## duplicates

| code | severity | rule |
|---|---|---|
| `duplicates.exact` | info → warning (≥ `duplicate_warning`) → high (≥ `duplicate_high`) | rows identical to an earlier row |
| `duplicates.except_ids` | same tiers | identical on every non-identifier column, different IDs |
| `duplicates.key` | warning (high if declared `unique`) | repeated values in a ≥ 95 %-unique identifier |
| `duplicates.entity_conflict` | warning | a categorical attribute constant for ≥ 95 % (not all) of repeated entity keys |

**Limitations.** Exact matching only. Legitimate repeated events look like duplicates.
Columns holding lists/dicts are compared by their `repr`.

## constant

`constant.single_value` (warning): one distinct non-missing value.
`constant.near` (warning): top value share ≥ `near_constant_ratio`.
`constant.dominant` (info): categorical/boolean top share ≥ `dominant_category_ratio`.

**Interpretation.** Little variation limits what a column can explain, but rare values
may be exactly the interesting cases (fraud flags). DataSI never calls a column useless.

## identifier

`identifier.detected` explains *why* a column was typed as an identifier: name pattern
(`*_id`, `uuid`, `record_number`, `customerId` …), ≥ `identifier_unique_ratio` unique,
strictly monotonic integers (with share of +1 steps), UUID format, one fixed
character-class format (`AA-999999`), or low reuse. Numeric identifiers are warnings
(easily mistaken for a quantity); others are info. `identifier.near_unique` flags
categorical columns that are nearly unique per row (keys, free text or quasi-identifiers).

**Limitations.** Name patterns are English-centric; use `id_columns` to declare IDs.

## types

| code | severity | rule |
|---|---|---|
| `types.numeric_text` | info; warning if several conventions or unparseable values | ≥ `type_parse_ratio` of text values are numbers/currency/percentages |
| `types.mostly_numeric` | warning | 50 % ≤ numeric share < `type_parse_ratio` |
| `types.date_text` | info | dates stored as text, one format |
| `types.date_formats` | warning | ≥ 2 recognised date formats each in ≥ 0.1 % of rows |
| `types.timezone_mixed` | warning | mix of no-offset / `Z` / `+hh:mm` |
| `types.boolean_text` | info; warning if > 2 spellings | yes/no/true/false/y/n … |
| `types.mixed` | warning | an object column mixes Python types |

Evidence for non-parsing values uses **character-class shapes** (`"N/A"` → `A/A`) so
formats are visible without exposing values. **Data is never converted.**

**Limitations.** Day/month order in `01/02/2026` is ambiguous and flagged as such.
Locale formats such as `1.234,5` are not recognised.

## rules

User-declared domain rules (`min`, `max`, `allowed`, `pattern`, `nullable`, `unique`).
Every violation is `rules.<kind>` with severity **high** and confidence 1.0. This is the
only way DataSI labels a value **confirmed invalid**.

## numeric

Profiles every numeric column (moments, quantiles, IQR, bias-corrected skewness and
excess kurtosis, histogram) and reports:

* `numeric.skewed` (info): |skewness| ≥ `skew_warning`.
* `numeric.heavy_tails` (info): excess kurtosis ≥ `excess_kurtosis_warning`.
* `numeric.zero_inflated` (info): share of zeros ≥ `zero_inflation_ratio`.
* `numeric.rare_negative` (warning): 0 < negative share ≤ 1 % with a positive median –
  phrased as a *potential anomaly*; no rule says negatives are impossible.
* `numeric.value_spike` (info): one non-zero value holds ≥ `value_spike_ratio` of a
  column with ≥ 20 distinct values.
* `numeric.sentinel` (warning): codes such as −1, −999, 9999 repeated and extreme.
* `numeric.infinite` (warning).

## outliers

Methods (`outlier_methods`, default `iqr` + `modified_zscore`):

| method | flag when | assumes | robust |
|---|---|---|---|
| `iqr` | outside Q1 − k·IQR, Q3 + k·IQR (`iqr_k`) | ordinal scale | yes |
| `zscore` | \|x − mean\| / sd > `zscore` | near-normal | no |
| `modified_zscore` | 0.6745 \|x − median\| / MAD > `modified_zscore` (Iglewicz–Hoaglin) | symmetric | yes |
| `isolation_forest` | sklearn IsolationForest, robust-scaled, multivariate | anomalies few & different | yes |

Positive columns with skewness ≥ 0.5 are scored on the **log scale** so a natural long
tail is not reported. Severity is **warning** only when some value lies beyond a
**sample-size-adjusted** robust threshold k_n = Φ⁻¹(1 − 0.005/n) (a value that extreme
appears anywhere in a normal sample of size n with probability 1 %) and fewer than
`outlier_warning_ratio` of values are flagged; otherwise **info**. Confidence is the
share of flagged values that every method agrees on.

**Interpretation.** A statistical outlier is *unusual*, not *wrong*. See
[research E3](research.md) for measured precision/recall per method and threshold.

## categorical

Profiles cardinality, Shannon entropy (bits and normalised), top labels and rare labels.

* `categorical.explosion` (warning): ≥ `high_cardinality_min` labels and unique ratio
  ≥ `high_cardinality_ratio`.
* `categorical.rare` (info): ≥ 2 labels below `rare_category_ratio`.
* `categorical.variants` (warning): labels equal after case-folding, whitespace and
  punctuation normalisation (`United`, `UNITED`, `united `).
* `categorical.similar_labels` (info): labels within one edit (two for ≥ 10 characters,
  transpositions count as one), or one label extending another by a word
  (`United` / `United Airlines`). Labels with digits are never paired.

Nothing is merged automatically.

## strings

`strings.empty` (empty/whitespace-only strings, not counted as missing),
`strings.placeholder` (`N/A`, `null`, `-`, `unknown`, … – missing values in disguise),
`strings.padding`, `strings.unusual_characters` (control, zero-width, NBSP, U+FFFD),
`strings.format` (a character-class format covering ≥ 90 % of values with exceptions;
warning at ≥ 99 %).

## datetime

`datetime.unparseable`, `datetime.future` (after `reference_time`; info for names like
`expiry`/`due`/`scheduled`), `datetime.placeholder` (1900-01-01, 1970-01-01,
9999-12-31, years < 1900), `datetime.mixed_granularity`. For the primary time column:
`datetime.irregular_interval` (regular series with gaps > `gap_factor` × interval),
`datetime.duplicate_timestamps`, `datetime.empty_periods` (zero-row periods inside the
range when the median period has ≥ 5 rows). Profiles min/max, span and granularity.

## temporal

Needs a primary time column (`time_column`, or auto-detected: the best-populated datetime
column with a time-like name; names like `birth`, `expiry`, `due` are avoided). Rows are
binned into the finest of hour/day/week/month/quarter/year giving ≤ 100 periods
(≥ `temporal_min_periods`).

* `temporal.volume_shift` (warning): CUSUM change in rows/period with relative change
  ≥ `volume_change_ratio`. First and last (possibly partial) periods are excluded.
* `temporal.volume_spike` (info): |count − rolling median| > 5 MAD-based SDs.
* `temporal.distribution_shift` (warning): CUSUM change in per-period medians **and**
  before/after KS D ≥ `ks_high`.
* `temporal.new_category` / `temporal.vanished_category` (info): labels with ≥ 1 % of
  rows first seen after the first 20 % of periods / last seen before the final 20 %.

**Limitations.** Seasonality and trends violate the independence assumption of the
CUSUM threshold and can be reported as shifts.

## correlation

Pearson and Spearman (numeric pairs), bias-corrected Cramér's V (categorical pairs),
correlation ratio η (categorical → numeric), and normalised mutual information on 10
quantile bins to surface **non-linear** dependence (NMI ≥ 0.5 with |ρ| < 0.5). Severity:
max(|r|, |ρ|) ≥ `correlation_high` → warning, ≥ `correlation_warning` → info; the other
measures are info. Computed on the analysis sample. **Correlation never implies
causation**, and findings say so.

## redundancy

Deterministic relationships, verified on all rows:

* `redundancy.identical` – identical columns.
* `redundancy.linear` – `b = a·x + c` (robust fit, coefficients snapped to 6 significant
  digits) holds for ≥ 99.9 % of complete rows within relative tolerance 1e-6 – e.g.
  `height_m = 0.01 × height_cm`.
* `redundancy.sum` – `z = x ± y` holds likewise (screened on 5,000 rows).
* `*_broken` – the identity holds for 95–99.9 % of rows: the breaking rows are likely
  errors in one of the columns (warning, with row positions).
* `redundancy.one_to_one` – two categoricals in a perfect bijection (code ↔ label).

## target

Enabled by `target=`. Classification vs regression is inferred.

* `target.missing` (warning/high; critical if all missing), `target.single_class`
  (critical), `target.imbalance` (minority < `imbalance_warning` → warning,
  < `imbalance_high` → high), `target.tiny_classes`.
* `target.suspicious_feature` (high): a **single feature** reaches held-out AUC (or R²)
  ≥ `target_association_warning`. The model is a per-level / per-quantile-bin target mean
  fitted on a random half and scored on the other half, so memorising unique IDs scores
  at chance. Findings say the feature is *suspiciously predictive*, never that it *leaks*
  (confidence 0.5): only knowledge of *when* a value is recorded can establish leakage.
* `target.identifier_predictive` (warning): an identifier with held-out AUC ≥ 0.7
  (ordered IDs encoding time or a sorted dataset).
* `target.missingness_predictive` (warning): a column's missing indicator alone reaches
  AUC ≥ 0.75.

## drift (comparison)

Schema changes (`drift.columns_missing` high, `drift.columns_added` warning,
`drift.type_changed` high), `drift.missingness`, numeric shift (`drift.numeric`: worse
of KS D vs `ks_*` and PSI vs `psi_*`), `drift.out_of_range` (≥ 1 % of current values
outside the reference range), categorical shift (`drift.categorical`: JSD vs `jsd_*`
and PSI), `drift.unseen_categories`, and an overall **domain classifier**
(`drift.domain_classifier`: 3-fold cross-validated AUC of gradient boosting separating
reference from current rows; ≥ 0.6 warning, ≥ 0.8 high). Identifier, text and datetime
columns are excluded from metrics. Decisions use **effect sizes**; the KS p-value is
reported only as context.

## Writing your own detector

```python
from datasi import Severity, investigate
from datasi.detectors import BaseDetector, Context
from datasi.findings import Category, Finding


class NegativePrices(BaseDetector):
    name = "negative_prices"
    category = Category.NUMERIC
    description = "Prices must be positive."

    def analyze(self, ctx: Context) -> list[Finding]:
        bad = ctx.numeric("price") < 0
        if not bad.any():
            return []
        return [self.finding("custom.negative_price", Severity.HIGH, "Negative prices",
                             f"{int(bad.sum())} rows have price < 0.", columns=["price"],
                             evidence={"row_examples": ctx.row_examples(bad)})]


report = investigate(df, extra_detectors=[NegativePrices()])
```

Any object with a `name` and `analyze(ctx) -> list[Finding]` satisfies the `Detector`
protocol. A detector that raises is recorded as an error in the report; the run continues.
