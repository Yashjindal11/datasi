from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from datasi.detectors.base import BaseDetector, Context, pct, register_detector
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType
from datasi.statistics import outliers as ol
from datasi.statistics.descriptive import histogram, numeric_summary

SENTINELS = frozenset(
    {-1.0, -9.0, -99.0, -999.0, -9999.0, -99999.0, 99.0, 999.0, 9999.0, 99999.0, 999999.0}
)


@register_detector
class NumericDetector(BaseDetector):
    name = "numeric"
    category = Category.NUMERIC
    description = (
        "Distribution shape of numeric columns: summary statistics, skew, heavy tails, "
        "zero inflation, rare negative values, value spikes and sentinel-like codes."
    )
    assumptions = (
        "Skewness/kurtosis are bias-corrected sample estimators; they are descriptive, "
        "not tests. Without domain rules no value is called impossible."
    )
    limitations = (
        "Shape statistics are sensitive to a few extreme values. Sentinel detection "
        "only knows common codes (-1, -999, 9999, ...)."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        t = ctx.thresholds
        out: list[Finding] = []
        for col in ctx.columns_of(ColumnType.NUMERIC):
            values = ctx.numeric(col)
            s = numeric_summary(values)
            prof = ctx.profile(col)
            prof["numeric"] = s
            prof["histogram"] = histogram(values)
            n = s["count"]
            if s["infinite"]:
                out.append(
                    self.finding(
                        "numeric.infinite",
                        Severity.WARNING,
                        f"Infinite values in '{col}'",
                        f"'{col}' contains {s['infinite']:,} infinite value(s).",
                        columns=[col],
                        evidence={"infinite": s["infinite"]},
                        rule="Any +/-inf -> warning.",
                        interpretation="Usually produced by division by zero upstream.",
                        suggestion="Trace the computation that produced this column.",
                    )
                )
            if n < 20:
                continue
            arr = values.to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            skew, kurt = s.get("skewness"), s.get("excess_kurtosis")
            if skew is not None and abs(skew) >= t.skew_warning:
                out.append(
                    self.finding(
                        "numeric.skewed",
                        Severity.INFO,
                        f"'{col}' is strongly skewed",
                        f"Sample skewness of '{col}' is {skew:.2f} (median {s['median']:.4g}, mean {s['mean']:.4g}).",
                        columns=[col],
                        evidence={"skewness": skew, "mean": s["mean"], "median": s["median"]},
                        rule=f"|skewness| >= skew_warning ({t.skew_warning}) -> info.",
                        interpretation=(
                            "Common for amounts, durations and counts. Mean-based summaries and "
                            "z-score outlier rules are unreliable for such columns."
                        ),
                        suggestion="Prefer medians/quantiles, or consider a log transform for modelling.",
                    )
                )
            if kurt is not None and kurt >= t.excess_kurtosis_warning:
                out.append(
                    self.finding(
                        "numeric.heavy_tails",
                        Severity.INFO,
                        f"'{col}' has heavy tails",
                        f"Excess kurtosis of '{col}' is {kurt:.1f} (0 for a normal distribution); "
                        f"the 99th percentile is {s['p99']:.4g} but the maximum is {s['max']:.4g}.",
                        columns=[col],
                        evidence={
                            "excess_kurtosis": kurt,
                            "p99": s["p99"],
                            "max": s["max"],
                            "p01": s["p01"],
                            "min": s["min"],
                        },
                        rule=f"Excess kurtosis >= excess_kurtosis_warning ({t.excess_kurtosis_warning}) -> info.",
                        interpretation="A few very large values dominate the variance; check whether they are genuine.",
                    )
                )
            zero_share = s["zeros"] / n
            if t.zero_inflation_ratio <= zero_share < t.near_constant_ratio and s["distinct"] > 2:
                out.append(
                    self.finding(
                        "numeric.zero_inflated",
                        Severity.INFO,
                        f"'{col}' is zero-inflated",
                        f"{pct(zero_share)} of values in '{col}' are exactly 0.",
                        columns=[col],
                        evidence={"zero_share": zero_share, "zeros": s["zeros"]},
                        rule=f"Share of zeros >= zero_inflation_ratio ({t.zero_inflation_ratio}) -> info.",
                        interpretation="Zeros may be true zeros or a default used for 'unknown'.",
                        suggestion="Check whether 0 is ever used as a placeholder for missing.",
                        confidence=0.5,
                    )
                )
            neg_share = s["negatives"] / n
            if 0 < neg_share <= 0.01 and s["median"] > 0:
                out.append(
                    self.finding(
                        "numeric.rare_negative",
                        Severity.WARNING,
                        f"Potential anomaly: rare negative values in '{col}'",
                        f"{s['negatives']:,} value(s) ({pct(neg_share)}) of '{col}' are negative "
                        f"while the median is {s['median']:.4g}.",
                        columns=[col],
                        evidence={
                            "negatives": s["negatives"],
                            "share": neg_share,
                            "min": s["min"],
                            "row_examples": ctx.row_examples(values.to_numpy() < 0),
                        },
                        rule="0 < negative share <= 1% in a column with positive median -> warning.",
                        interpretation=(
                            "Could be refunds/corrections that are legitimate, sign errors, or codes. "
                            "No domain rule says negatives are impossible here."
                        ),
                        suggestion=f"Add a rule (rules.{col}.min) if negatives are truly invalid.",
                        confidence=0.5,
                    )
                )
            out.extend(self._spikes(ctx, col, values, arr, s))
        return out

    def _spikes(
        self, ctx: Context, col: str, values: pd.Series, arr: np.ndarray, s: dict[str, float]
    ) -> list[Finding]:
        t = ctx.thresholds
        out: list[Finding] = []
        vc = pd.Series(arr).value_counts()
        if s["distinct"] < 20 or vc.empty:
            return out
        fences = ol.iqr_bounds(values, t.iqr_extreme_k)
        reported: set[float] = set()
        for value, count in vc.head(5).items():
            v = float(value)
            share = count / len(arr)
            extreme = fences is not None and (v < fences[0] or v > fences[1])
            sentinel = v in SENTINELS and count >= 2 and (extreme or share >= 0.01)
            if sentinel:
                reported.add(v)
                out.append(
                    self.finding(
                        "numeric.sentinel",
                        Severity.WARNING,
                        f"Possible sentinel value {v:g} in '{col}'",
                        f"The value {v:g} occurs {count:,} times ({pct(share)}) in '{col}'"
                        + (
                            " and lies far outside the bulk of the distribution."
                            if extreme
                            else "."
                        ),
                        columns=[col],
                        evidence={
                            "value": v,
                            "count": int(count),
                            "share": share,
                            "outside_extreme_fences": extreme,
                        },
                        rule="A common placeholder code (-1, -999, 9999, ...) repeated and extreme or >= 1% of rows.",
                        interpretation="Codes like this are often used to mean 'unknown' or 'not applicable'.",
                        suggestion=f"Confirm the meaning of {v:g}; treat it as missing if it is a code.",
                        confidence=0.7 if extreme else 0.5,
                    )
                )
            elif v != 0 and share >= t.value_spike_ratio:
                out.append(
                    self.finding(
                        "numeric.value_spike",
                        Severity.INFO,
                        f"Value spike at {v:g} in '{col}'",
                        f"{pct(share)} of values in '{col}' are exactly {v:g}, although the column has "
                        f"{int(s['distinct']):,} distinct values.",
                        columns=[col],
                        evidence={"value": v, "count": int(count), "share": share},
                        rule=f"One non-zero value >= value_spike_ratio ({t.value_spike_ratio}) of a column with >= 20 distinct values.",
                        interpretation="A default, cap, rounding convention or imputed value may be in use.",
                        suggestion=f"Check how {v:g} is produced.",
                        confidence=0.5,
                    )
                )
        if not reported:
            for v in SENTINELS:
                count = int((arr == v).sum())
                if count >= 2 and fences is not None and (v < fences[0] or v > fences[1]):
                    out.append(
                        self.finding(
                            "numeric.sentinel",
                            Severity.WARNING,
                            f"Possible sentinel value {v:g} in '{col}'",
                            f"The value {v:g} occurs {count:,} times in '{col}' and lies outside the "
                            f"extreme IQR fences ({fences[0]:.4g}, {fences[1]:.4g}).",
                            columns=[col],
                            evidence={"value": v, "count": count, "fences": list(fences)},
                            rule="A common placeholder code repeated outside the extreme IQR fences.",
                            interpretation="Codes like this are often used to mean 'unknown'.",
                            suggestion=f"Confirm the meaning of {v:g}.",
                            confidence=0.7,
                        )
                    )
        return out


@register_detector
class OutlierDetector(BaseDetector):
    name = "outliers"
    category = Category.NUMERIC
    description = (
        "Statistical outliers per numeric column with the configured methods (IQR, "
        "z-score, modified z-score, isolation forest) and multivariate outliers with an "
        "isolation forest when enabled."
    )
    assumptions = (
        "See each method's assumptions: z-score assumes near-normality; IQR and modified "
        "z-score are robust; isolation forest assumes anomalies are few and different."
    )
    limitations = (
        "A statistical outlier is not a data error. Skewed columns produce many IQR "
        "outliers by construction. Only domain rules can confirm invalid values."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        t = ctx.thresholds
        methods = list(dict.fromkeys(ctx.config.outlier_methods))
        uni = [m for m in methods if m != "isolation_forest"]
        out: list[Finding] = []
        for col in ctx.columns_of(ColumnType.NUMERIC):
            values = ctx.numeric(col)
            finite = values.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            n = int(finite.size)
            if n < 20 or not uni:
                continue
            # Right-skewed positive data (amounts, durations) is scored on the log scale so
            # its natural long tail is not reported as anomalous.
            log_scale = bool(finite.min() > 0 and float(sps.skew(finite)) >= 1.0)
            scored = np.log(values.where(values > 0)) if log_scale else values
            masks: dict[str, np.ndarray] = {
                m: ol.outlier_mask(
                    scored, m, iqr_k=t.iqr_k, zscore=t.zscore, modified_zscore=t.modified_zscore
                )
                for m in uni
            }
            stacked = np.vstack(list(masks.values()))
            any_mask = stacked.any(axis=0)
            if not any_mask.any():
                continue
            all_mask = stacked.all(axis=0)
            extreme = ol.iqr_mask(scored, t.iqr_extreme_k)
            share = float(any_mask.sum() / n)
            flagged = values[any_mask]
            bounds = ol.iqr_bounds(scored, t.iqr_k)
            if bounds is not None and log_scale:
                bounds = (float(np.exp(bounds[0])), float(np.exp(bounds[1])))
            sev = (
                Severity.WARNING
                if extreme.any() and share < t.outlier_warning_ratio
                else Severity.INFO
            )
            ctx.profile(col)["outliers"] = {m: int(v.sum()) for m, v in masks.items()}
            out.append(
                self.finding(
                    "outliers.univariate",
                    sev,
                    f"Statistical outliers in '{col}'",
                    f"{int(any_mask.sum()):,} value(s) ({pct(share)}) of '{col}' are flagged by at least one "
                    f"method ({', '.join(f'{m}: {int(v.sum()):,}' for m, v in masks.items())}); "
                    f"{int(extreme.sum()):,} lie beyond the extreme IQR fences"
                    + (" (scored on the log scale)." if log_scale else "."),
                    columns=[col],
                    evidence={
                        "log_scale": log_scale,
                        "per_method": {m: int(v.sum()) for m, v in masks.items()},
                        "flagged_by_all": int(all_mask.sum()),
                        "extreme": int(extreme.sum()),
                        "share": share,
                        "iqr_bounds": list(bounds) if bounds else None,
                        "flagged_min": float(flagged.min()),
                        "flagged_max": float(flagged.max()),
                        "row_examples": ctx.row_examples(extreme if extreme.any() else any_mask),
                    },
                    rule=(
                        f"Extreme values (beyond {t.iqr_extreme_k}x IQR) in a column where fewer than "
                        f"outlier_warning_ratio ({t.outlier_warning_ratio}) of values are flagged -> "
                        "warning (isolated anomalies); otherwise info (tail behaviour)."
                    ),
                    interpretation=(
                        "These are statistical outliers, not confirmed errors. Isolated extreme "
                        "values deserve a look; many moderate outliers usually reflect a skewed "
                        "or heavy-tailed distribution."
                    ),
                    suggestion="Review the extreme rows; add a domain rule if a valid range is known.",
                    confidence=int(all_mask.sum()) / max(int(any_mask.sum()), 1),
                )
            )
        if "isolation_forest" in methods:
            f = self._multivariate(ctx)
            if f:
                out.append(f)
        return out

    def _multivariate(self, ctx: Context) -> Finding | None:
        cols = ctx.columns_of(ColumnType.NUMERIC)[: ctx.config.performance.max_pairwise_columns]
        if len(cols) < 2:
            return None
        frame = pd.DataFrame({c: ctx.numeric(c) for c in cols})
        mask = ol.isolation_forest_mask(
            frame,
            ctx.thresholds.isolation_contamination,
            ctx.config.performance.random_state,
            ctx.config.performance.sample_rows,
        )
        if not mask.any():
            return None
        share = float(mask.mean())
        return self.finding(
            "outliers.multivariate",
            Severity.INFO,
            "Multivariate outliers (isolation forest)",
            f"{int(mask.sum()):,} row(s) ({pct(share)}) are isolated quickly by an isolation forest "
            f"over {len(cols)} numeric columns.",
            columns=cols,
            evidence={
                "rows": int(mask.sum()),
                "share": share,
                "row_examples": ctx.row_examples(mask),
            },
            rule=f"Isolation forest with contamination={ctx.thresholds.isolation_contamination}; always info.",
            interpretation=(
                "These rows have unusual combinations of values. The count depends directly on "
                "the contamination setting, so treat it as a ranking, not a verdict."
            ),
            suggestion="Review the listed rows for data-entry or join errors.",
            confidence=0.3,
        )
