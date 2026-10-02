from __future__ import annotations

import re

import numpy as np
import pandas as pd

from datasi.detectors._time import (
    TimeBins,
    change_confidence,
    change_date,
    fmt_time,
    points,
    strongest_change,
    time_bins,
)
from datasi.detectors.base import BaseDetector, Context, pct, register_detector
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType
from datasi.statistics import changepoint as cpm
from datasi.statistics import distribution as dist

PLACEHOLDER_DATES = {
    "1900-01-01",
    "1899-12-30",
    "1899-12-31",
    "1970-01-01",
    "2000-01-01",
    "2099-12-31",
    "2999-12-31",
    "9999-12-31",
}
_FUTURE_OK_RE = re.compile(
    r"expir|due|schedul|plan|forecast|end|until|valid|deadline|renew|next|target", re.I
)


@register_detector
class DatetimeDetector(BaseDetector):
    name = "datetime"
    category = Category.TEMPORAL
    description = (
        "Per-column date checks: unparseable values, future dates, placeholder-like dates "
        "(1900-01-01, 1970-01-01, 9999-12-31), mixed granularity, duplicate timestamps, "
        "coverage, empty periods and irregular sampling of the primary time column."
    )
    assumptions = (
        "'Now' is the run time unless reference_time is configured. Gaps are judged on "
        "periods with zero rows inside the observed range."
    )
    limitations = (
        "Columns whose names suggest planned/expiry dates are allowed to lie in the future. "
        "Timezone-naive values are assumed to share one timezone."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        out: list[Finding] = []
        for col in ctx.columns_of(ColumnType.DATETIME):
            raw = ctx.frame[col]
            t = ctx.datetimes(col)
            present = raw.notna()
            valid = t.dropna()
            if valid.empty:
                continue
            failed = present & t.isna()
            span_days = (valid.max() - valid.min()) / pd.Timedelta(days=1)
            has_time = bool((valid != valid.dt.normalize()).any())
            midnight_share = float((valid == valid.dt.normalize()).mean())
            ctx.profile(col)["datetime"] = {
                "min": valid.min(),
                "max": valid.max(),
                "span_days": span_days,
                "distinct": int(valid.nunique()),
                "granularity": _granularity(valid),
            }
            if failed.any():
                n = int(failed.sum())
                out.append(
                    self.finding(
                        "datetime.unparseable",
                        Severity.WARNING,
                        f"Unparseable dates in '{col}'",
                        f"{n:,} non-missing value(s) ({pct(n / int(present.sum()))}) of '{col}' could not be parsed as dates.",
                        columns=[col],
                        evidence={
                            "rows": n,
                            "row_examples": ctx.row_examples(failed),
                            "examples": ctx.examples(raw[failed].head(20)),
                        },
                        rule="Any non-missing value that fails date parsing in a date column -> warning.",
                        interpretation="Invalid dates (e.g. 2026-02-30), typos, or a different format.",
                        suggestion="Inspect the listed rows.",
                    )
                )
            future = t > ctx.now
            if future.any():
                n = int(future.sum())
                expected = bool(_FUTURE_OK_RE.search(col))
                out.append(
                    self.finding(
                        "datetime.future",
                        Severity.INFO if expected else Severity.WARNING,
                        f"Future dates in '{col}'",
                        f"{n:,} value(s) ({pct(n / len(valid))}) of '{col}' are after {ctx.now:%Y-%m-%d %H:%M} "
                        f"(latest: {valid.max():%Y-%m-%d}).",
                        columns=[col],
                        evidence={
                            "rows": n,
                            "max": valid.max(),
                            "reference_time": ctx.now,
                            "row_examples": ctx.row_examples(future),
                        },
                        rule="Dates after the reference time -> warning; info for names like expiry/due/scheduled.",
                        interpretation=(
                            "Plausible for planned or expiry dates."
                            if expected
                            else "For event timestamps this may indicate a clock, timezone, or entry error."
                        ),
                        suggestion="Check the reference time (reference_time) and the column's meaning.",
                        confidence=0.4 if expected else 0.6,
                    )
                )
            days = valid.dt.strftime("%Y-%m-%d")
            ph = days.isin(PLACEHOLDER_DATES) | (valid.dt.year < 1900)
            ph_counts = days[ph].value_counts()
            ph_counts = ph_counts[ph_counts >= max(2, int(0.0005 * len(valid)))]
            if not ph_counts.empty:
                n = int(ph_counts.sum())
                out.append(
                    self.finding(
                        "datetime.placeholder",
                        Severity.WARNING,
                        f"Placeholder-like dates in '{col}'",
                        f"{n:,} value(s) of '{col}' fall on dates commonly used as defaults: "
                        + ", ".join(f"{k} ({v:,})" for k, v in ph_counts.head(5).items())
                        + ".",
                        columns=[col],
                        evidence={"dates": {str(k): int(v) for k, v in ph_counts.items()}},
                        rule="Repeated epoch/default dates (1900-01-01, 1970-01-01, 9999-12-31, ...) or years < 1900 -> warning.",
                        interpretation="These are usually defaults standing in for unknown dates.",
                        suggestion="Treat them as missing if they are defaults.",
                        confidence=0.7,
                    )
                )
            if has_time and 0.5 <= midnight_share < 0.999 and len(valid) >= 50:
                out.append(
                    self.finding(
                        "datetime.mixed_granularity",
                        Severity.INFO,
                        f"Mixed date/time granularity in '{col}'",
                        f"{pct(midnight_share)} of values in '{col}' are exactly midnight while the rest carry a time of day.",
                        columns=[col],
                        evidence={"midnight_share": midnight_share},
                        rule="Between 50% and 99.9% of timestamps at 00:00:00 -> info.",
                        interpretation="Some sources may record dates only; time-of-day analyses would be biased.",
                    )
                )
            if col == ctx.time_column:
                out.extend(self._time_column(ctx, col, valid))
        return out

    def _time_column(self, ctx: Context, col: str, valid: pd.Series) -> list[Finding]:
        out: list[Finding] = []
        uniq = valid.drop_duplicates().sort_values()
        diffs = uniq.diff().dropna()
        dup_share = 1 - len(uniq) / len(valid)
        if len(diffs) >= 10:
            secs = diffs.dt.total_seconds().to_numpy()
            med = float(np.median(secs))
            cv = float(np.std(secs) / np.mean(secs)) if np.mean(secs) > 0 else 0.0
            ctx.profile(col)["datetime"]["median_interval_seconds"] = med
            regular = dup_share < 0.01 and float(np.mean(np.isclose(secs, med))) >= 0.8
            if regular:
                irregular = ~np.isclose(secs, med)
                gaps = secs > ctx.thresholds.gap_factor * med
                if irregular.any():
                    idx = np.flatnonzero(gaps)[:10]
                    out.append(
                        self.finding(
                            "datetime.irregular_interval",
                            Severity.WARNING if gaps.any() else Severity.INFO,
                            f"Irregular intervals in regularly sampled '{col}'",
                            f"'{col}' is sampled every {pd.Timedelta(seconds=med)} for {pct(1 - float(irregular.mean()))} "
                            f"of steps; {int(irregular.sum()):,} step(s) differ, {int(gaps.sum()):,} of them gaps "
                            f"longer than {ctx.thresholds.gap_factor:g}x the usual interval.",
                            columns=[col],
                            evidence={
                                "interval": str(pd.Timedelta(seconds=med)),
                                "irregular_steps": int(irregular.sum()),
                                "gaps": [
                                    {"from": uniq.iloc[i], "to": uniq.iloc[i + 1]} for i in idx
                                ],
                                "interval_cv": cv,
                            },
                            rule=f"Regular series (>= 80% identical steps); gaps > gap_factor ({ctx.thresholds.gap_factor}) x interval -> warning.",
                            interpretation="Missing periods may be outages or extraction gaps.",
                            suggestion="Check the source for the missing intervals.",
                            confidence=0.8,
                        )
                    )
        if dup_share > 0 and len(valid) >= 20:
            regular_like = len(diffs) >= 10 and dup_share < 0.05
            if regular_like:
                n = len(valid) - len(uniq)
                out.append(
                    self.finding(
                        "datetime.duplicate_timestamps",
                        Severity.INFO,
                        f"Repeated timestamps in '{col}'",
                        f"{n:,} row(s) ({pct(dup_share)}) share a timestamp with another row in '{col}', "
                        "which is otherwise almost unique.",
                        columns=[col],
                        evidence={"rows": n, "share": dup_share},
                        rule="Time column < 5% repeated timestamps -> info (repeats are unusual there).",
                        interpretation="Could be duplicated events or legitimately simultaneous events.",
                        suggestion="Check whether duplicated timestamps are duplicated records.",
                        confidence=0.4,
                    )
                )
        bins = time_bins(ctx)
        if bins is not None:
            counts = bins.counts()
            ctx.section("temporal").update(
                {"time_column": col, "period": bins.unit, "volume": points(counts, bins.unit)}
            )
            inner = counts.iloc[1:-1]
            empty = inner[inner == 0]
            if not empty.empty and float(counts.median()) >= 5:
                out.append(
                    self.finding(
                        "datetime.empty_periods",
                        Severity.WARNING,
                        f"Periods without any rows in '{col}'",
                        f"{len(empty):,} {bins.unit}(s) inside the observed range have no rows, while the "
                        f"median {bins.unit} has {int(counts.median()):,}.",
                        columns=[col],
                        evidence={
                            "periods": [fmt_time(k, bins.unit) for k in empty.index[:20]],
                            "median_rows": float(counts.median()),
                        },
                        rule=f"Zero-row {bins.unit}s between the first and last period when the median is >= 5 rows -> warning.",
                        interpretation="Likely extraction gaps or outages rather than a true absence of events.",
                        suggestion="Check the source for these periods.",
                        confidence=0.7,
                    )
                )
        return out


def _granularity(valid: pd.Series) -> str:
    if (valid == valid.dt.normalize()).all():
        return "day"
    if (valid.dt.second == 0).all() and (valid.dt.microsecond == 0).all():
        return "minute"
    if (valid.dt.microsecond == 0).all():
        return "second"
    return "sub-second"


@register_detector
class TemporalDetector(BaseDetector):
    name = "temporal"
    category = Category.TEMPORAL
    description = (
        "Change over time along the primary time column: volume shifts and spikes, level "
        "shifts in numeric columns, and categories that appear or disappear."
    )
    assumptions = (
        "One row per event; the primary time column orders the data meaningfully. Change "
        "points use CUSUM with binary segmentation on per-period aggregates."
    )
    limitations = (
        "Seasonality and trends violate the independence assumption behind the CUSUM "
        "threshold and may be reported as shifts. The first and last periods may be "
        "partial and are excluded from volume analysis."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        bins = time_bins(ctx)
        if bins is None:
            return []
        t = ctx.thresholds
        out: list[Finding] = []
        counts = bins.counts().astype(float)
        inner = counts.iloc[1:-1]
        cp = strongest_change(inner)
        if cp is not None and cp.mean_before > 0:
            rel = cp.shift / cp.mean_before
            if abs(rel) >= t.volume_change_ratio:
                when = change_date(inner, cp)
                word = "increased" if rel > 0 else "dropped"
                out.append(
                    self.finding(
                        "temporal.volume_shift",
                        Severity.WARNING,
                        f"Row volume {word} around {fmt_time(when, bins.unit)}",
                        f"Rows per {bins.unit} {word} from about {cp.mean_before:,.0f} to {cp.mean_after:,.0f} "
                        f"({rel:+.0%}) starting {fmt_time(when, bins.unit)}.",
                        columns=[bins.column],
                        evidence={
                            "change_at": when,
                            "before": cp.mean_before,
                            "after": cp.mean_after,
                            "relative_change": rel,
                            "cusum_statistic": cp.statistic,
                            "series": points(counts, bins.unit),
                        },
                        rule=f"Significant CUSUM change in rows/{bins.unit} with relative change >= volume_change_ratio ({t.volume_change_ratio}).",
                        interpretation="A source may have been added/removed, or collection changed; it may also be real growth or decline.",
                        suggestion="Compare with known pipeline or business changes around that date.",
                        confidence=change_confidence(cp, inner.size),
                    )
                )
        spikes = cpm.robust_spikes(inner.to_numpy())
        if spikes.size:
            labels = [fmt_time(inner.index[i], bins.unit) for i in spikes[:10]]
            out.append(
                self.finding(
                    "temporal.volume_spike",
                    Severity.INFO,
                    f"Unusual row counts in {spikes.size} {bins.unit}(s)",
                    f"Row counts in {', '.join(labels)} deviate from the local median by more than 5 robust SDs.",
                    columns=[bins.column],
                    evidence={
                        "periods": [
                            {"t": fmt_time(inner.index[i], bins.unit), "rows": float(inner.iloc[i])}
                            for i in spikes[:20]
                        ]
                    },
                    rule="|count - rolling median (7 periods)| > 5 x MAD-based SD -> info.",
                    interpretation="Bulk loads, duplicates, outages or genuine events can all cause spikes.",
                    suggestion="Check those periods for duplicated loads or missing data.",
                    confidence=0.5,
                )
            )
        out.extend(self._numeric_shifts(ctx, bins))
        out.extend(self._category_lifecycle(ctx, bins))
        return out

    def _numeric_shifts(self, ctx: Context, bins: TimeBins) -> list[Finding]:
        out: list[Finding] = []
        t = ctx.thresholds
        for col in ctx.columns_of(ColumnType.NUMERIC)[
            : ctx.config.performance.max_pairwise_columns
        ]:
            v = ctx.numeric(col)
            df = pd.DataFrame({"p": bins.labels, "v": v}).dropna()
            if df.empty:
                continue
            g = df.groupby("p")["v"].agg(["median", "count"]).reindex(bins.index)
            med = g["median"].where(g["count"] >= 5)
            cp = strongest_change(med)
            if cp is None:
                continue
            series = med.dropna()
            when = change_date(med, cp)
            before = v[(bins.labels < when).to_numpy()]
            after = v[(bins.labels >= when).to_numpy()]
            cmp = dist.compare_numeric(before, after)
            d = cmp.get("ks_statistic")
            if d is None or d < t.ks_high:
                continue
            out.append(
                self.finding(
                    "temporal.distribution_shift",
                    Severity.WARNING,
                    f"Distribution of '{col}' shifted around {fmt_time(when, bins.unit)}",
                    f"The {bins.unit}ly median of '{col}' moved from {cp.mean_before:.4g} to {cp.mean_after:.4g} "
                    f"starting {fmt_time(when, bins.unit)}; before vs after KS D = {d:.2f}.",
                    columns=[col, bins.column],
                    evidence={
                        "change_at": when,
                        "median_before": cp.mean_before,
                        "median_after": cp.mean_after,
                        "ks_statistic": d,
                        "psi": cmp.get("psi"),
                        "series": points(med, bins.unit),
                    },
                    rule=f"Significant CUSUM change in per-{bins.unit} medians and before/after KS D >= ks_high ({t.ks_high}).",
                    interpretation="A change in units, definitions, upstream logic or the population may have occurred.",
                    suggestion=f"Check whether the definition or source of '{col}' changed then.",
                    confidence=change_confidence(cp, series.size),
                )
            )
        return out

    def _category_lifecycle(self, ctx: Context, bins: TimeBins) -> list[Finding]:
        out: list[Finding] = []
        n_periods = len(bins.index)
        if n_periods < ctx.thresholds.temporal_min_periods:
            return out
        early_end = bins.index[max(1, n_periods // 5)]
        late_start = bins.index[min(n_periods - 1, n_periods - n_periods // 5 - 1)]
        for col in ctx.columns_of(ColumnType.CATEGORICAL):
            if ctx.schema[col].n_unique > ctx.config.performance.max_group_levels * 4:
                continue
            df = pd.DataFrame({"p": bins.labels, "c": ctx.frame[col]}).dropna()
            if df.empty:
                continue
            first = df.groupby("c", observed=True)["p"].min()
            last = df.groupby("c", observed=True)["p"].max()
            share = df["c"].value_counts(normalize=True)
            new = [k for k in first.index if first[k] > early_end and share[k] >= 0.01]
            gone = [k for k in last.index if last[k] < late_start and share[k] >= 0.01]
            if new:
                out.append(
                    self.finding(
                        "temporal.new_category",
                        Severity.INFO,
                        f"New categories appear in '{col}' over time",
                        f"{len(new)} label(s) of '{col}' first appear after {fmt_time(early_end, bins.unit)}: "
                        + ", ".join(
                            f"'{ctx.label(k)}' (from {fmt_time(first[k], bins.unit)})"
                            for k in new[:5]
                        )
                        + ".",
                        columns=[col, bins.column],
                        evidence={
                            "labels": [
                                {
                                    "label": ctx.label(k),
                                    "first_seen": first[k],
                                    "share": float(share[k]),
                                }
                                for k in new[:20]
                            ]
                        },
                        rule="A label with >= 1% of rows whose first occurrence is after the first 20% of periods.",
                        interpretation="New codes may come from a new source, product, or a recoding.",
                        suggestion="Make sure downstream mappings and models know these labels.",
                        confidence=0.6,
                    )
                )
            if gone:
                out.append(
                    self.finding(
                        "temporal.vanished_category",
                        Severity.INFO,
                        f"Categories disappear from '{col}' over time",
                        f"{len(gone)} label(s) of '{col}' are not seen after {fmt_time(late_start, bins.unit)}: "
                        + ", ".join(
                            f"'{ctx.label(k)}' (last {fmt_time(last[k], bins.unit)})"
                            for k in gone[:5]
                        )
                        + ".",
                        columns=[col, bins.column],
                        evidence={
                            "labels": [
                                {
                                    "label": ctx.label(k),
                                    "last_seen": last[k],
                                    "share": float(share[k]),
                                }
                                for k in gone[:20]
                            ]
                        },
                        rule="A label with >= 1% of rows whose last occurrence is before the final 20% of periods.",
                        interpretation="A code may have been retired or renamed, or a source dropped.",
                        suggestion="Check for renamed codes (see similar-label findings).",
                        confidence=0.6,
                    )
                )
        return out
