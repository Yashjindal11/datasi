from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from datasi.detectors._time import (
    change_confidence,
    change_date,
    fmt_time,
    points,
    strongest_change,
    time_bins,
)
from datasi.detectors.base import BaseDetector, Context, pct, register_detector, tiered
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType
from datasi.statistics.association import cramers_v, phi_coefficient

MIN_MISSING_FOR_PATTERNS = 10


@register_detector
class MissingnessDetector(BaseDetector):
    name = "missingness"
    category = Category.MISSINGNESS
    description = (
        "Measures missing values per column and row, then looks for structure in the "
        "missingness: concentration in groups, change over time, columns that go missing "
        "together, and association with other numeric variables."
    )
    assumptions = (
        "Missing means null/NaN/None after loading. Placeholder strings such as 'N/A' "
        "are reported separately by the strings detector."
    )
    limitations = (
        "Structure in missingness suggests data are not missing completely at random but "
        "cannot prove the mechanism. Group and time comparisons use effect sizes on the "
        "observed data and may reflect confounding."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        frame = ctx.frame
        if ctx.n_rows == 0:
            return []
        t = ctx.thresholds
        isna = frame.isna()
        miss = isna.sum()
        ratios = miss / ctx.n_rows
        ctx.section("missingness").update(
            {
                "columns": {c: {"count": int(miss[c]), "ratio": float(ratios[c])} for c in frame},
                "total_cells": int(frame.size),
                "missing_cells": int(miss.sum()),
            }
        )
        out: list[Finding] = []
        some = [c for c in frame if 0 < ratios[c] < t.missing_warning]
        if some:
            out.append(
                self.finding(
                    "missing.low",
                    Severity.INFO,
                    f"{len(some)} column(s) have a small share of missing values",
                    f"{len(some)} column(s) have missing values below {pct(t.missing_warning)}.",
                    columns=some,
                    evidence={c: float(ratios[c]) for c in some[:50]},
                    rule=f"0 < missing ratio < missing_warning ({t.missing_warning})",
                )
            )
        for col in frame:
            r = float(ratios[col])
            if r < t.missing_warning:
                continue
            sev = tiered(r, [(t.missing_high, Severity.HIGH)], Severity.WARNING)
            title = (
                f"Column '{col}' is entirely missing"
                if r == 1.0
                else f"High missingness in '{col}'"
            )
            out.append(
                self.finding(
                    "missing.complete" if r == 1.0 else "missing.high",
                    sev,
                    title,
                    f"{int(miss[col]):,} of {ctx.n_rows:,} values ({pct(r)}) are missing.",
                    columns=[col],
                    evidence={"missing": int(miss[col]), "rows": ctx.n_rows, "ratio": r},
                    rule=(
                        f"missing ratio >= missing_high ({t.missing_high}) -> high; "
                        f">= missing_warning ({t.missing_warning}) -> warning"
                    ),
                    suggestion=(
                        "Confirm whether the field is populated upstream; see related "
                        "missingness-pattern findings for where the gaps concentrate."
                    ),
                )
            )
        out.extend(self._rows(ctx, isna))
        candidates = [
            c
            for c in frame
            if miss[c] >= MIN_MISSING_FOR_PATTERNS and 0 < ratios[c] < 1 and miss[c] < ctx.n_rows
        ]
        for col in candidates:
            indicator = isna[col]
            f = self._by_group(ctx, col, indicator)
            if f:
                out.append(f)
            f = self._over_time(ctx, col, indicator)
            if f:
                out.append(f)
            f = self._related_numeric(ctx, col, indicator)
            if f:
                out.append(f)
        out.extend(self._co_missing(ctx, isna, candidates))
        return out

    def _rows(self, ctx: Context, isna: pd.DataFrame) -> list[Finding]:
        if ctx.frame.shape[1] < 2:
            return []
        per_row = isna.mean(axis=1)
        empty = per_row == 1.0
        mostly = (per_row >= 0.5) & ~empty
        out = []
        if empty.any():
            out.append(
                self.finding(
                    "missing.empty_rows",
                    Severity.WARNING,
                    "Rows with every value missing",
                    f"{int(empty.sum()):,} row(s) ({pct(float(empty.mean()))}) contain no values.",
                    evidence={"rows": int(empty.sum()), "row_examples": ctx.row_examples(empty)},
                    rule="Any fully empty row is a warning.",
                    interpretation="Often trailing blank lines or failed record extraction.",
                    suggestion="Check the export for blank lines or failed joins.",
                    confidence=0.7,
                )
            )
        if mostly.mean() >= 0.01:
            out.append(
                self.finding(
                    "missing.sparse_rows",
                    Severity.INFO,
                    "Rows that are mostly empty",
                    f"{int(mostly.sum()):,} row(s) ({pct(float(mostly.mean()))}) are missing at "
                    "least half of their values.",
                    evidence={"rows": int(mostly.sum()), "row_examples": ctx.row_examples(mostly)},
                    rule="Reported when >= 1% of rows are at least 50% empty.",
                )
            )
        return out

    def _by_group(self, ctx: Context, col: str, indicator: pd.Series) -> Finding | None:
        t = ctx.thresholds
        max_levels = ctx.config.performance.max_group_levels
        best: tuple[float, str, pd.DataFrame] | None = None
        overall = float(indicator.mean())
        for g in ctx.columns_of(ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            if g == col:
                continue
            nun = ctx.schema[g].n_unique
            if nun < 2 or nun > max_levels:
                continue
            groups = ctx.frame[g].astype("string").fillna("<missing>")
            stats = (
                pd.DataFrame({"g": groups, "m": indicator})
                .groupby("g", observed=True)["m"]
                .agg(["mean", "size"])
            )
            stats = stats[stats["size"] >= max(10, 0.01 * ctx.n_rows)]
            if len(stats) < 2:
                continue
            total_missing = indicator.sum()
            rest_rate = (total_missing - stats["mean"] * stats["size"]) / (
                ctx.n_rows - stats["size"]
            ).replace(0, np.nan)
            gap = (stats["mean"] - rest_rate).abs().max()
            if not np.isfinite(gap) or gap < t.missing_group_min_gap:
                continue
            v = cramers_v(indicator, groups)
            if v is None:
                continue
            if best is None or v > best[0]:
                stats = stats.assign(rest=rest_rate)
                best = (v, g, stats)
        if best is None:
            return None
        v, g, stats = best
        top = stats.sort_values("mean", ascending=False).head(5)
        levels = [
            {"group": ctx.label(k), "missing_ratio": float(r["mean"]), "rows": int(r["size"])}
            for k, r in top.iterrows()
        ]
        hi = levels[0]
        hi_ratio = float(top["mean"].iloc[0])
        return self.finding(
            "missing.by_group",
            Severity.WARNING,
            f"Missingness in '{col}' depends on '{g}'",
            f"'{col}' is missing in {pct(hi_ratio)} of rows where {g} = "
            f"'{hi['group']}', versus {pct(overall)} overall (Cramér's V = {v:.2f}).",
            columns=[col, g],
            evidence={
                "group_column": g,
                "cramers_v": v,
                "overall_ratio": overall,
                "groups": levels,
            },
            rule=(
                "A group's missing rate differs from the rest of the data by at least "
                f"missing_group_min_gap ({t.missing_group_min_gap}) for groups with >= 1% of rows."
            ),
            interpretation=(
                "Missingness is not uniform across groups, so the data are probably not "
                "missing completely at random. The collection process may differ between groups."
            ),
            suggestion=(
                f"Check how '{col}' is collected for each '{g}' value before imputing with a "
                "single global statistic."
            ),
            confidence=min(0.9, 0.5 + v / 2),
        )

    def _over_time(self, ctx: Context, col: str, indicator: pd.Series) -> Finding | None:
        bins = time_bins(ctx)
        if bins is None or col == bins.column:
            return None
        rates = bins.rate(indicator)
        cp = strongest_change(rates)
        if cp is None or abs(cp.shift) < ctx.thresholds.temporal_shift_gap:
            return None
        when = change_date(rates, cp)
        direction = "increased" if cp.shift > 0 else "decreased"
        return self.finding(
            "missing.over_time",
            Severity.WARNING,
            f"Missingness in '{col}' {direction} around {fmt_time(when, bins.unit)}",
            f"The share of missing '{col}' values {direction} from {pct(cp.mean_before)} to "
            f"{pct(cp.mean_after)} per {bins.unit}, with the change starting at "
            f"{fmt_time(when, bins.unit)} (by '{bins.column}').",
            columns=[col, bins.column],
            evidence={
                "time_column": bins.column,
                "period": bins.unit,
                "change_at": when,
                "rate_before": cp.mean_before,
                "rate_after": cp.mean_after,
                "cusum_statistic": cp.statistic,
                "series": points(rates, bins.unit),
            },
            rule=(
                "Significant CUSUM change point (1% level, independence assumed) with an "
                f"absolute shift >= temporal_shift_gap ({ctx.thresholds.temporal_shift_gap})."
            ),
            interpretation=(
                "A sudden change in missingness often coincides with a change in the upstream "
                "source, form, or pipeline. It could also reflect a genuine change in the population."
            ),
            suggestion=f"Check whether the upstream source for '{col}' changed around that date.",
            confidence=change_confidence(cp, int(rates.notna().sum())),
        )

    def _related_numeric(self, ctx: Context, col: str, indicator: pd.Series) -> Finding | None:
        best: tuple[float, str, dict[str, Any]] | None = None
        rows = ctx.sample().index
        miss_mask = indicator.loc[rows].to_numpy()
        for other in ctx.columns_of(ColumnType.NUMERIC):
            if other == col:
                continue
            v = ctx.numeric(other).loc[rows].to_numpy()
            a, b = v[miss_mask], v[~miss_mask]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size < 30 or b.size < 30:
                continue
            d = float(sps.ks_2samp(a, b).statistic)
            if d >= 0.3 and (best is None or d > best[0]):
                best = (
                    d,
                    other,
                    {
                        "median_when_missing": float(np.median(a)),
                        "median_when_present": float(np.median(b)),
                    },
                )
        if best is None:
            return None
        d, other, ev = best
        return self.finding(
            "missing.related_numeric",
            Severity.INFO,
            f"Missingness in '{col}' is associated with '{other}'",
            f"The distribution of '{other}' differs between rows where '{col}' is missing and "
            f"where it is present (KS D = {d:.2f}; medians {ev['median_when_missing']:.4g} vs "
            f"{ev['median_when_present']:.4g}).",
            columns=[col, other],
            evidence={"related_column": other, "ks_statistic": d, **ev},
            rule="Two-sample KS statistic >= 0.30 with at least 30 rows on each side.",
            interpretation=(
                f"Whether '{col}' is recorded may depend on '{other}' (missing at random "
                "conditional on it), or both may share a common cause."
            ),
            suggestion=f"Consider '{other}' when modelling or imputing '{col}'.",
            confidence=min(0.8, 0.3 + d),
        )

    def _co_missing(self, ctx: Context, isna: pd.DataFrame, cols: list[str]) -> list[Finding]:
        cols = cols[: ctx.config.performance.max_pairwise_columns]
        if len(cols) < 2:
            return []
        thr = ctx.thresholds.missing_cooccurrence_phi
        ind = isna[cols].to_numpy()
        parent = {c: c for c in cols}

        def find(c: str) -> str:
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c

        pairs: list[dict[str, Any]] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                phi = phi_coefficient(ind[:, i], ind[:, j])
                if phi is not None and phi >= thr:
                    pairs.append({"a": cols[i], "b": cols[j], "phi": phi})
                    parent[find(cols[i])] = find(cols[j])
        clusters: dict[str, list[str]] = {}
        for c in cols:
            if any(c in (p["a"], p["b"]) for p in pairs):
                clusters.setdefault(find(c), []).append(c)
        out = []
        for members in clusters.values():
            both = isna[members].all(axis=1)
            mp = [p for p in pairs if p["a"] in members and p["b"] in members]
            out.append(
                self.finding(
                    "missing.co_occurrence",
                    Severity.INFO,
                    f"{len(members)} columns tend to be missing together",
                    f"Columns {members} are missing in the same rows (phi >= {thr}); "
                    f"{int(both.sum()):,} row(s) miss all of them.",
                    columns=members,
                    evidence={"pairs": mp, "rows_missing_all": int(both.sum())},
                    rule=f"Pairwise phi coefficient of missing indicators >= missing_cooccurrence_phi ({thr}).",
                    interpretation=(
                        "Fields that go missing together often come from the same source "
                        "table, form section, or join."
                    ),
                    suggestion="Check whether a join or source system supplies these fields.",
                    confidence=0.6,
                )
            )
        if pairs:
            ctx.section("missingness")["co_missing_pairs"] = pairs
        return out
