from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from datasi.detectors.base import BaseDetector, Context, register_detector
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType
from datasi.statistics import association as assoc

MI_ROWS = 50_000
EXACT_MIN = 0.999
BROKEN_MIN = 0.95


def _numeric_frame(ctx: Context, sample: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: ctx.numeric(c).loc[sample.index] for c in cols})


def _feature_columns(ctx: Context) -> tuple[list[str], list[str]]:
    limit = ctx.config.performance.max_pairwise_columns
    num = [c for c in ctx.schema.columns if ctx.is_numeric_like(c)]
    num = [c for c in num if ctx.schema[c].n_unique > 1][:limit]
    cats = [
        c
        for c in ctx.columns_of(ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
        if c not in num
        and 1 < ctx.schema[c].n_unique <= ctx.config.performance.max_group_levels * 4
    ][:limit]
    return num, cats


@register_detector
class CorrelationDetector(BaseDetector):
    name = "correlation"
    category = Category.RELATIONSHIPS
    description = (
        "Pairwise association: Pearson and Spearman for numeric pairs, bias-corrected "
        "Cramér's V for categorical pairs, the correlation ratio for categorical-numeric "
        "pairs, and normalised mutual information to surface strong non-linear dependence."
    )
    assumptions = (
        "Pairs are computed on rows where both values are present, on a deterministic "
        "sample when the dataset exceeds performance.sample_rows."
    )
    limitations = (
        "Correlation is not causation and says nothing about which variable drives the "
        "other. Mutual information on discretised data is biased upward for many levels."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        t = ctx.thresholds
        sample = ctx.sample()
        num, cats = _feature_columns(ctx)
        out: list[Finding] = []
        relationships: list[dict[str, Any]] = []
        section = ctx.section("correlation")
        section["sampled_rows"] = len(sample)
        if len(num) >= 2:
            frame = _numeric_frame(ctx, sample, num)
            pear = assoc.correlation_matrix(frame, "pearson")
            spear = assoc.correlation_matrix(frame, "spearman")
            section["numeric_columns"] = num
            section["pearson"] = pear.round(4).to_numpy().tolist()
            section["spearman"] = spear.round(4).to_numpy().tolist()
            for a, b in itertools.combinations(num, 2):
                r, rho = pear.loc[a, b], spear.loc[a, b]
                if not (np.isfinite(r) or np.isfinite(rho)):
                    continue
                best = max(abs(r) if np.isfinite(r) else 0, abs(rho) if np.isfinite(rho) else 0)
                if best < t.correlation_warning:
                    continue
                relationships.append(
                    {"a": a, "b": b, "kind": "correlation", "pearson": r, "spearman": rho}
                )
                if best >= 0.9999:
                    continue  # reported as a deterministic relationship by the redundancy detector
                sev = Severity.WARNING if best >= t.correlation_high else Severity.INFO
                out.append(
                    self.finding(
                        "correlation.numeric",
                        sev,
                        f"'{a}' and '{b}' are highly correlated",
                        f"Pearson r = {r:.3f}, Spearman rho = {rho:.3f} between '{a}' and '{b}'.",
                        columns=[a, b],
                        evidence={
                            "pearson": r,
                            "spearman": rho,
                            "rows": int(frame[[a, b]].dropna().shape[0]),
                        },
                        rule=(
                            f"max(|r|, |rho|) >= correlation_high ({t.correlation_high}) -> warning; "
                            f">= correlation_warning ({t.correlation_warning}) -> info."
                        ),
                        interpretation=(
                            "The columns carry largely overlapping information (possible redundancy). "
                            "This does not imply either causes the other."
                        ),
                        suggestion="Consider keeping one, or combining them, for models sensitive to collinearity.",
                        confidence=0.9,
                    )
                )
        if len(cats) >= 2:
            sub = sample[cats]
            for a, b in itertools.combinations(cats, 2):
                v = assoc.cramers_v(sub[a], sub[b])
                if v is None or v < t.cramers_v_warning:
                    continue
                relationships.append(
                    {"a": a, "b": b, "kind": "categorical_association", "cramers_v": v}
                )
                out.append(
                    self.finding(
                        "correlation.categorical",
                        Severity.INFO,
                        f"'{a}' and '{b}' are strongly associated",
                        f"Bias-corrected Cramér's V = {v:.3f} between '{a}' and '{b}'.",
                        columns=[a, b],
                        evidence={"cramers_v": v},
                        rule=f"Cramér's V >= cramers_v_warning ({t.cramers_v_warning}) -> info.",
                        interpretation="One category largely determines the other (e.g. code and name, city and region).",
                        suggestion="Check whether both are needed.",
                        confidence=0.8,
                    )
                )
        for c in cats:
            for n in num:
                eta = assoc.correlation_ratio(sample[c], ctx.numeric(n).loc[sample.index])
                if eta is not None and eta >= t.correlation_high:
                    relationships.append({"a": c, "b": n, "kind": "correlation_ratio", "eta": eta})
                    out.append(
                        self.finding(
                            "correlation.mixed",
                            Severity.INFO,
                            f"'{c}' almost determines '{n}'",
                            f"Correlation ratio eta = {eta:.3f}: '{c}' explains {eta**2:.1%} of the variance of '{n}'.",
                            columns=[c, n],
                            evidence={"eta": eta},
                            rule=f"eta >= correlation_high ({t.correlation_high}) -> info.",
                            interpretation=f"'{n}' may be a per-'{c}' constant or lookup value.",
                            confidence=0.8,
                        )
                    )
        out.extend(self._nonlinear(ctx, sample, num, relationships))
        section["relationships"] = relationships
        return out

    def _nonlinear(
        self, ctx: Context, sample: pd.DataFrame, num: list[str], rels: list[dict[str, Any]]
    ) -> list[Finding]:
        if len(num) < 2:
            return []
        sub = sample.head(MI_ROWS) if len(sample) > MI_ROWS else sample
        frame = _numeric_frame(ctx, sub, num[:30])
        spear = frame.corr(method="spearman", min_periods=10)
        out = []
        for a, b in itertools.combinations(frame.columns, 2):
            rho = spear.loc[a, b]
            if np.isfinite(rho) and abs(rho) >= 0.5:
                continue
            nmi = assoc.normalized_mutual_information(frame[a], frame[b])
            if nmi is None or nmi < 0.5:
                continue
            rels.append({"a": a, "b": b, "kind": "nonlinear", "nmi": nmi, "spearman": rho})
            out.append(
                self.finding(
                    "correlation.nonlinear",
                    Severity.INFO,
                    f"Non-linear dependence between '{a}' and '{b}'",
                    f"Normalised mutual information = {nmi:.2f} while Spearman rho = {rho:.2f}.",
                    columns=[a, b],
                    evidence={"nmi": nmi, "spearman": rho},
                    rule="NMI (10 quantile bins) >= 0.5 with |rho| < 0.5 -> info.",
                    interpretation="The columns are strongly related but not monotonically (e.g. U-shaped or periodic).",
                    confidence=0.5,
                )
            )
        return out


@register_detector
class RedundancyDetector(BaseDetector):
    name = "redundancy"
    category = Category.RELATIONSHIPS
    description = (
        "Deterministic relationships: duplicated columns, exact linear transformations "
        "(unit conversions such as height_m = 0.01 x height_cm), sums/differences of two "
        "other columns (total = x + y), and one-to-one categorical encodings."
    )
    assumptions = (
        "An identity is deterministic if it holds for >= 99.9% of complete rows within a "
        "relative tolerance of 1e-6; if it holds for 95-99.9% the breaking rows are reported "
        "as likely errors. Lines are fitted robustly; sums are screened on a sample and "
        "verified on all rows."
    )
    limitations = (
        "Only linear forms with up to two inputs are searched; ratios and products are not. "
        "A deterministic relationship may be intended (derived columns)."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        out: list[Finding] = []
        num, cats = _feature_columns(ctx)
        num = [c for c in num if ctx.schema[c].column_type == ColumnType.NUMERIC][:40]
        full = pd.DataFrame({c: ctx.numeric(c) for c in num})
        screen = full.sample(min(len(full), 5000), random_state=0) if len(full) else full
        rho = screen.corr(method="spearman", min_periods=20) if len(num) >= 2 else None
        reported: set[frozenset[str]] = set()
        for a, b in itertools.combinations(num, 2):
            pair = full[[a, b]].dropna()
            if len(pair) < 20:
                continue
            x, y = pair[a].to_numpy(), pair[b].to_numpy()
            if np.array_equal(x, y):
                reported.add(frozenset((a, b)))
                out.append(self._identical(a, b, len(pair)))
                continue
            if x.std() == 0 or y.std() == 0 or rho is None or not abs(rho.loc[a, b]) >= 0.95:
                continue
            slope, intercept, ok = _robust_line(x, y)
            share = float(ok.mean())
            if share < BROKEN_MIN:
                continue
            reported.add(frozenset((a, b)))
            bad_rows = pair.index[~ok]
            out.append(
                self._relation(
                    "linear",
                    [a, b],
                    f"'{b}' = {slope:.6g} x '{a}' + {intercept:.6g}",
                    share,
                    len(pair),
                    [int(i) for i in bad_rows[:10]],
                    {"slope": slope, "intercept": intercept},
                    "One column is probably derived from the other (e.g. a unit conversion).",
                )
            )
        out.extend(self._sums(full, screen, num, reported))
        out.extend(self._bijections(ctx, cats))
        return out

    def _relation(
        self,
        kind: str,
        columns: list[str],
        formula: str,
        share: float,
        rows: int,
        breaking: list[int],
        extra: dict[str, Any],
        meaning: str,
    ) -> Finding:
        exact = share >= EXACT_MIN
        n_break = round((1 - share) * rows)
        evidence = {"formula": formula, "holds_share": share, "rows": rows, **extra}
        if exact:
            return self.finding(
                f"redundancy.{kind}",
                Severity.WARNING if kind == "linear" else Severity.INFO,
                f"{formula} (deterministic)",
                f"The identity {formula} holds for {share:.2%} of {rows:,} complete rows.",
                columns=columns,
                evidence=evidence,
                rule=(
                    f"Identity holds (relative tolerance 1e-6) on >= {EXACT_MIN:.1%} of rows -> "
                    + (
                        "warning (redundant transformation)."
                        if kind == "linear"
                        else "info (derived total)."
                    )
                ),
                interpretation=meaning,
                suggestion="Keep one representation for modelling; derived columns add no information.",
                confidence=0.95,
            )
        return self.finding(
            f"redundancy.{kind}_broken",
            Severity.WARNING,
            f"{formula} holds except in {n_break:,} rows",
            f"The identity {formula} holds for {share:.2%} of {rows:,} complete rows; "
            f"{n_break:,} row(s) break it.",
            columns=columns,
            evidence={**evidence, "breaking_rows": n_break, "row_examples": breaking},
            rule=f"Identity holds on >= {BROKEN_MIN:.0%} but < {EXACT_MIN:.1%} of rows -> warning.",
            interpretation=(
                "The columns are almost certainly derived from each other, so the rows that "
                "break the identity are likely data errors in one of the columns."
            ),
            suggestion="Inspect the breaking rows to see which column is wrong.",
            confidence=0.8,
        )

    def _identical(self, a: str, b: str, rows: int) -> Finding:
        return self.finding(
            "redundancy.identical",
            Severity.WARNING,
            f"'{a}' and '{b}' are identical",
            f"'{a}' and '{b}' contain the same value on all {rows:,} rows where both are present.",
            columns=[a, b],
            evidence={"rows": rows},
            rule="Identical values on every complete row -> warning.",
            interpretation="Probably a duplicated column (e.g. from a join).",
            suggestion="Drop one of them.",
            confidence=0.95,
        )

    def _sums(
        self,
        full: pd.DataFrame,
        screen: pd.DataFrame,
        num: list[str],
        reported: set[frozenset[str]],
    ) -> list[Finding]:
        out: list[Finding] = []
        cols = num[:25]
        if len(cols) < 3:
            return out
        arr = screen[cols].to_numpy(dtype=float)
        for k, z in enumerate(cols):
            for i, j in itertools.combinations(range(len(cols)), 2):
                if k in (i, j):
                    continue
                for sign, op in ((1.0, "+"), (-1.0, "-")):
                    pred = arr[:, i] + sign * arr[:, j]
                    ok = np.isfinite(pred) & np.isfinite(arr[:, k])
                    if ok.sum() < 20 or _holds(arr[ok, k], pred[ok]) < BROKEN_MIN:
                        continue
                    a, b = cols[i], cols[j]
                    key = frozenset((z, a, b))
                    if key in reported:
                        continue
                    sub = full[[z, a, b]].dropna()
                    hold = _holds_mask(sub[z].to_numpy(), (sub[a] + sign * sub[b]).to_numpy())
                    share = float(hold.mean())
                    if share < BROKEN_MIN:
                        continue
                    reported.add(key)
                    out.append(
                        self._relation(
                            "sum",
                            [z, a, b],
                            f"'{z}' = '{a}' {op} '{b}'",
                            share,
                            len(sub),
                            [int(i) for i in sub.index[~hold][:10]],
                            {"operation": op},
                            "A derived total/difference column; expected in many schemas but redundant for modelling.",
                        )
                    )
        return out

    def _bijections(self, ctx: Context, cats: list[str]) -> list[Finding]:
        out: list[Finding] = []
        cats = [c for c in cats if ctx.schema[c].n_unique >= 3][:30]
        for a, b in itertools.combinations(cats, 2):
            pair = ctx.frame[[a, b]].dropna()
            if len(pair) < 20:
                continue
            ab = pair.groupby(a, observed=True)[b].nunique()
            if (ab > 1).any():
                continue
            ba = pair.groupby(b, observed=True)[a].nunique()
            if (ba > 1).any():
                continue
            out.append(
                self.finding(
                    "redundancy.one_to_one",
                    Severity.INFO,
                    f"'{a}' and '{b}' encode the same information",
                    f"Every '{a}' value maps to exactly one '{b}' value and vice versa ({int(ab.size):,} pairs).",
                    columns=[a, b],
                    evidence={"levels": int(ab.size)},
                    rule="Perfect one-to-one mapping between two categorical columns -> info.",
                    interpretation="Typically a code and its label (e.g. country_code and country_name).",
                    suggestion="Keep one for modelling; keep both only for readability.",
                    confidence=0.9,
                )
            )
        return out


def _holds_mask(actual: np.ndarray, predicted: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    scale = np.maximum(np.abs(actual), np.abs(predicted))
    scale = np.where(scale == 0, 1.0, scale)
    ok: np.ndarray = np.abs(actual - predicted) <= rtol * scale + 1e-9
    return ok


def _holds(actual: np.ndarray, predicted: np.ndarray, rtol: float = 1e-6) -> float:
    ok = _holds_mask(actual, predicted, rtol)
    return float(ok.mean()) if ok.size else 0.0


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least-squares line refitted on the best-fitting 90% of rows, so a few corrupted
    rows do not hide an otherwise exact relationship. Returns slope, intercept and the
    mask of rows on which the line holds exactly."""
    keep = np.ones(x.size, dtype=bool)
    slope, intercept = 0.0, 0.0
    for _ in range(3):
        slope, intercept = (float(v) for v in np.polyfit(x[keep], y[keep], 1))
        resid = np.abs(y - (slope * x + intercept))
        keep = resid <= np.quantile(resid, 0.9)
        if keep.sum() < 10:
            break
    # Snap near-round coefficients (0.0100000003 -> 0.01) if that fits at least as well.
    ok = _holds_mask(y, slope * x + intercept)
    s_round = float(f"{slope:.6g}")
    i_round = float(f"{intercept:.6g}") if abs(intercept) > 1e-9 else 0.0
    ok_round = _holds_mask(y, s_round * x + i_round)
    if ok_round.mean() >= ok.mean():
        return s_round, i_round, ok_round
    return slope, intercept, ok
