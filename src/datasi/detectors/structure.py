from __future__ import annotations

from datasi.detectors.base import BaseDetector, Context, pct, register_detector
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType


@register_detector
class ConstantDetector(BaseDetector):
    name = "constant"
    category = Category.SCHEMA
    description = (
        "Columns with a single value, near-constant columns, and categories that dominate a column."
    )
    assumptions = "Variation is measured over non-missing values."
    limitations = (
        "A near-constant column may be a legitimate rare-event flag (e.g. fraud = 1 in "
        "0.1% of rows); it is never called useless."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        t = ctx.thresholds
        out: list[Finding] = []
        for col, c in ctx.schema.columns.items():
            if c.n_present == 0:
                continue
            if c.column_type == ColumnType.CONSTANT:
                value = ctx.counts(col).index[0] if not ctx.counts(col).empty else None
                shown = ctx.label(value) if value is not None and len(str(value)) <= 40 else None
                out.append(
                    self.finding(
                        "constant.single_value",
                        Severity.WARNING,
                        f"Column '{col}' is constant",
                        f"All {c.n_present:,} non-missing values of '{col}' are identical"
                        + (f" ('{shown}')." if shown else ".")
                        + (f" {pct(c.missing_ratio)} of rows are missing." if c.n_missing else ""),
                        columns=[col],
                        evidence={
                            "value": shown,
                            "non_missing": c.n_present,
                            "missing": c.n_missing,
                        },
                        rule="Exactly one distinct non-missing value -> warning.",
                        interpretation=(
                            "The column carries no information within this dataset; it may be a "
                            "filter artefact (data pre-filtered on this field) or an unused field."
                            + (
                                " Missingness itself may still be informative."
                                if c.n_missing
                                else ""
                            )
                        ),
                        suggestion="Check whether the dataset was filtered on this column.",
                        confidence=0.9,
                    )
                )
                continue
            if c.column_type in (ColumnType.IDENTIFIER, ColumnType.TEXT, ColumnType.UNKNOWN):
                continue
            counts = ctx.frame[col].value_counts(dropna=True)
            if counts.empty:
                continue
            share = float(counts.iloc[0] / c.n_present)
            label = ctx.label(counts.index[0])
            if share >= t.near_constant_ratio:
                sev, code, title = (
                    Severity.WARNING,
                    "constant.near",
                    f"Column '{col}' is nearly constant",
                )
                rule = (
                    f"Top value share >= near_constant_ratio ({t.near_constant_ratio}) -> warning."
                )
            elif share >= t.dominant_category_ratio and c.column_type in (
                ColumnType.CATEGORICAL,
                ColumnType.BOOLEAN,
            ):
                sev, code, title = (
                    Severity.INFO,
                    "constant.dominant",
                    f"One value dominates '{col}'",
                )
                rule = f"Top value share >= dominant_category_ratio ({t.dominant_category_ratio}) -> info."
            else:
                continue
            out.append(
                self.finding(
                    code,
                    sev,
                    title,
                    f"{pct(share)} of non-missing values of '{col}' are '{label}' "
                    f"({c.n_unique:,} distinct values in total).",
                    columns=[col],
                    evidence={
                        "top_value": label,
                        "top_share": share,
                        "distinct": c.n_unique,
                        "other_rows": int(c.n_present - counts.iloc[0]),
                    },
                    rule=rule,
                    interpretation=(
                        "Little variation limits what this column can explain, but the rare "
                        "values may be exactly the interesting cases (e.g. rare events)."
                    ),
                    suggestion="Decide whether the rare values are meaningful or noise.",
                )
            )
        return out


@register_detector
class IdentifierDetector(BaseDetector):
    name = "identifier"
    category = Category.SCHEMA
    description = (
        "Explains which columns look like identifiers (near-unique, monotonic, UUID or "
        "fixed-format values, identifier-like names) and flags identifiers stored as "
        "numbers or near-unique columns that are not typed as identifiers."
    )
    assumptions = "Identifiers label records; their numeric value carries no meaning."
    limitations = (
        "Name patterns are English-centric. A unique measurement (e.g. a precise "
        "timestamp or amount) is only flagged when other signals agree."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        out: list[Finding] = []
        for col in ctx.columns_of(ColumnType.IDENTIFIER):
            c = ctx.schema[col]
            reasons = c.reasons[1:] or c.reasons
            numeric = ctx.frame[col].dtype.kind in "iuf"
            out.append(
                self.finding(
                    "identifier.detected",
                    Severity.WARNING if numeric else Severity.INFO,
                    f"'{col}' looks like an identifier",
                    f"'{col}' has {c.n_unique:,} distinct values ({pct(c.unique_ratio)} of non-missing rows).",
                    columns=[col],
                    evidence={
                        "reasons": reasons,
                        "unique_ratio": c.unique_ratio,
                        "distinct": c.n_unique,
                    },
                    rule=(
                        "Numeric identifiers -> warning (easily mistaken for a quantitative "
                        "feature); otherwise info."
                    ),
                    interpretation=(
                        "Identifiers label records. Using them as model features usually leads "
                        "to memorisation, and ordered IDs can leak time or sort order."
                    ),
                    suggestion="Exclude it from features, or declare it in id_columns.",
                    confidence=0.85 if len(reasons) >= 2 else 0.6,
                )
            )
        for col, c in ctx.schema.columns.items():
            if c.column_type != ColumnType.CATEGORICAL:
                continue
            if c.unique_ratio >= ctx.thresholds.identifier_unique_ratio and c.n_unique >= 50:
                out.append(
                    self.finding(
                        "identifier.near_unique",
                        Severity.INFO,
                        f"'{col}' is nearly unique per row",
                        f"'{col}' has {c.n_unique:,} distinct values for {c.n_present:,} "
                        f"non-missing rows ({pct(c.unique_ratio)}).",
                        columns=[col],
                        evidence={"unique_ratio": c.unique_ratio},
                        rule=f"Categorical with unique ratio >= identifier_unique_ratio ({ctx.thresholds.identifier_unique_ratio}).",
                        interpretation="It may be a key, a free-text field, or a quasi-identifier (privacy).",
                        confidence=0.5,
                    )
                )
        return out
