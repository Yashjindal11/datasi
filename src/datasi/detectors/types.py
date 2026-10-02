from __future__ import annotations

import pandas as pd

from datasi.detectors.base import BaseDetector, Context, pct, register_detector
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType, StorageHint
from datasi.statistics import strings as st


def _non_matching_shapes(
    counts: pd.Series, mask_fn: object, top: int = 5
) -> list[dict[str, object]]:
    """Character-class shapes (e.g. 'AAA', '99-AA') of values that failed to parse.

    Shapes describe the format of offending values without revealing them.
    """
    shapes: dict[str, int] = {}
    for v, n in counts.items():
        if not mask_fn(str(v)):  # type: ignore[operator]
            sh = st.shape(str(v).strip())
            shapes[sh] = shapes.get(sh, 0) + int(n)
    ranked = sorted(shapes.items(), key=lambda kv: -kv[1])[:top]
    return [{"shape": k, "rows": v} for k, v in ranked]


def _is_number(v: str) -> bool:
    return st.number_kind(v) is not None


@register_detector
class TypeDetector(BaseDetector):
    name = "types"
    category = Category.TYPES
    description = (
        "Values stored as the wrong type (numbers, currency, percentages, dates or booleans "
        "stored as text), mixed Python types, inconsistent date formats, inconsistent "
        "boolean spellings and columns that are mostly but not entirely numeric."
    )
    assumptions = (
        "A column is assumed to mean one type when at least type_parse_ratio of its values "
        "parse as that type. Data is never converted; only reported."
    )
    limitations = (
        "Ambiguous day/month order (01/02/2026) cannot be resolved from the values alone. "
        "Locale-specific number formats (1.234,5) are not recognised as numbers."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        out: list[Finding] = []
        thr = ctx.thresholds.type_parse_ratio
        for col, c in ctx.schema.columns.items():
            hints = set(c.hints)
            if StorageHint.MIXED_TYPES in hints:
                out.append(
                    self.finding(
                        "types.mixed",
                        Severity.WARNING,
                        f"Mixed value types in '{col}'",
                        f"'{col}' holds values of several Python types: "
                        + ", ".join(f"{k} ({v:,})" for k, v in c.python_types.items())
                        + ".",
                        columns=[col],
                        evidence={"python_types": c.python_types},
                        rule="More than one Python type in an object column -> warning.",
                        interpretation="Often caused by merging sources or by JSON fields with varying types.",
                        suggestion="Decide on one type and convert explicitly at ingestion.",
                    )
                )
            if not (
                pd.api.types.is_object_dtype(ctx.frame[col])
                or pd.api.types.is_string_dtype(ctx.frame[col])
            ):
                continue
            counts = ctx.counts(col)
            if counts.empty:
                continue
            if hints & {
                StorageHint.NUMERIC_AS_STRING,
                StorageHint.CURRENCY_AS_STRING,
                StorageHint.PERCENT_AS_STRING,
            }:
                out.append(self._numeric_text(ctx, col, counts, hints))
            elif c.column_type in (ColumnType.CATEGORICAL, ColumnType.TEXT):
                parsed = st.numeric_parse(counts)
                if 0.5 <= parsed.ratio < thr:
                    bad = _non_matching_shapes(counts, _is_number)
                    out.append(
                        self.finding(
                            "types.mostly_numeric",
                            Severity.WARNING,
                            f"'{col}' is mostly numeric but contains non-numeric values",
                            f"{pct(parsed.ratio)} of non-missing values in '{col}' parse as numbers; "
                            f"the rest do not (most common formats: "
                            + ", ".join(f"'{b['shape']}'" for b in bad[:3])
                            + ").",
                            columns=[col],
                            evidence={"numeric_ratio": parsed.ratio, "non_numeric_formats": bad},
                            rule=f"0.5 <= numeric parse ratio < type_parse_ratio ({thr}).",
                            interpretation=(
                                "The non-numeric values may be placeholders for missing data "
                                "(e.g. 'N/A', '-'), units, or data-entry errors."
                            ),
                            suggestion="Inspect the non-numeric values before converting the column.",
                            confidence=0.7,
                        )
                    )
            if StorageHint.DATETIME_AS_STRING in hints:
                out.extend(self._date_text(ctx, col, counts))
            if StorageHint.BOOLEAN_AS_TEXT in hints:
                tokens = sorted({str(v).strip() for v in counts.index})
                severity = Severity.WARNING if len(tokens) > 2 else Severity.INFO
                out.append(
                    self.finding(
                        "types.boolean_text",
                        severity,
                        f"Boolean values stored as text in '{col}'"
                        + (" with inconsistent spellings" if len(tokens) > 2 else ""),
                        f"'{col}' contains only boolean-like tokens: {tokens}.",
                        columns=[col],
                        evidence={
                            "tokens": tokens,
                            "counts": {str(k): int(v) for k, v in counts.items()},
                        },
                        rule="Boolean text -> info; more than two spellings -> warning.",
                        suggestion="Map the tokens to a boolean type explicitly.",
                    )
                )
        return out

    def _numeric_text(
        self, ctx: Context, col: str, counts: pd.Series, hints: set[StorageHint]
    ) -> Finding:
        parsed = st.numeric_parse(counts)
        if StorageHint.CURRENCY_AS_STRING in hints:
            kind = "currency amounts"
        elif StorageHint.PERCENT_AS_STRING in hints:
            kind = "percentages"
        else:
            kind = "numbers"
        bad = _non_matching_shapes(counts, _is_number)
        mixed = parsed.kind == "mixed"
        return self.finding(
            "types.numeric_text",
            Severity.WARNING if (mixed or bad) else Severity.INFO,
            f"'{col}' stores {kind} as text",
            f"{pct(parsed.ratio)} of non-missing values in '{col}' are {kind} written as text "
            f"(formats: {', '.join(f'{k}: {v:,}' for k, v in parsed.kinds.items())}).",
            columns=[col],
            evidence={
                "parse_ratio": parsed.ratio,
                "formats": parsed.kinds,
                "unparsed_formats": bad,
            },
            rule="Numeric text -> info; several numeric conventions or unparseable values -> warning.",
            suggestion="Convert to a numeric type at ingestion, stripping symbols explicitly.",
        )

    def _date_text(self, ctx: Context, col: str, counts: pd.Series) -> list[Finding]:
        out: list[Finding] = []
        shapes = st.date_shapes(counts)
        total = int(counts.sum())
        significant = {k: v for k, v in shapes.items() if v / total >= 0.001}
        tz = st.timezone_styles(counts)
        if len(significant) > 1:
            ambiguous = any("or" in k for k in significant)
            out.append(
                self.finding(
                    "types.date_formats",
                    Severity.WARNING,
                    f"Inconsistent date formats in '{col}'",
                    f"'{col}' mixes {len(significant)} date formats: "
                    + ", ".join(f"{k} ({v:,} rows)" for k, v in significant.items())
                    + ".",
                    columns=[col],
                    evidence={"formats": significant},
                    rule="More than one recognised date format, each in >= 0.1% of rows -> warning.",
                    interpretation=(
                        "Several sources or entry methods may feed this column."
                        + (
                            " Day/month order is ambiguous for slash formats, so parsed dates may be wrong."
                            if ambiguous
                            else ""
                        )
                    ),
                    suggestion="Parse each format explicitly rather than relying on automatic inference.",
                    confidence=0.9,
                )
            )
        tz_used = {k: v for k, v in tz.items() if v}
        if len(tz_used) > 1:
            out.append(
                self.finding(
                    "types.timezone_mixed",
                    Severity.WARNING,
                    f"Mixed timezone notation in '{col}'",
                    f"Timestamps in '{col}' use different timezone notations: {tz_used}.",
                    columns=[col],
                    category=Category.TEMPORAL,
                    evidence={"styles": tz_used},
                    rule="More than one of {no offset, Z, +hh:mm} -> warning.",
                    interpretation="Timestamps without an offset may be in local time while others are UTC.",
                    suggestion="Normalise all timestamps to UTC with explicit offsets.",
                )
            )
        if not out:
            fmt = next(iter(shapes), "unrecognised format")
            out.append(
                self.finding(
                    "types.date_text",
                    Severity.INFO,
                    f"Dates stored as text in '{col}'",
                    f"'{col}' holds date/time values as text ({fmt}).",
                    columns=[col],
                    evidence={"formats": shapes},
                    rule="Date text with a single format -> info.",
                    suggestion="Parse to a datetime type with an explicit format.",
                )
            )
        return out
