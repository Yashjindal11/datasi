from __future__ import annotations

import pandas as pd

from datasi.detectors.base import BaseDetector, Context, pct, register_detector, tiered
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType


def _hashable(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns holding unhashable objects (lists, dicts) are compared by their repr."""
    out = {}
    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_object_dtype(s):
            try:
                s.head(1000).map(hash)
            except TypeError:
                s = s.map(repr)
        out[c] = s
    return pd.DataFrame(out)


@register_detector
class DuplicateDetector(BaseDetector):
    name = "duplicates"
    category = Category.DUPLICATES
    description = (
        "Exact duplicate rows, rows identical except for their identifiers, duplicated "
        "values in key-like columns, and repeated entities whose attributes conflict."
    )
    assumptions = (
        "Rows are records of distinct events/entities unless the data is known to "
        "contain legitimate repeats (e.g. identical transactions)."
    )
    limitations = (
        "Exact matching only; near-duplicate text (typos) is not matched row-wise. "
        "Legitimate repeated records are indistinguishable from accidental duplicates."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        if ctx.n_rows < 2 or ctx.frame.shape[1] == 0:
            return []
        t = ctx.thresholds
        out: list[Finding] = []
        frame = _hashable(ctx.frame)
        dup = frame.duplicated(keep="first")
        n_dup = int(dup.sum())
        ratio = n_dup / ctx.n_rows
        ctx.section("duplicates").update({"exact_duplicate_rows": n_dup, "ratio": ratio})
        rule = (
            f"duplicate share >= duplicate_high ({t.duplicate_high}) -> high; "
            f">= duplicate_warning ({t.duplicate_warning}) -> warning; otherwise info"
        )
        if n_dup:
            groups = frame[frame.duplicated(keep=False)]
            n_groups = int(groups.drop_duplicates().shape[0])
            out.append(
                self.finding(
                    "duplicates.exact",
                    tiered(
                        ratio,
                        [
                            (t.duplicate_high, Severity.HIGH),
                            (t.duplicate_warning, Severity.WARNING),
                        ],
                        Severity.INFO,
                    ),
                    "Exact duplicate rows",
                    f"{n_dup:,} row(s) ({pct(ratio)}) are exact copies of an earlier row, "
                    f"forming {n_groups:,} group(s) of identical rows.",
                    evidence={
                        "duplicate_rows": n_dup,
                        "ratio": ratio,
                        "groups": n_groups,
                        "row_examples": ctx.row_examples(dup),
                    },
                    rule=rule,
                    interpretation=(
                        "Duplicates are often caused by repeated ingestion or many-to-one joins; "
                        "they can also be legitimate repeated events."
                    ),
                    suggestion="Check whether the data has a natural key and how it was loaded.",
                    confidence=0.6,
                )
            )
        ids = [c for c in ctx.columns_of(ColumnType.IDENTIFIER) if c in frame]
        content = [c for c in frame.columns if c not in ids]
        if ids and content:
            dup_content = frame.duplicated(subset=content, keep="first") & ~dup
            n = int(dup_content.sum())
            if n:
                r = n / ctx.n_rows
                out.append(
                    self.finding(
                        "duplicates.except_ids",
                        tiered(
                            r,
                            [
                                (t.duplicate_high, Severity.HIGH),
                                (t.duplicate_warning, Severity.WARNING),
                            ],
                            Severity.INFO,
                        ),
                        "Rows identical apart from identifier columns",
                        f"{n:,} row(s) ({pct(r)}) match an earlier row on every non-identifier "
                        f"column but carry different identifiers ({', '.join(ids)}).",
                        columns=ids,
                        evidence={
                            "rows": n,
                            "ratio": r,
                            "identifier_columns": ids,
                            "row_examples": ctx.row_examples(dup_content),
                        },
                        rule=rule,
                        interpretation=(
                            "The same record may have been ingested twice with fresh IDs, or "
                            "distinct entities may legitimately share all attributes."
                        ),
                        suggestion="Check whether new identifiers are generated on re-load.",
                        confidence=0.5,
                    )
                )
        out.extend(self._keys(ctx, frame, ids))
        out.extend(self._conflicting_entities(ctx, ids))
        return out

    def _keys(self, ctx: Context, frame: pd.DataFrame, ids: list[str]) -> list[Finding]:
        out = []
        declared = {c for c, r in ctx.config.rules.items() if r.unique and c in frame}
        for col in sorted(set(ids) | declared):
            c = ctx.schema[col]
            if col not in declared and c.unique_ratio < ctx.thresholds.identifier_unique_ratio:
                continue  # repeated-entity keys are expected to repeat
            s = frame[col].dropna()
            dup = s.duplicated(keep=False)
            if not dup.any():
                continue
            n_vals = int(s[dup].nunique())
            n_rows = int(dup.sum())
            out.append(
                self.finding(
                    "duplicates.key",
                    Severity.HIGH if col in declared else Severity.WARNING,
                    f"Duplicate values in key-like column '{col}'",
                    f"{n_vals:,} value(s) of '{col}' appear more than once ({n_rows:,} rows), "
                    f"although {pct(c.unique_ratio)} of its values are unique.",
                    columns=[col],
                    evidence={
                        "duplicated_values": n_vals,
                        "rows_involved": n_rows,
                        "unique_ratio": c.unique_ratio,
                        "row_examples": ctx.row_examples(
                            frame[col].duplicated(keep=False) & frame[col].notna()
                        ),
                    },
                    rule=(
                        "Declared unique (rules) -> high; otherwise an identifier with unique "
                        f"ratio >= identifier_unique_ratio ({ctx.thresholds.identifier_unique_ratio}) "
                        "that still repeats -> warning."
                    ),
                    interpretation="A primary key that repeats usually indicates duplicated records.",
                    suggestion=f"Inspect the rows sharing a '{col}' value.",
                    confidence=0.9 if col in declared else 0.6,
                )
            )
        return out

    def _conflicting_entities(self, ctx: Context, ids: list[str]) -> list[Finding]:
        """Entity keys that repeat should usually carry the same entity attributes."""
        out = []
        cats = ctx.columns_of(ColumnType.CATEGORICAL)
        for key in ids:
            c = ctx.schema[key]
            if c.unique_ratio >= ctx.thresholds.identifier_unique_ratio or c.n_unique < 20:
                continue
            sub = ctx.frame[[key, *cats]].dropna(subset=[key])
            if sub.empty:
                continue
            nun = sub.groupby(key, observed=True)[cats].nunique() if cats else pd.DataFrame()
            for attr in cats:
                repeated = sub[key].value_counts()
                rep_keys = repeated[repeated > 1].index
                if len(rep_keys) < 20:
                    break
                per_key = nun.loc[rep_keys, attr]
                consistent = float((per_key <= 1).mean())
                # Mostly 1:1 with a few conflicts -> an attribute of the entity with errors.
                if 0.95 <= consistent < 1.0:
                    n_conf = int((per_key > 1).sum())
                    out.append(
                        self.finding(
                            "duplicates.entity_conflict",
                            Severity.WARNING,
                            f"Repeated '{key}' values carry conflicting '{attr}'",
                            f"For {pct(consistent)} of repeated '{key}' values, '{attr}' is "
                            f"constant, but {n_conf:,} '{key}' value(s) map to several '{attr}' values.",
                            columns=[key, attr],
                            evidence={
                                "consistent_share": consistent,
                                "conflicting_keys": n_conf,
                                "repeated_keys": len(rep_keys),
                            },
                            rule="An attribute constant for >= 95% (but not all) of repeated keys.",
                            interpretation=(
                                f"'{attr}' behaves like a property of the '{key}' entity, so the "
                                "conflicts may be data-entry errors, or the attribute may change over time."
                            ),
                            suggestion=f"Review the {n_conf} conflicting '{key}' values.",
                            confidence=0.6,
                        )
                    )
        return out
