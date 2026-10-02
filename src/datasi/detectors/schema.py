from __future__ import annotations

import re
from collections import defaultdict

from datasi.detectors.base import BaseDetector, Context, register_detector
from datasi.findings import Category, Finding, Severity

_UNNAMED_RE = re.compile(r"^(Unnamed: \d+|index|level_\d+)$")


@register_detector
class SchemaDetector(BaseDetector):
    name = "schema"
    category = Category.SCHEMA
    description = (
        "Structural checks on the table itself: empty data, duplicate or colliding column "
        "names, exported index columns, and configuration that references absent columns."
    )
    assumptions = "Column names are meaningful to the user; no semantic meaning is assumed."
    limitations = "Does not compare against an external schema contract."

    def analyze(self, ctx: Context) -> list[Finding]:
        out: list[Finding] = []
        ds = ctx.dataset
        if ds.n_rows == 0:
            out.append(
                self.finding(
                    "schema.empty",
                    Severity.CRITICAL,
                    "Dataset has no rows",
                    f"The dataset has 0 rows and {ds.n_columns} columns.",
                    rule="0 rows is always critical: nothing else can be assessed.",
                    suggestion="Check the extraction query or file export that produced it.",
                )
            )
        for name in ds.duplicate_column_names:
            out.append(
                self.finding(
                    "schema.duplicate_name",
                    Severity.WARNING,
                    f"Duplicate column name '{name}'",
                    f"More than one column is named '{name}'; later copies were analysed as "
                    f"'{name}__2', '{name}__3', ...",
                    columns=[name],
                    rule="Duplicate names make column selection ambiguous.",
                    suggestion="Rename or drop the duplicated columns at the source.",
                )
            )
        folded: dict[str, list[str]] = defaultdict(list)
        for col in ds.columns:
            folded[col.strip().lower()].append(col)
            if col != col.strip():
                out.append(
                    self.finding(
                        "schema.name_whitespace",
                        Severity.INFO,
                        f"Column name '{col}' has leading/trailing whitespace",
                        f"The column name {col!r} contains surrounding whitespace.",
                        columns=[col],
                        suggestion="Strip column names to avoid lookup errors.",
                    )
                )
            if col.strip() == "":
                out.append(
                    self.finding(
                        "schema.blank_name",
                        Severity.WARNING,
                        "Blank column name",
                        "A column has an empty name.",
                        columns=[col],
                    )
                )
            if _UNNAMED_RE.match(col):
                c = ctx.schema[col]
                out.append(
                    self.finding(
                        "schema.exported_index",
                        Severity.INFO,
                        f"Column '{col}' looks like an exported index",
                        f"Column '{col}' has a name typical of a saved DataFrame index "
                        f"({c.column_type.value}, {c.unique_ratio:.0%} unique).",
                        columns=[col],
                        interpretation="It may be an artefact of saving a DataFrame with its index.",
                        suggestion="Drop it if it carries no meaning.",
                        confidence=0.7,
                    )
                )
        for group in folded.values():
            if len(group) > 1 and len(set(group)) > 1:
                out.append(
                    self.finding(
                        "schema.name_collision",
                        Severity.WARNING,
                        "Column names differ only by case or whitespace",
                        f"Columns {group} have the same name after case folding and stripping.",
                        columns=group,
                        interpretation="These may be the same field from two sources or schema versions.",
                        suggestion="Check whether they should be merged.",
                        confidence=0.6,
                    )
                )
        referenced = {
            "target": [ctx.config.target] if ctx.config.target else [],
            "time_column": [ctx.config.time_column] if ctx.config.time_column else [],
            "id_columns": ctx.config.id_columns,
            "rules": list(ctx.config.rules),
        }
        for key, cols in referenced.items():
            for col in cols:
                if col not in ctx.schema.columns:
                    out.append(
                        self.finding(
                            "schema.config_column_missing",
                            Severity.CRITICAL if key == "target" else Severity.WARNING,
                            f"Configured column '{col}' is not in the dataset",
                            f"Configuration '{key}' refers to '{col}', which does not exist.",
                            columns=[col],
                            rule="Target absent is critical; other configured columns are warnings.",
                            suggestion="Check the column name or the dataset version.",
                        )
                    )
        return out
