from __future__ import annotations

import numpy as np
import pandas as pd

from datasi.detectors.base import BaseDetector, Context, pct, register_detector
from datasi.findings import Category, Finding, Severity


@register_detector
class RulesDetector(BaseDetector):
    name = "rules"
    category = Category.TYPES
    description = (
        "Checks user-declared domain rules (min, max, allowed values, regex pattern, "
        "nullability). Violations are confirmed invalid values, unlike statistical outliers."
    )
    assumptions = "The declared rules are correct for this dataset."
    limitations = (
        "Only as good as the rules supplied; no rules means no confirmed-invalid findings."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        out: list[Finding] = []
        for col, rule in ctx.config.rules.items():
            if col not in ctx.frame:
                continue
            s = ctx.frame[col]
            present = s.notna()
            checks: list[tuple[str, str, pd.Series]] = []
            if rule.min is not None or rule.max is not None:
                v = ctx.numeric(col)
                if rule.min is not None:
                    checks.append(("min", f"value < {rule.min:g}", v < rule.min))
                if rule.max is not None:
                    checks.append(("max", f"value > {rule.max:g}", v > rule.max))
                unparsable = present & v.isna()
                if unparsable.any():
                    checks.append(("numeric", "not a number", unparsable))
            if rule.allowed is not None:
                allowed = set(rule.allowed)
                checks.append(
                    ("allowed", "value not in allowed set", present & ~s.astype(str).isin(allowed))
                )
            if rule.pattern is not None:
                ok = s[present].astype(str).str.fullmatch(rule.pattern).astype(bool)
                checks.append(
                    (
                        "pattern",
                        f"does not match /{rule.pattern}/",
                        present & ~ok.reindex(s.index, fill_value=True),
                    )
                )
            if not rule.nullable:
                checks.append(("nullable", "missing although declared non-nullable", ~present))
            for kind, desc, mask in checks:
                m = np.asarray(mask.fillna(False), dtype=bool)
                n = int(m.sum())
                if not n:
                    continue
                out.append(
                    self.finding(
                        f"rules.{kind}",
                        Severity.HIGH,
                        f"Rule violation in '{col}': {desc}",
                        f"{n:,} row(s) ({pct(n / max(ctx.n_rows, 1))}) of '{col}' violate the declared rule ({desc}).",
                        columns=[col],
                        evidence={
                            "rule": rule.model_dump(exclude_none=True),
                            "violations": n,
                            "row_examples": ctx.row_examples(m),
                            "examples": ctx.examples(s[m].head(20)),
                        },
                        rule="Any violation of a user-declared domain rule -> high (confirmed invalid).",
                        suggestion="Fix or exclude the violating rows; check the rule if many rows fail.",
                        confidence=1.0,
                    )
                )
        return out
