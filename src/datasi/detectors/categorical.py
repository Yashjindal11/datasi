from __future__ import annotations

import difflib
from collections import defaultdict

import numpy as np
import pandas as pd

from datasi.detectors.base import BaseDetector, Context, pct, register_detector
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType
from datasi.statistics import strings as st
from datasi.statistics.descriptive import entropy_bits, normalized_entropy

MAX_LABELS_FOR_FUZZY = 150
PLACEHOLDERS = frozenset(
    {
        "n/a",
        "na",
        "nan",
        "null",
        "none",
        "nil",
        "-",
        "--",
        "?",
        "unknown",
        "missing",
        "undefined",
        "#n/a",
        "not available",
        "tbd",
        ".",
    }
)


@register_detector
class CategoricalDetector(BaseDetector):
    name = "categorical"
    category = Category.CATEGORICAL
    description = (
        "Frequency structure of categorical columns: cardinality, entropy, rare levels, "
        "category explosion, and labels that differ only by case, whitespace, punctuation "
        "or small spelling differences."
    )
    assumptions = (
        "Labels that normalise to the same string (case/whitespace/punctuation) are "
        "intended to be the same category. Fuzzy matches are only suggestions."
    )
    limitations = (
        "Fuzzy matching uses character similarity (difflib ratio >= 0.88) on the most "
        "frequent labels; it cannot know that 'UA' means 'United Airlines', and may pair "
        "genuinely different short labels."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        t = ctx.thresholds
        out: list[Finding] = []
        for col in ctx.columns_of(ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            counts = ctx.counts(col)
            if counts.empty:
                continue
            c = ctx.schema[col]
            total = int(counts.sum())
            shares = counts / total
            rare = shares[shares < t.rare_category_ratio]
            ctx.profile(col)["categorical"] = {
                "distinct": int(counts.size),
                "entropy_bits": entropy_bits(counts),
                "normalized_entropy": normalized_entropy(counts),
                "top": [
                    {"label": ctx.label(k), "count": int(v), "share": float(v / total)}
                    for k, v in counts.head(15).items()
                ],
                "rare_levels": int(rare.size),
                "rare_share": float(rare.sum()),
            }
            if counts.size >= t.high_cardinality_min and c.unique_ratio >= t.high_cardinality_ratio:
                out.append(
                    self.finding(
                        "categorical.explosion",
                        Severity.WARNING,
                        f"Very high cardinality in '{col}'",
                        f"'{col}' has {counts.size:,} distinct labels across {total:,} rows "
                        f"({pct(c.unique_ratio)} unique).",
                        columns=[col],
                        evidence={"distinct": int(counts.size), "unique_ratio": c.unique_ratio},
                        rule=(
                            f"distinct >= high_cardinality_min ({t.high_cardinality_min}) and unique "
                            f"ratio >= high_cardinality_ratio ({t.high_cardinality_ratio})."
                        ),
                        interpretation="It may be free text, an identifier, or a code that needs grouping.",
                        suggestion="Group rare levels or treat it as text/identifier rather than a category.",
                        confidence=0.6,
                    )
                )
            elif rare.size >= 2 and counts.size >= 3:
                out.append(
                    self.finding(
                        "categorical.rare",
                        Severity.INFO,
                        f"Rare categories in '{col}'",
                        f"{rare.size:,} of {counts.size:,} labels in '{col}' each cover less than "
                        f"{pct(t.rare_category_ratio)} of rows ({pct(float(rare.sum()))} of rows in total).",
                        columns=[col],
                        evidence={
                            "rare_levels": int(rare.size),
                            "rare_share": float(rare.sum()),
                            "examples": [ctx.label(x) for x in rare.index[:10]],
                        },
                        rule=f"At least 2 labels below rare_category_ratio ({t.rare_category_ratio}).",
                        interpretation="Rare levels can be typos, retired codes, or genuine small groups.",
                        suggestion="Check whether rare labels are variants of frequent ones.",
                    )
                )
            out.extend(self._variants(ctx, col, counts))
        return out

    def _variants(self, ctx: Context, col: str, counts: pd.Series) -> list[Finding]:
        out: list[Finding] = []
        groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for label, n in counts.items():
            groups[st.normalize_label(str(label))].append((str(label), int(n)))
        variant_groups = [g for g in groups.values() if len(g) > 1]
        if variant_groups:
            variant_groups.sort(key=lambda g: -sum(n for _, n in g))
            kinds = set()
            for g in variant_groups:
                raw = [x for x, _ in g]
                if len({x.strip() for x in raw}) < len(raw):
                    kinds.add("whitespace")
                if len({x.lower() for x in raw}) < len(raw):
                    kinds.add("case")
                if len({x.strip().lower() for x in raw}) == len(raw):
                    kinds.add("punctuation/separators")
            rows = sum(sum(n for _, n in g) - max(n for _, n in g) for g in variant_groups)
            out.append(
                self.finding(
                    "categorical.variants",
                    Severity.WARNING,
                    f"Inconsistent spellings of the same label in '{col}'",
                    f"{len(variant_groups):,} label group(s) in '{col}' differ only by "
                    f"{', '.join(sorted(kinds))}; {rows:,} row(s) use a minority spelling.",
                    columns=[col],
                    evidence={
                        "groups": [
                            [
                                {"label": ctx.label(x), "count": n}
                                for x, n in sorted(g, key=lambda p: -p[1])
                            ]
                            for g in variant_groups[:10]
                        ],
                        "kinds": sorted(kinds),
                        "rows_affected": rows,
                    },
                    rule="Labels equal after case folding, whitespace and punctuation normalisation -> warning.",
                    interpretation="Probably the same category entered inconsistently; DataSI does not merge them.",
                    suggestion="Normalise labels (strip, case-fold) at ingestion if they are the same category.",
                    confidence=0.85,
                )
            )
        # Fuzzy matches among normalised labels (one representative per group).
        reps = sorted(
            ((k, sum(n for _, n in g), g) for k, g in groups.items() if k), key=lambda x: -x[1]
        )
        reps = reps[:MAX_LABELS_FOR_FUZZY]
        if len(reps) < 2:
            return out
        pairs = []
        for i, (k, _n, _g) in enumerate(reps):
            for j in range(i + 1, len(reps)):
                k2 = reps[j][0]
                if abs(len(k) - len(k2)) > max(3, 0.3 * max(len(k), len(k2))):
                    continue
                if min(len(k), len(k2)) < 4 or any(ch.isdigit() for ch in k + k2):
                    continue  # 'Route 12' vs 'Route 13' are different things
                ratio = difflib.SequenceMatcher(None, k, k2).ratio()
                if ratio >= 0.88:
                    pairs.append((reps[i], reps[j], ratio))
        if not pairs:
            # Token extension: 'united' vs 'united airlines'
            head = reps[:100]
            for i, (k, n, _g) in enumerate(head):
                for j, (k2, n2, _g2) in enumerate(head):
                    if i == j or len(k) < 4:
                        continue
                    if k2.startswith(k + " ") and n2 < n and len(k2.split()) == len(k.split()) + 1:
                        pairs.append((reps[i], reps[j], 0.6))
        if pairs:
            pairs.sort(key=lambda p: -p[2])
            shown = [
                {
                    "a": ctx.label(p[0][2][0][0]),
                    "a_count": p[0][1],
                    "b": ctx.label(p[1][2][0][0]),
                    "b_count": p[1][1],
                    "similarity": round(p[2], 3),
                }
                for p in pairs[:10]
            ]
            out.append(
                self.finding(
                    "categorical.similar_labels",
                    Severity.INFO,
                    f"Possibly related labels in '{col}'",
                    f"{len(pairs):,} pair(s) of labels in '{col}' are very similar, e.g. "
                    f"'{shown[0]['a']}' and '{shown[0]['b']}'.",
                    columns=[col],
                    evidence={"pairs": shown},
                    rule="difflib similarity >= 0.88 between normalised labels (or one label extending another by one word).",
                    interpretation="These may be spelling variants of one category, or genuinely distinct categories.",
                    suggestion="Review the pairs; map confirmed variants to one label.",
                    confidence=float(np.mean([p[2] for p in pairs[:10]])) * 0.6,
                )
            )
        return out


@register_detector
class StringDetector(BaseDetector):
    name = "strings"
    category = Category.STRINGS
    description = (
        "Text hygiene: empty and whitespace-only strings, placeholder tokens that disguise "
        "missing values, leading/trailing whitespace, invisible or control characters, and "
        "values that break a column's dominant format."
    )
    assumptions = "A column with one dominant character-class format is expected to follow it."
    limitations = (
        "Format checks use character-class shapes (digits->9, letters->A), so they cannot "
        "validate check digits or semantics."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        out: list[Finding] = []
        for col, c in ctx.schema.columns.items():
            s = ctx.frame[col]
            if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                continue
            counts = ctx.counts(col)
            if counts.empty:
                continue
            total = int(counts.sum())
            labels = pd.Series(counts.index.astype(str), index=counts.index)
            stripped = labels.str.strip()
            empty = counts[(labels == "").to_numpy()].sum()
            blank = counts[((stripped == "") & (labels != "")).to_numpy()].sum()
            if empty or blank:
                out.append(
                    self.finding(
                        "strings.empty",
                        Severity.WARNING,
                        f"Empty or blank strings in '{col}'",
                        f"'{col}' has {int(empty):,} empty and {int(blank):,} whitespace-only value(s) "
                        "that are not counted as missing.",
                        columns=[col],
                        evidence={"empty": int(empty), "whitespace_only": int(blank)},
                        rule="Any empty or whitespace-only string -> warning.",
                        interpretation="These are probably missing values in disguise.",
                        suggestion="Convert them to proper nulls at ingestion.",
                        confidence=0.8,
                    )
                )
            ph_mask = stripped.str.lower().isin(PLACEHOLDERS).to_numpy()
            if ph_mask.any() and c.column_type != ColumnType.TEXT:
                ph = counts[ph_mask]
                out.append(
                    self.finding(
                        "strings.placeholder",
                        Severity.WARNING,
                        f"Placeholder values in '{col}'",
                        f"{int(ph.sum()):,} value(s) ({pct(ph.sum() / total)}) of '{col}' are placeholder "
                        f"tokens such as {sorted({str(x) for x in ph.index})[:5]}.",
                        columns=[col],
                        category=Category.MISSINGNESS,
                        evidence={"tokens": {str(k): int(v) for k, v in ph.items()}},
                        rule="Values equal (case-insensitively) to common missing-value tokens -> warning.",
                        interpretation="These probably represent missing data, so the true missing rate is higher.",
                        suggestion="Treat these tokens as missing (e.g. na_values in read_csv).",
                        confidence=0.8,
                    )
                )
            padded = counts[((labels != stripped) & (stripped != "")).to_numpy()].sum()
            if padded and c.column_type != ColumnType.TEXT:
                out.append(
                    self.finding(
                        "strings.padding",
                        Severity.INFO,
                        f"Leading/trailing whitespace in '{col}'",
                        f"{int(padded):,} value(s) ({pct(padded / total)}) of '{col}' have surrounding whitespace.",
                        columns=[col],
                        evidence={"rows": int(padded)},
                        rule="Any padded value in a non-text column -> info.",
                        suggestion="Strip values to avoid mismatched joins and duplicate categories.",
                    )
                )
            bad: dict[str, int] = {}
            rows_bad = 0
            for label, n in counts.items():
                chars = st.unusual_characters(str(label))
                if chars:
                    rows_bad += int(n)
                    for ch in set(chars):
                        bad[ch] = bad.get(ch, 0) + int(n)
            if rows_bad:
                out.append(
                    self.finding(
                        "strings.unusual_characters",
                        Severity.WARNING,
                        f"Invisible or control characters in '{col}'",
                        f"{rows_bad:,} value(s) of '{col}' contain control, zero-width, non-breaking "
                        f"or replacement characters ({', '.join(sorted(bad))}).",
                        columns=[col],
                        evidence={"code_points": bad, "rows": rows_bad},
                        rule="Any control/zero-width/NBSP/U+FFFD character -> warning.",
                        interpretation="Usually encoding problems or copy-paste artefacts; they break equality and joins.",
                        suggestion="Normalise text encoding and remove invisible characters.",
                        confidence=0.9,
                    )
                )
            if (
                c.column_type in (ColumnType.IDENTIFIER, ColumnType.CATEGORICAL)
                and counts.size >= 20
            ):
                f = self._format(ctx, col, counts, total)
                if f:
                    out.append(f)
            if c.column_type == ColumnType.TEXT:
                lengths = st.length_stats(counts)
                ctx.profile(col)["text"] = lengths
        return out

    def _format(self, ctx: Context, col: str, counts: pd.Series, total: int) -> Finding | None:
        shapes: dict[str, int] = {}
        for label, n in counts.items():
            sh = st.compact_shape(str(label).strip())
            shapes[sh] = shapes.get(sh, 0) + int(n)
        ranked = sorted(shapes.items(), key=lambda kv: -kv[1])
        dom, dom_n = ranked[0]
        share = dom_n / total
        if share < 0.9 or share == 1.0 or "9" not in dom:
            return None
        others = ranked[1:6]
        off = total - dom_n
        return self.finding(
            "strings.format",
            Severity.WARNING if share >= 0.99 else Severity.INFO,
            f"Values deviating from the dominant format in '{col}'",
            f"{pct(share)} of values in '{col}' have the format '{dom}', but {off:,} value(s) do not "
            f"(e.g. {', '.join(repr(k) for k, _ in others[:3])}).",
            columns=[col],
            evidence={
                "dominant_format": dom,
                "dominant_share": share,
                "other_formats": [{"format": k, "rows": v} for k, v in others],
            },
            rule="A format covering >= 90% of rows with exceptions; >= 99% -> warning (isolated breaks).",
            interpretation="Exceptions to a strict format are often malformed or legacy values.",
            suggestion="Validate the exceptions against the expected format.",
            confidence=0.6 if share >= 0.99 else 0.4,
        )
