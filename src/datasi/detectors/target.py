from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasi.detectors.base import BaseDetector, Context, pct, register_detector, tiered
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType

MAX_CLASSES = 50


def task_type(ctx: Context, target: str) -> str:
    c = ctx.schema[target]
    if c.column_type in (ColumnType.BOOLEAN, ColumnType.CATEGORICAL, ColumnType.CONSTANT):
        return "classification"
    if c.column_type == ColumnType.NUMERIC:
        s = ctx.numeric(target).dropna()
        if c.n_unique <= 20 and bool(np.all(np.mod(s, 1) == 0)):
            return "classification"
        return "regression"
    return "classification" if c.n_unique <= MAX_CLASSES else "unsupported"


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return idx[:half], idx[half:]


def _encode_feature(
    train_x: pd.Series, train_y: pd.Series, test_x: pd.Series, numeric: bool
) -> pd.Series:
    """Out-of-sample single-feature prediction: mean target per level (categoricals) or per
    training quantile bin (numerics). Unseen levels fall back to the global mean, so
    unique identifiers score at chance level instead of perfectly."""
    prior = float(train_y.mean())
    if numeric:
        edges = np.unique(np.quantile(train_x.dropna(), np.linspace(0, 1, 21)))
        if edges.size < 2:
            return pd.Series(prior, index=test_x.index)
        # Open-ended outer bins; a mass point at the minimum keeps its own bin.
        edges = np.r_[-np.inf, edges[:-1], np.inf]
        tr = pd.cut(train_x, edges, labels=False)
        te = pd.cut(test_x, edges, labels=False)
    else:
        tr, te = train_x.astype("string"), test_x.astype("string")
    tr = tr.astype("object").where(tr.notna(), "<missing>")
    te = te.astype("object").where(te.notna(), "<missing>")
    means = pd.Series(train_y.to_numpy(), index=tr.to_numpy()).groupby(level=0).mean()
    return te.map(means).astype(float).fillna(prior)


def predictive_score(
    x: pd.Series, y: pd.Series, numeric: bool, task: str, classes: list[Any], seed: int = 0
) -> float | None:
    """Holdout predictive power of one feature, in [0, 1] (higher = more predictive).

    Binary classification: AUC of the encoded feature, folded so 0.5 = chance.
    Multiclass: macro one-vs-rest AUC. Regression: holdout R^2 (clipped at 0).
    """
    ok = y.notna()
    x, y = x[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
    if len(y) < 40 or x.notna().sum() < 20:
        return None
    tr, te = _split(len(y), seed)
    if task == "regression":
        yv = y.astype(float)
        pred = _encode_feature(x.iloc[tr], yv.iloc[tr], x.iloc[te], numeric)
        resid = float(((yv.iloc[te] - pred) ** 2).sum())
        total = float(((yv.iloc[te] - yv.iloc[tr].mean()) ** 2).sum())
        return max(0.0, 1 - resid / total) if total > 0 else None
    aucs = []
    targets = classes if len(classes) > 2 else classes[:1]
    for cls in targets:
        yb = (y.astype(str) == str(cls)).astype(float)
        if yb.iloc[te].nunique() < 2:
            continue
        pred = _encode_feature(x.iloc[tr], yb.iloc[tr], x.iloc[te], numeric)
        auc = roc_auc_score(yb.iloc[te], pred)
        aucs.append(max(auc, 1 - auc))
    if not aucs:
        return None
    return float(np.mean(aucs))


@register_detector
class TargetDetector(BaseDetector):
    name = "target"
    category = Category.TARGET
    description = (
        "With a target column configured: missing targets, class imbalance or target "
        "distribution, and single features that predict the target suspiciously well "
        "out of sample (a common symptom of leakage)."
    )
    assumptions = (
        "Predictive power is estimated on a random 50/50 split with a per-level/per-bin "
        "target-mean model, so memorising identifiers does not score well."
    )
    limitations = (
        "A highly predictive feature may be legitimate. Leakage depends on when a value "
        "becomes known, which data alone cannot reveal; findings say 'suspicious', never 'leak'."
    )

    def analyze(self, ctx: Context) -> list[Finding]:
        target = ctx.config.target
        if not target or target not in ctx.frame:
            return []
        t = ctx.thresholds
        out: list[Finding] = []
        y = ctx.frame[target]
        n_missing = int(y.isna().sum())
        task = task_type(ctx, target)
        section = ctx.section("target")
        section.update({"column": target, "task": task, "missing": n_missing})
        if n_missing:
            r = n_missing / ctx.n_rows
            out.append(
                self.finding(
                    "target.missing",
                    Severity.CRITICAL
                    if r == 1
                    else tiered(r, [(t.missing_warning, Severity.HIGH)], Severity.WARNING),
                    f"Missing target values in '{target}'",
                    f"{n_missing:,} row(s) ({pct(r)}) have no value for the target '{target}'.",
                    columns=[target],
                    evidence={"missing": n_missing, "ratio": r},
                    rule=f"All missing -> critical; >= missing_warning ({t.missing_warning}) -> high; otherwise warning.",
                    interpretation="Unlabelled rows cannot be used for supervised training and may differ systematically.",
                    suggestion="Find out why labels are missing before dropping these rows.",
                )
            )
        if task == "unsupported":
            return out
        classes: list[Any] = []
        if task == "classification":
            counts = y.dropna().astype(str).value_counts()
            classes = list(counts.index)
            shares = counts / counts.sum()
            section["classes"] = [
                {"label": ctx.label(k), "count": int(v), "share": float(shares[k])}
                for k, v in counts.head(MAX_CLASSES).items()
            ]
            if len(counts) < 2:
                out.append(
                    self.finding(
                        "target.single_class",
                        Severity.CRITICAL,
                        f"Target '{target}' has a single class",
                        "All labelled rows have the same target value.",
                        columns=[target],
                        rule="One class -> critical: nothing can be learned.",
                        suggestion="Check filtering that may have removed the other classes.",
                    )
                )
                return out
            minority = float(shares.min())
            if minority < t.imbalance_warning:
                out.append(
                    self.finding(
                        "target.imbalance",
                        Severity.HIGH if minority < t.imbalance_high else Severity.WARNING,
                        f"Class imbalance in '{target}'",
                        f"The rarest class ('{ctx.label(shares.idxmin())}') is {pct(minority)} of labelled rows "
                        f"({int(counts.min()):,} rows); the most common is {pct(float(shares.max()))}.",
                        columns=[target],
                        evidence={
                            "minority_share": minority,
                            "minority_count": int(counts.min()),
                            "classes": len(counts),
                        },
                        rule=f"Minority share < imbalance_high ({t.imbalance_high}) -> high; < imbalance_warning ({t.imbalance_warning}) -> warning.",
                        interpretation="Accuracy will be misleading; minority-class metrics need enough examples.",
                        suggestion="Use stratified splits and metrics such as PR-AUC or balanced accuracy.",
                    )
                )
            tiny = counts[counts < 10]
            if len(tiny) and len(counts) > 2:
                out.append(
                    self.finding(
                        "target.tiny_classes",
                        Severity.WARNING,
                        f"Classes with fewer than 10 rows in '{target}'",
                        f"{len(tiny)} class(es) of '{target}' have fewer than 10 labelled rows.",
                        columns=[target],
                        evidence={"classes": {ctx.label(k): int(v) for k, v in tiny.items()}},
                        rule="Any class with < 10 rows (multiclass) -> warning.",
                        suggestion="Merge or drop tiny classes; they cannot be stratified or evaluated reliably.",
                    )
                )
        out.extend(self._associations(ctx, target, y, task, classes))
        return out

    def _associations(
        self, ctx: Context, target: str, y: pd.Series, task: str, classes: list[Any]
    ) -> list[Finding]:
        t = ctx.thresholds
        sample = ctx.sample()
        ys = sample[target]
        if task == "regression":
            ys = ctx.numeric(target).loc[sample.index]
        scores: list[dict[str, Any]] = []
        out: list[Finding] = []
        for col, c in ctx.schema.columns.items():
            if col == target or c.column_type in (
                ColumnType.CONSTANT,
                ColumnType.UNKNOWN,
                ColumnType.TEXT,
            ):
                continue
            numeric = ctx.is_numeric_like(col) or (
                c.column_type == ColumnType.IDENTIFIER and ctx.frame[col].dtype.kind in "iuf"
            )
            x = ctx.numeric(col).loc[sample.index] if numeric else sample[col]
            if c.column_type == ColumnType.DATETIME:
                x = (
                    ctx.datetimes(col)
                    .loc[sample.index]
                    .astype("int64")
                    .where(ctx.datetimes(col).loc[sample.index].notna())
                )
                numeric = True
            score = predictive_score(
                x, ys, numeric, task, classes, ctx.config.performance.random_state
            )
            miss_score = None
            if c.n_missing >= 20 and c.n_present >= 20:
                miss_score = predictive_score(
                    sample[col].isna().astype(int),
                    ys,
                    True,
                    task,
                    classes,
                    ctx.config.performance.random_state,
                )
            scores.append(
                {
                    "column": col,
                    "score": score,
                    "missingness_score": miss_score,
                    "type": c.column_type.value,
                }
            )
            metric = "holdout R^2" if task == "regression" else "holdout AUC"
            if score is not None and score >= t.target_association_warning:
                out.append(
                    self.finding(
                        "target.suspicious_feature",
                        Severity.HIGH,
                        f"'{col}' alone predicts '{target}' almost perfectly",
                        f"A single-feature model on '{col}' reaches {metric} = {score:.3f} on held-out rows.",
                        columns=[col, target],
                        evidence={
                            "score": score,
                            "metric": metric,
                            "feature_type": c.column_type.value,
                        },
                        rule=f"Single-feature {metric} >= target_association_warning ({t.target_association_warning}) -> high.",
                        interpretation=(
                            "Either a genuinely strong predictor, or a value recorded after/because of the "
                            "outcome (target leakage). Correlation alone cannot tell these apart."
                        ),
                        suggestion=f"Verify when '{col}' becomes known relative to '{target}' in production.",
                        confidence=0.5,
                    )
                )
            elif c.column_type == ColumnType.IDENTIFIER and score is not None and score >= 0.7:
                out.append(
                    self.finding(
                        "target.identifier_predictive",
                        Severity.WARNING,
                        f"Identifier '{col}' carries information about '{target}'",
                        f"Binned values of identifier '{col}' reach {metric} = {score:.3f} on held-out rows.",
                        columns=[col, target],
                        evidence={"score": score, "metric": metric},
                        rule=f"Identifier with {metric} >= 0.7 -> warning.",
                        interpretation="IDs assigned in order can encode time or source; the data may be sorted by outcome.",
                        suggestion="Exclude the identifier and use time-aware or grouped validation.",
                        confidence=0.5,
                    )
                )
            if miss_score is not None and miss_score >= 0.75:
                out.append(
                    self.finding(
                        "target.missingness_predictive",
                        Severity.WARNING,
                        f"Whether '{col}' is missing predicts '{target}'",
                        f"The missing-indicator of '{col}' alone reaches {metric} = {miss_score:.3f}.",
                        columns=[col, target],
                        evidence={"score": miss_score, "metric": metric},
                        rule=f"Missing-indicator {metric} >= 0.75 -> warning.",
                        interpretation="Missingness is informative; it may be filled in only after the outcome is known.",
                        suggestion=f"Check the process that populates '{col}'.",
                        confidence=0.5,
                    )
                )
        scores.sort(key=lambda s: -(s["score"] or 0))
        ctx.section("target")["feature_scores"] = scores[:30]
        return out
