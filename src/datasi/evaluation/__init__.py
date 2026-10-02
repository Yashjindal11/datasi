"""Detector evaluation against synthetic ground truth.

Two granularities:

* **Column level** - did a detector flag the column that received an issue (and stay
  quiet on the others)? Universe = all columns of the dataset (or the dataset itself
  for dataset-level issues such as duplicates).
* **Row level** - for issues with row positions (outliers, duplicates), compare a
  row mask with the injected rows.

Metrics: precision, recall, false positive rate and false negative rate. Undefined
ratios (0/0) are reported as None, never silently as 0 or 1.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from datasi.findings import Finding, Severity
from datasi.synthetic import GroundTruth

ISSUE_CODES: dict[str, frozenset[str]] = {
    "missing": frozenset({"missing.low", "missing.high", "missing.complete"}),
    "duplicates": frozenset({"duplicates.exact"}),
    "outliers": frozenset({"outliers.univariate"}),
    "category_variants": frozenset({"categorical.variants"}),
    "numeric_as_text": frozenset({"types.numeric_text", "types.mostly_numeric"}),
    "constant": frozenset({"constant.single_value"}),
    "redundant": frozenset(
        {"redundancy.linear", "redundancy.linear_broken", "redundancy.identical"}
    ),
    "sum": frozenset({"redundancy.sum", "redundancy.sum_broken"}),
    "sentinel": frozenset({"numeric.sentinel"}),
    "placeholder": frozenset({"strings.placeholder"}),
    "missing_temporal": frozenset({"missing.over_time"}),
    "missing_group": frozenset({"missing.by_group"}),
    "leakage": frozenset({"target.suspicious_feature"}),
    "drift": frozenset({"drift.numeric", "drift.categorical"}),
}
DATASET_LEVEL = frozenset({"duplicates"})
# For multi-column issues the first column is the one that 'has' the issue.
_PRIMARY_ONLY = frozenset({"missing_group", "missing_temporal", "sum"})


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @staticmethod
    def _ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    @property
    def precision(self) -> float | None:
        return self._ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return self._ratio(self.tp, self.tp + self.fn)

    @property
    def fpr(self) -> float | None:
        return self._ratio(self.fp, self.fp + self.tn)

    @property
    def fnr(self) -> float | None:
        return self._ratio(self.fn, self.tp + self.fn)

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "precision": self.precision,
            "recall": self.recall,
            "fpr": self.fpr,
            "fnr": self.fnr,
        }


def confusion(predicted: set[Any], actual: set[Any], universe: set[Any]) -> Confusion:
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(universe - predicted - actual)
    return Confusion(tp, fp, fn, tn)


def _units(kind: str, columns: Iterable[str]) -> set[str]:
    cols = list(columns)
    if kind in DATASET_LEVEL:
        return {"*"}
    if kind in _PRIMARY_ONLY:
        return set(cols[:1])
    return set(cols)


def evaluate_findings(
    findings: list[Finding],
    truth: GroundTruth,
    kinds: Iterable[str] | None = None,
    min_severity: Severity = Severity.INFO,
) -> dict[str, Confusion]:
    """Column-level confusion per issue kind."""
    out: dict[str, Confusion] = {}
    for kind in kinds or truth.kinds():
        codes = ISSUE_CODES[kind]
        actual: set[str] = set()
        for issue in truth.of_kind(kind):
            actual |= _units(kind, issue.columns)
        predicted: set[str] = set()
        for f in findings:
            if f.code in codes and f.severity.rank >= min_severity.rank:
                predicted |= _units(kind, f.columns)
        universe = {"*"} if kind in DATASET_LEVEL else set(truth.columns)
        out[kind] = confusion(predicted, actual, universe)
    return out


def evaluate_rows(mask: np.ndarray, true_rows: set[int]) -> Confusion:
    predicted = set(np.flatnonzero(np.asarray(mask, dtype=bool)).tolist())
    return confusion(predicted, true_rows, set(range(len(mask))))


def outlier_row_scores(
    frame: pd.DataFrame,
    truth: GroundTruth,
    methods: Iterable[str] = ("iqr", "zscore", "modified_zscore", "isolation_forest"),
    **thresholds: Any,
) -> dict[str, Confusion]:
    """Row-level confusion of each outlier method on every column with injected outliers."""
    from datasi.statistics import outliers as ol

    out: dict[str, Confusion] = {}
    for method in methods:
        total = Confusion(0, 0, 0, 0)
        for issue in truth.of_kind("outliers"):
            col = issue.columns[0]
            mask = ol.outlier_mask(frame[col], method, **thresholds)  # type: ignore[arg-type]
            total = total + evaluate_rows(mask, set(issue.rows))
        out[method] = total
    return out


def summarize(trials: list[dict[str, Confusion]]) -> pd.DataFrame:
    """Pool confusion counts over trials (micro-average) per issue kind."""
    pooled: dict[str, Confusion] = {}
    for t in trials:
        for kind, c in t.items():
            pooled[kind] = pooled.get(kind, Confusion(0, 0, 0, 0)) + c
    rows = [{"issue": k, **v.to_dict()} for k, v in sorted(pooled.items())]
    return pd.DataFrame(rows)
