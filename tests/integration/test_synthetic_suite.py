"""Synthetic-data suite: generated problems must be detected; evaluation metrics must be sane."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasi import compare, inspect
from datasi.evaluation import (
    Confusion,
    confusion,
    evaluate_findings,
    evaluate_rows,
    outlier_row_scores,
    summarize,
)
from datasi.synthetic import generate_dataset, make_drift_pair

pytestmark = pytest.mark.integration

ALL_ISSUES = {
    "missingness": 0.15,
    "duplicates": 0.03,
    "outliers": 0.01,
    "category_variants": 0.02,
    "numeric_as_text": True,
    "constant_column": True,
    "redundant_columns": True,
    "sentinels": 0.005,
    "placeholders": 0.01,
    "missing_after": 0.2,
    "missing_by_group": 0.7,
    "leakage": True,
}


def test_generator_is_deterministic() -> None:
    a = generate_dataset(500, seed=3, **ALL_ISSUES)  # type: ignore[arg-type]
    b = generate_dataset(500, seed=3, **ALL_ISSUES)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.truth.issues == b.truth.issues


def test_generator_records_truth() -> None:
    s = generate_dataset(1000, duplicates=0.05, outliers=0.02, seed=1)
    dup_rows = s.truth.rows("duplicates")
    assert len(dup_rows) == 50
    assert set(np.flatnonzero(s.frame.duplicated().to_numpy())) == dup_rows
    out_rows = s.truth.rows("outliers", "amount")
    assert len(out_rows) == 20
    med = s.frame["amount"].median()
    assert all(s.frame.loc[r, "amount"] > med for r in out_rows)


def test_all_planted_issues_are_detected() -> None:
    s = generate_dataset(6000, seed=11, **ALL_ISSUES)  # type: ignore[arg-type]
    report = inspect(s.frame, target=s.target, config={"reference_time": "2026-10-01"})
    scores = evaluate_findings(report.findings, s.truth)
    missed = {k: c for k, c in scores.items() if c.recall != 1.0}
    assert not missed, missed


def test_clean_data_has_no_issue_findings() -> None:
    s = generate_dataset(6000, seed=12)
    report = inspect(s.frame, target=s.target, config={"reference_time": "2026-10-01"})
    noisy = [
        f
        for f in report.filter(min_severity="warning")
        if f.code not in {"identifier.detected", "outliers.univariate"}
    ]
    assert noisy == [], [f.title for f in noisy]


def test_drift_pair_detected_and_null_pair_quiet() -> None:
    ref, cur, truth = make_drift_pair(4000, shift=0.5, category_shift=0.2, seed=2)
    report = compare(ref, cur)
    scores = evaluate_findings(report.findings, truth, kinds=["drift"])
    assert scores["drift"].recall == 1.0
    assert scores["drift"].fp == 0
    ref, cur, _ = make_drift_pair(4000, shift=0, category_shift=0, seed=3)
    quiet = compare(ref, cur)
    assert quiet.filter(min_severity="warning") == []


def test_confusion_metrics() -> None:
    c = confusion({"a", "b"}, {"b", "c"}, {"a", "b", "c", "d"})
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.precision == 0.5 and c.recall == 0.5 and c.fpr == 0.5 and c.fnr == 0.5
    assert Confusion(0, 0, 0, 3).precision is None  # undefined, not 0
    assert evaluate_rows(np.array([True, False, True]), {0}).fp == 1


def test_outlier_row_scores_and_summary() -> None:
    s = generate_dataset(3000, outliers=0.01, seed=4)
    scores = outlier_row_scores(s.frame, s.truth, methods=["iqr", "modified_zscore"])
    assert scores["modified_zscore"].recall is not None and scores["modified_zscore"].recall > 0.9
    table = summarize([scores, scores])
    assert set(table["issue"]) == {"iqr", "modified_zscore"}
    assert table.loc[table["issue"] == "iqr", "tp"].iloc[0] == 2 * scores["iqr"].tp
