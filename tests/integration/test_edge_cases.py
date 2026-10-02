"""Edge cases and large-data behaviour."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from datasi import inspect, investigate
from datasi.detectors import BaseDetector, Context
from datasi.findings import Category, Finding, Severity


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": []}),
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [None] * 5, "b": [None] * 5}),
        pd.DataFrame({"a": [np.inf, -np.inf, 1.0, 2.0]}),
        pd.DataFrame({"a": pd.Series([[1], {"k": 2}, (3,), None], dtype=object)}),
        pd.DataFrame({"日本語": ["é", "ß", "😀", "\x00"], "b": [1, 2, 3, 4]}),
        pd.DataFrame({0: [1, 2], 1: ["x", "y"]}),
        pd.DataFrame({"t": pd.to_datetime(["2026-01-01", None, "2026-01-03"]).tz_localize("UTC")}),
        pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])}),
        pd.DataFrame(
            {
                "b": pd.array([True, None, False], dtype="boolean"),
                "i": pd.array([1, None, 3], dtype="Int64"),
            }
        ),
    ],
)
def test_edge_frames_do_not_crash(frame: pd.DataFrame) -> None:
    report = inspect(frame, config={"reference_time": "2026-10-01"})
    assert all(d.status == "ok" for d in report.detectors), [
        d.error for d in report.detectors if d.error
    ]
    report.to_html()
    report.to_markdown()


def test_input_not_modified() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=300),
            "s": rng.choice(["x", " y", "Y"], 300),
            "d": ["2026-01-01"] * 300,
        },
        index=np.arange(300) * 2,
    )
    before = frame.copy(deep=True)
    inspect(frame, target="s")
    pd.testing.assert_frame_equal(frame, before)


def test_custom_detector_plugin() -> None:
    class NoNegatives(BaseDetector):
        name = "no_negatives"
        category = Category.NUMERIC

        def analyze(self, ctx: Context) -> list[Finding]:
            cols = [
                c
                for c in ctx.frame
                if pd.api.types.is_numeric_dtype(ctx.frame[c]) and (ctx.frame[c] < 0).any()
            ]
            return [
                self.finding("custom.neg", Severity.HIGH, f"Negatives in {c}", "found", columns=[c])
                for c in cols
            ]

    report = investigate(
        pd.DataFrame({"x": [1, -1]}), detectors=[], extra_detectors=[NoNegatives()]
    )
    assert [f.code for f in report.findings] == ["custom.neg"]


def test_failing_detector_is_reported_not_raised() -> None:
    class Broken(BaseDetector):
        name = "broken"
        category = Category.SCHEMA

        def analyze(self, ctx: Context) -> list[Finding]:
            raise RuntimeError("boom")

    report = investigate(pd.DataFrame({"x": [1]}), detectors=["schema"], extra_detectors=[Broken()])
    run = next(d for d in report.detectors if d.name == "broken")
    assert run.status == "error" and "boom" in (run.error or "")
    assert any(f.code == "datasi.detector_error" for f in report.findings)


def test_unknown_detector_name() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        investigate(pd.DataFrame({"x": [1]}), detectors=["nope"])


@pytest.mark.slow
def test_large_dataset_runs_and_samples() -> None:
    from datasi.synthetic import generate_dataset

    s = generate_dataset(400_000, missingness=0.05, duplicates=0.01, outliers=0.005, seed=5)
    start = time.perf_counter()
    report = inspect(s.frame, target=s.target, config={"performance": {"sample_rows": 50_000}})
    elapsed = time.perf_counter() - start
    assert report.dataset.analysis_sample_rows == 50_000
    dup = next(f for f in report.findings if f.code == "duplicates.exact")
    assert dup.evidence["duplicate_rows"] == 4000  # counts stay exact under sampling
    assert elapsed < 120
