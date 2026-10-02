from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from datasi import Report, inspect
from datasi.reports.html import embed_json


@pytest.fixture(scope="module")
def report() -> Report:
    rng = np.random.default_rng(0)
    n = 800
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=n, freq="12h"),
            "city": rng.choice(
                ["Paris", "PARIS", "Lyon", "</script><img src=x onerror=alert(1)>"], n
            ),
            "amount": rng.lognormal(3, 1, n),
            "y": rng.integers(0, 2, n),
        }
    )
    frame.loc[:100, "amount"] = np.nan
    return inspect(frame, target="y", name="xss <test>", config={"reference_time": "2026-10-01"})


def test_json_round_trip(report: Report) -> None:
    again = Report.from_json(report.to_json())
    assert [f.to_dict() for f in again.findings] == [f.to_dict() for f in report.findings]
    assert again.summary == report.summary
    data = json.loads(report.to_json())
    assert {"summary", "recommendations", "schema", "profiles", "sections", "detectors"} <= set(
        data
    )


def test_html_is_self_contained_and_escaped(report: Report) -> None:
    html = report.to_html()
    assert html.startswith("<!doctype html>")
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html
    assert "default-src 'none'" in html  # CSP: no network access
    # Raw data labels must never appear unescaped in markup.
    assert "<img src=x" not in html
    assert "</script><img" not in html
    assert "xss &lt;test&gt;" in html
    blob = re.search(r'<script type="application/json" id="datasi-data">(.*?)</script>', html, re.S)
    assert blob is not None
    assert json.loads(blob.group(1))["dataset"]["name"] == "xss <test>"
    assert "innerHTML" not in html


def test_embed_json_escapes_markup() -> None:
    text = embed_json({"x": "</script><b>&"})
    assert "<" not in text and ">" not in text and "&" not in text
    assert json.loads(text) == {"x": "</script><b>&"}


def test_markdown_structure(report: Report) -> None:
    md = report.to_markdown()
    for heading in [
        "## Executive Summary",
        "## Dataset Overview",
        "## Schema",
        "## Critical and High Findings",
        "## Recommendations",
    ]:
        assert heading in md
    assert "**Observation.**" in md


def test_summary_and_filters(report: Report) -> None:
    s = report.summary
    assert s.findings == len(report.findings)
    assert sum(s.by_severity.values()) == s.findings
    assert "DataSI report for" in str(s)
    assert all(f.severity.rank >= 1 for f in report.filter(min_severity="warning"))
    assert all("amount" in f.columns for f in report.filter(column="amount"))
    ranks = [f.severity.rank for f in report.findings]
    assert ranks == sorted(ranks, reverse=True)


def test_recommendations_cite_findings(report: Report) -> None:
    ids = {f.id for f in report.findings}
    for r in report.recommendations:
        assert r["finding_id"] in ids
        assert r["because"]


def test_save_formats(tmp_path, report: Report) -> None:  # type: ignore[no-untyped-def]
    for ext in ("html", "md", "json"):
        assert report.save(tmp_path / f"r.{ext}").stat().st_size > 100
    with pytest.raises(ValueError):
        report.save(tmp_path / "r.pdf")


def test_no_raw_values_by_default() -> None:
    frame = pd.DataFrame(
        {"email": [f"person{i}@example.com" for i in range(300)], "age": [30] * 299 + [-5]}
    )
    r = inspect(frame, config={"rules": {"age": {"min": 0}}})
    assert "person1@example.com" not in r.to_json()
    r2 = inspect(
        frame, config={"rules": {"age": {"min": 0}}, "privacy": {"include_examples": True}}
    )
    assert any(f.evidence.get("examples") == ["-5"] for f in r2.findings)


def test_sampling_is_disclosed() -> None:
    frame = pd.DataFrame({"a": np.arange(5000), "b": np.random.default_rng(0).normal(size=5000)})
    r = inspect(frame, config={"performance": {"sample_rows": 1000}})
    assert r.dataset.analysis_sample_rows == 1000
    assert "1,000-row sample" in r.to_markdown()
