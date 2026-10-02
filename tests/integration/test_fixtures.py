"""Deterministic expectations on the committed CSV fixtures (also exercises the CSV loader)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasi import compare, inspect

DATA = Path(__file__).resolve().parents[1] / "data"
EXPECTED = json.loads((DATA / "expected.json").read_text())

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_fixture_findings(key: str) -> None:
    if key.startswith("compare:"):
        _, ref, cur = key.split(":")
        report = compare(DATA / ref, DATA / cur)
    else:
        report = inspect(DATA / key, config={"reference_time": "2026-10-01"})
    found = {f.code for f in report.findings}
    missing = set(EXPECTED[key]) - found
    assert not missing, f"{key}: expected {sorted(missing)}; got {sorted(found)}"


@pytest.mark.parametrize("key", sorted(k for k in EXPECTED if not k.startswith("compare:")))
def test_fixture_reports_are_deterministic(key: str) -> None:
    a = inspect(DATA / key, config={"reference_time": "2026-10-01"})
    b = inspect(DATA / key, config={"reference_time": "2026-10-01"})
    assert [f.to_dict() for f in a.findings] == [f.to_dict() for f in b.findings]
