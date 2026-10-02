from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from datasi import investigate
from datasi.findings import Finding


def run_detector(frame: pd.DataFrame, name: str | list[str], **kwargs: Any) -> list[Finding]:
    names = [name] if isinstance(name, str) else name
    kwargs.setdefault("config", {})
    kwargs["config"].setdefault("reference_time", "2026-10-01")
    report = investigate(frame, detectors=names, **kwargs)
    errors = [d for d in report.detectors if d.status == "error"]
    assert not errors, errors
    return report.findings


def codes(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings}


def by_code(findings: list[Finding], code: str) -> list[Finding]:
    return [f for f in findings if f.code == code]


@pytest.fixture
def rng():  # type: ignore[no-untyped-def]
    import numpy as np

    return np.random.default_rng(1234)
