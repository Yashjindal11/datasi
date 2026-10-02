import json

import numpy as np
import pytest

from datasi.configuration import Config, ConfigError, Thresholds
from datasi.findings import Category, Finding, Severity, to_jsonable


def make_finding(**kw: object) -> Finding:
    base: dict[str, object] = {
        "detector": "missingness",
        "code": "missing.high",
        "category": Category.MISSINGNESS,
        "severity": Severity.WARNING,
        "title": "High missingness",
        "observation": "18.4% of values are missing.",
        "columns": ("income",),
    }
    base.update(kw)
    return Finding.model_validate(base)


def test_severity_ordering() -> None:
    assert Severity.INFO.rank < Severity.WARNING.rank < Severity.HIGH.rank < Severity.CRITICAL.rank
    assert Severity.at_least(Severity.HIGH) == [Severity.HIGH, Severity.CRITICAL]


def test_evidence_is_json_safe() -> None:
    f = make_finding(
        evidence={
            "rate": np.float64(0.184),
            "n": np.int64(3),
            "nan": float("nan"),
            "a": np.arange(2),
        }
    )
    assert f.evidence == {"rate": 0.184, "n": 3, "nan": None, "a": [0, 1]}
    json.dumps(f.to_dict())


def test_id_is_stable_and_column_specific() -> None:
    assert make_finding().id == make_finding(title="other").id
    assert make_finding().id != make_finding(columns=("age",)).id


def test_explain_separates_observation_and_interpretation() -> None:
    text = make_finding(interpretation="May indicate a source change.", confidence=0.4).explain()
    assert "Observation: 18.4%" in text
    assert "Possible interpretation: May indicate" in text
    assert "Confidence: 0.40" in text


def test_confidence_bounds() -> None:
    with pytest.raises(ValueError):
        make_finding(confidence=1.5)


def test_to_jsonable_handles_infinities() -> None:
    assert to_jsonable([np.inf, 1.0]) == [None, 1.0]


def test_config_rejects_unknown_detector() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        Config.from_mapping({"detectors": {"nope": True}})


def test_config_rejects_unordered_thresholds() -> None:
    with pytest.raises(ValueError, match="missing_warning"):
        Thresholds(missing_warning=0.6, missing_high=0.5)


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        Config.from_mapping({"thresholds": {"missing_warn": 0.1}})


def test_config_yaml_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = Config.from_mapping(
        {"detectors": {"outliers": False}, "thresholds": {"correlation_warning": 0.8}}
    )
    path = tmp_path / "c.yaml"
    path.write_text(cfg.to_yaml())
    loaded = Config.from_file(path)
    assert not loaded.enabled("outliers")
    assert loaded.enabled("missingness")
    assert loaded.thresholds.correlation_warning == 0.8


def test_config_file_errors_are_readable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        Config.from_file(path)
