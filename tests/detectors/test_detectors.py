"""Each detector tested independently on data with one planted problem (and without it)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import by_code, codes, run_detector
from datasi.findings import Severity

N = 1000


# --------------------------------------------------------------------------- schema
def test_schema_detector() -> None:
    frame = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "a", "A ", "Unnamed: 0"])
    found = run_detector(frame, "schema", config={"target": "y"})
    assert {
        "schema.duplicate_name",
        "schema.name_collision",
        "schema.name_whitespace",
        "schema.exported_index",
        "schema.config_column_missing",
    } <= codes(found)
    assert by_code(found, "schema.config_column_missing")[0].severity == Severity.CRITICAL


def test_schema_empty_dataset() -> None:
    found = run_detector(pd.DataFrame({"a": []}), "schema")
    assert by_code(found, "schema.empty")[0].severity == Severity.CRITICAL


# --------------------------------------------------------------------- missingness
def test_missingness_levels(rng) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "low": rng.normal(size=N),
            "warn": rng.normal(size=N),
            "high": rng.normal(size=N),
            "none": [np.nan] * N,
            "ok": rng.normal(size=N),
        }
    )
    frame.loc[:9, "low"] = np.nan
    frame.loc[:199, "warn"] = np.nan
    frame.loc[:699, "high"] = np.nan
    found = run_detector(frame, "missingness")
    sev = {
        f.columns[0]: f.severity for f in found if f.code in ("missing.high", "missing.complete")
    }
    assert sev == {"warn": Severity.WARNING, "high": Severity.HIGH, "none": Severity.HIGH}
    assert "low" in by_code(found, "missing.low")[0].columns
    assert "ok" not in {c for f in found for c in f.columns}


def test_missingness_by_group(rng) -> None:  # type: ignore[no-untyped-def]
    group = rng.choice(["a", "b", "c"], N)
    income = pd.Series(rng.normal(50, 5, N))
    income[(group == "c") & (rng.uniform(size=N) < 0.8)] = np.nan
    found = run_detector(pd.DataFrame({"group": group, "income": income}), "missingness")
    f = by_code(found, "missing.by_group")[0]
    assert f.columns == ("income", "group")
    assert f.evidence["groups"][0]["group"] == "c"
    assert f.interpretation and "possib" not in f.observation.lower()


def test_missingness_over_time(rng) -> None:  # type: ignore[no-untyped-def]
    ts = pd.date_range("2026-01-01", periods=N, freq="6h")
    v = pd.Series(rng.normal(size=N))
    v[ts >= "2026-06-01"] = np.where(
        rng.uniform(size=(ts >= "2026-06-01").sum()) < 0.6, np.nan, 1.0
    )
    found = run_detector(pd.DataFrame({"event_time": ts, "v": v}), "missingness")
    f = by_code(found, "missing.over_time")[0]
    assert f.evidence["change_at"].startswith("2026-0")
    assert f.evidence["rate_after"] > f.evidence["rate_before"] + 0.4
    assert f.confidence is not None and f.confidence < 1


def test_missingness_cooccurrence_and_related(rng) -> None:  # type: ignore[no-untyped-def]
    x = rng.normal(size=N)
    joint = x > 1.0  # missing together, and only for large x
    frame = pd.DataFrame(
        {"x": x, "a": np.where(joint, np.nan, 1.0), "b": np.where(joint, np.nan, 2.0)}
    )
    found = run_detector(frame, "missingness")
    assert set(by_code(found, "missing.co_occurrence")[0].columns) == {"a", "b"}
    assert by_code(found, "missing.related_numeric")[0].evidence["related_column"] == "x"


def test_missingness_empty_rows() -> None:
    frame = pd.DataFrame({"a": [1, None, 3], "b": [1, None, 2]})
    assert "missing.empty_rows" in codes(run_detector(frame, "missingness"))


# ---------------------------------------------------------------------- duplicates
def test_duplicates(rng) -> None:  # type: ignore[no-untyped-def]
    base = pd.DataFrame(
        {"id": np.arange(N), "a": rng.integers(0, 1000, N), "b": rng.normal(size=N)}
    )
    dup = pd.concat([base, base.iloc[:20]], ignore_index=True)
    found = run_detector(dup, "duplicates")
    f = by_code(found, "duplicates.exact")[0]
    assert f.evidence["duplicate_rows"] == 20
    assert f.severity == Severity.WARNING  # 2% -> between warning (0.1%) and high (5%)
    assert by_code(found, "duplicates.key")[0].columns == ("id",)
    reissued = pd.concat([base, base.iloc[:30].assign(id=np.arange(N, N + 30))], ignore_index=True)
    assert (
        by_code(run_detector(reissued, "duplicates"), "duplicates.except_ids")[0].evidence["rows"]
        == 30
    )


def test_no_duplicates_in_clean_data(rng) -> None:  # type: ignore[no-untyped-def]
    assert run_detector(pd.DataFrame({"a": rng.normal(size=N)}), "duplicates") == []


def test_unhashable_values_do_not_crash() -> None:
    frame = pd.DataFrame({"a": [[1, 2], [1, 2], {"x": 1}], "b": [1, 1, 2]})
    assert (
        by_code(run_detector(frame, "duplicates"), "duplicates.exact")[0].evidence["duplicate_rows"]
        == 1
    )


def test_conflicting_entity_attributes(rng) -> None:  # type: ignore[no-untyped-def]
    cust = np.repeat(np.arange(200), 5)
    tier = np.array(["gold", "silver"])[cust % 2]
    tier = tier.copy()
    tier[[0, 10]] = "bronze"  # two customers get a conflicting tier once
    frame = pd.DataFrame({"customer_id": cust, "tier": tier, "v": rng.normal(size=1000)})
    f = by_code(run_detector(frame, "duplicates"), "duplicates.entity_conflict")[0]
    assert f.evidence["conflicting_keys"] == 2


# ------------------------------------------------------------- constant/identifier
def test_constant_near_constant_dominant(rng) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "const": ["x"] * N,
            "near": ["ACTIVE"] * 998 + ["CLOSED"] * 2,
            "dominant": ["a"] * 960 + ["b"] * 40,
            "fine": rng.choice(list("abc"), N),
        }
    )
    found = run_detector(frame, "constant")
    assert {f.columns[0]: f.code for f in found} == {
        "const": "constant.single_value",
        "near": "constant.near",
        "dominant": "constant.dominant",
    }
    assert "useless" not in " ".join(str(f.interpretation) for f in found).lower()


def test_identifier_detector() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": np.arange(N),
            "uuid": [f"{i:08x}-0000-4000-8000-{i:012x}" for i in range(N)],
            "amount": np.linspace(1, 2, N),
        }
    )
    found = run_detector(frame, "identifier")
    by_col = {f.columns[0]: f for f in found if f.code == "identifier.detected"}
    assert set(by_col) == {"transaction_id", "uuid"}
    assert by_col["transaction_id"].severity == Severity.WARNING  # numeric id
    assert any("monotonic" in r for r in by_col["transaction_id"].evidence["reasons"])


# --------------------------------------------------------------------------- types
def test_types_detector(rng) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "price": [f"${x:,.2f}" for x in rng.uniform(1, 5000, N)],
            "mostly": [str(i) for i in range(N - 100)] + ["N/A"] * 50 + ["unknown"] * 50,
            "dates": ["2026-01-01", "01/02/2026", "2026/03/01", "2026-01-05"] * (N // 4),
            "flags": rng.choice(["Yes", "Y", "No", "no"], N),
            "mixed": pd.Series(["a", 1, 2.5, "b"] * (N // 4), dtype=object),
        }
    )
    found = run_detector(frame, "types")
    got = {(f.code, f.columns[0]) for f in found}
    assert ("types.numeric_text", "price") in got
    assert ("types.mostly_numeric", "mostly") in got
    assert ("types.date_formats", "dates") in got
    assert ("types.boolean_text", "flags") in got
    assert ("types.mixed", "mixed") in got
    shapes = next(
        f for f in by_code(found, "types.mostly_numeric") if f.columns == ("mostly",)
    ).evidence["non_numeric_formats"]
    assert {s["shape"] for s in shapes} == {"A/A", "AAAAAAA"}  # shapes, not raw values


def test_timezone_mixture() -> None:
    frame = pd.DataFrame(
        {"ts": ["2026-01-01T10:00:00Z", "2026-01-01T11:00:00+02:00", "2026-01-01 12:00:00"] * 30}
    )
    assert "types.timezone_mixed" in codes(run_detector(frame, "types"))


def test_types_does_not_modify_data() -> None:
    frame = pd.DataFrame({"x": ["1", "2", "3"] * 10})
    before = frame.copy()
    run_detector(frame, ["types", "numeric", "outliers"])
    pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------- rules
def test_rules_confirm_invalid(rng) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "age": [25, 40, -3, 130, None],
            "code": ["AB1", "AB2", "xx", "AB3", "AB4"],
            "status": ["a", "b", "c", "a", "z"],
        }
    )
    rules = {
        "age": {"min": 0, "max": 120, "nullable": False},
        "code": {"pattern": r"AB\d"},
        "status": {"allowed": ["a", "b", "c"]},
    }
    found = run_detector(frame, "rules", config={"rules": rules})
    got = {(f.code, f.columns[0], f.evidence["violations"]) for f in found}
    assert got == {
        ("rules.min", "age", 1),
        ("rules.max", "age", 1),
        ("rules.nullable", "age", 1),
        ("rules.pattern", "code", 1),
        ("rules.allowed", "status", 1),
    }
    assert all(f.severity == Severity.HIGH and f.confidence == 1.0 for f in found)


# -------------------------------------------------------------------------- numeric
def test_numeric_shape(rng) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "skewed": rng.lognormal(0, 1.5, N),
            "zeros": np.where(rng.uniform(size=N) < 0.7, 0, rng.uniform(1, 10, N)),
            "refunds": np.r_[rng.uniform(10, 100, N - 3), [-5, -7, -9]],
            "sentinel": np.r_[rng.normal(50, 5, N - 20), [-999.0] * 20],
            "spike": np.r_[rng.normal(50, 5, N - 200), [42.0] * 200],
            "inf": np.r_[rng.normal(size=N - 1), [np.inf]],
        }
    )
    got = {(f.code, f.columns[0]) for f in run_detector(frame, "numeric")}
    assert ("numeric.skewed", "skewed") in got
    assert ("numeric.zero_inflated", "zeros") in got
    assert ("numeric.rare_negative", "refunds") in got
    assert ("numeric.sentinel", "sentinel") in got
    assert ("numeric.value_spike", "spike") in got
    assert ("numeric.infinite", "inf") in got


def test_numeric_clean_normal_has_no_findings(rng) -> None:  # type: ignore[no-untyped-def]
    assert run_detector(pd.DataFrame({"x": rng.normal(100, 10, N)}), "numeric") == []


def test_outliers_detector(rng) -> None:  # type: ignore[no-untyped-def]
    x = rng.normal(100, 10, N)
    x[[3, 300]] = [1000, -800]
    found = run_detector(pd.DataFrame({"x": x}), "outliers")
    f = by_code(found, "outliers.univariate")[0]
    assert f.severity == Severity.WARNING
    assert {3, 300} <= set(f.evidence["row_examples"])
    assert "not confirmed errors" in str(f.interpretation)


def test_outliers_lognormal_scored_on_log_scale(rng) -> None:  # type: ignore[no-untyped-def]
    found = run_detector(pd.DataFrame({"amount": rng.lognormal(3, 0.8, 5000)}), "outliers")
    assert all(f.severity == Severity.INFO for f in found)
    assert all(f.evidence["log_scale"] for f in found)


def test_outlier_methods_configurable(rng) -> None:  # type: ignore[no-untyped-def]
    x = rng.normal(size=500)
    x[0] = 50
    found = run_detector(
        pd.DataFrame({"x": x, "y": rng.normal(size=500)}),
        "outliers",
        config={"outlier_methods": ["zscore", "isolation_forest"]},
    )
    assert set(by_code(found, "outliers.univariate")[0].evidence["per_method"]) == {"zscore"}
    assert "outliers.multivariate" in codes(found)


# ---------------------------------------------------------------------- categorical
def test_categorical_variants_and_rare(rng) -> None:  # type: ignore[no-untyped-def]
    airline = rng.choice(["United", "Delta", "American"], N).astype(object)
    airline[:15] = "UNITED"
    airline[15:25] = "united "
    airline[25:30] = "Untied"
    airline[30:32] = "Spirit"
    airline[32:34] = "Frontier"
    found = run_detector(pd.DataFrame({"airline": airline}), "categorical")
    v = by_code(found, "categorical.variants")[0]
    assert {"case", "whitespace"} <= set(v.evidence["kinds"])
    labels = {x["label"] for x in v.evidence["groups"][0]}
    assert {"United", "UNITED", "united "} <= labels
    assert by_code(found, "categorical.similar_labels")
    assert by_code(found, "categorical.rare")


def test_category_explosion() -> None:
    frame = pd.DataFrame({"label": [f"c{i % 700}" for i in range(N)]})
    assert "categorical.explosion" in codes(run_detector(frame, "categorical"))


def test_redaction(rng) -> None:  # type: ignore[no-untyped-def]
    vals = rng.choice(["Alice", "Bob"], N).astype(object)
    vals[:5] = "ALICE"
    found = run_detector(
        pd.DataFrame({"name": vals}), "categorical", config={"privacy": {"redact_labels": True}}
    )
    text = str([f.evidence for f in found])
    assert "Alice" not in text and "<label:" in text


# -------------------------------------------------------------------------- strings
def test_strings_hygiene(rng) -> None:  # type: ignore[no-untyped-def]
    base = list(rng.choice(["alpha", "beta", "gamma"], N - 20))
    vals = base + ["", "  ", "N/A", "null", " alpha", "beta\u200b", "x\x00"] + ["alpha"] * 13
    found = run_detector(pd.DataFrame({"s": vals}), "strings")
    assert {
        "strings.empty",
        "strings.placeholder",
        "strings.padding",
        "strings.unusual_characters",
    } <= codes(found)


def test_strings_format_breaks() -> None:
    ids = [f"TX-{i:06d}" for i in range(N)]
    ids[5] = "TX-12"
    ids[9] = "tx_000009"
    found = run_detector(pd.DataFrame({"txn_ref": ids}), "strings")
    f = by_code(found, "strings.format")[0]
    assert f.evidence["dominant_format"] == "A+-9+"


# ------------------------------------------------------------------------- datetime
def test_datetime_detector(rng) -> None:  # type: ignore[no-untyped-def]
    ts = pd.Series(pd.date_range("2025-01-01", periods=N, freq="h"))
    ts[[10, 11, 12]] = pd.Timestamp("1900-01-01")
    ts[20] = pd.Timestamp("2030-01-01")
    found = run_detector(pd.DataFrame({"created_at": ts}), "datetime")
    assert {"datetime.placeholder", "datetime.future"} <= codes(found)
    assert by_code(found, "datetime.future")[0].severity == Severity.WARNING


def test_future_ok_for_expiry() -> None:
    frame = pd.DataFrame({"expiry_date": pd.date_range("2026-06-01", periods=300, freq="D")})
    assert by_code(run_detector(frame, "datetime"), "datetime.future")[0].severity == Severity.INFO


def test_unparseable_dates() -> None:
    vals = [f"2026-01-{d:02d}" for d in range(1, 29)] * 4 + ["2026-02-30", "not a date"]
    vals = vals + [f"2026-02-{d:02d}" for d in range(1, 9)] * 3
    found = run_detector(pd.DataFrame({"day": vals}), "datetime")
    assert by_code(found, "datetime.unparseable")[0].evidence["rows"] == 2


def test_regular_series_gaps() -> None:
    ts = pd.Series(pd.date_range("2026-01-01", periods=500, freq="h"))
    ts = ts[(ts < "2026-01-05") | (ts > "2026-01-07")]
    found = run_detector(pd.DataFrame({"timestamp": ts.reset_index(drop=True)}), "datetime")
    f = by_code(found, "datetime.irregular_interval")[0]
    assert f.severity == Severity.WARNING
    assert f.evidence["gaps"][0]["from"].startswith("2026-01-04T23")


# ------------------------------------------------------------------------- temporal
def test_temporal_volume_and_shift(rng) -> None:  # type: ignore[no-untyped-def]
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    counts = np.where(days < "2025-07-01", 40, 12)
    ts = np.repeat(days.values, counts)
    n = ts.size
    after = ts >= np.datetime64("2025-07-01")
    value = np.where(after, rng.normal(80, 5, n), rng.normal(50, 5, n))
    cat = np.where(after, rng.choice(["a", "b", "new"], n), rng.choice(["a", "b", "old"], n))
    found = run_detector(pd.DataFrame({"event_date": ts, "value": value, "cat": cat}), "temporal")
    got = codes(found)
    assert {
        "temporal.volume_shift",
        "temporal.distribution_shift",
        "temporal.new_category",
        "temporal.vanished_category",
    } <= got
    vol = by_code(found, "temporal.volume_shift")[0]
    assert vol.evidence["relative_change"] < -0.5


def test_temporal_stationary_has_no_shift(rng) -> None:  # type: ignore[no-untyped-def]
    ts = pd.date_range("2025-01-01", periods=5000, freq="2h")
    frame = pd.DataFrame({"event_time": ts, "v": rng.normal(size=5000)})
    found = run_detector(frame, "temporal")
    assert "temporal.distribution_shift" not in codes(found)
    assert "temporal.volume_shift" not in codes(found)


# -------------------------------------------------------------------- relationships
def test_correlation_detector(rng) -> None:  # type: ignore[no-untyped-def]
    a = rng.normal(size=N)
    c = rng.choice(list("xyz"), N)
    frame = pd.DataFrame(
        {
            "a": a,
            "b": a + rng.normal(0, 0.05, N),
            "u": rng.normal(size=N),
            "c": c,
            "c2": pd.Series(c).map({"x": "X", "y": "Y", "z": "Z"}),
        }
    )
    frame["sq"] = frame["u"] ** 2 + rng.normal(0, 0.01, N)
    found = run_detector(frame, "correlation")
    got = {(f.code, frozenset(f.columns)) for f in found}
    assert ("correlation.numeric", frozenset({"a", "b"})) in got
    assert ("correlation.categorical", frozenset({"c", "c2"})) in got
    assert ("correlation.nonlinear", frozenset({"u", "sq"})) in got
    assert all(
        "caus" not in (f.interpretation or "") or "not imply" in (f.interpretation or "")
        for f in found
    )


def test_redundancy_detector(rng) -> None:  # type: ignore[no-untyped-def]
    cm = rng.uniform(150, 200, N).round(1)
    x, y = rng.uniform(0, 10, N).round(2), rng.uniform(0, 10, N).round(2)
    total = x + y
    total[:20] += 5  # 2% broken
    frame = pd.DataFrame(
        {"height_cm": cm, "height_m": cm / 100, "x": x, "y": y, "total": total, "x_copy": x}
    )
    found = run_detector(frame, "redundancy")
    got = {(f.code, frozenset(f.columns)) for f in found}
    assert ("redundancy.linear", frozenset({"height_cm", "height_m"})) in got
    assert ("redundancy.identical", frozenset({"x", "x_copy"})) in got
    broken = [f for f in found if f.code == "redundancy.sum_broken"]
    assert broken and broken[0].evidence["breaking_rows"] == 20


def test_one_to_one_mapping(rng) -> None:  # type: ignore[no-untyped-def]
    code = rng.choice(["US", "CA", "MX"], N)
    name = pd.Series(code).map({"US": "United States", "CA": "Canada", "MX": "Mexico"})
    found = run_detector(pd.DataFrame({"code": code, "name": name}), "redundancy")
    assert "redundancy.one_to_one" in codes(found)


# ---------------------------------------------------------------------------- target
def test_target_imbalance_and_leakage(rng) -> None:  # type: ignore[no-untyped-def]
    y = (rng.uniform(size=4000) < 0.03).astype(int)
    frame = pd.DataFrame(
        {
            "y": y,
            "leak": y * 10 + rng.normal(0, 0.1, 4000),
            "noise": rng.normal(size=4000),
            "row_id": np.arange(4000),
        }
    )
    found = run_detector(frame, "target", target="y")
    assert by_code(found, "target.imbalance")[0].severity == Severity.WARNING
    leak = by_code(found, "target.suspicious_feature")
    assert [f.columns[0] for f in leak] == ["leak"]
    assert "leak" not in leak[0].title.lower().replace(
        "'leak'", ""
    )  # says 'predicts', never claims leakage
    assert leak[0].confidence == 0.5


def test_target_sorted_identifier(rng) -> None:  # type: ignore[no-untyped-def]
    y = np.r_[np.zeros(1500), np.ones(1500)].astype(int)
    frame = pd.DataFrame({"y": y, "record_id": np.arange(3000), "x": rng.normal(size=3000)})
    found = run_detector(frame, "target", target="y")
    assert any(f.columns[0] == "record_id" for f in found)


def test_target_missing_and_single_class() -> None:
    frame = pd.DataFrame({"y": [1, 1, 1, None] * 25, "x": range(100)})
    found = run_detector(frame, "target", target="y")
    assert {"target.missing", "target.single_class"} <= codes(found)


def test_regression_target(rng) -> None:  # type: ignore[no-untyped-def]
    x = rng.normal(size=3000)
    frame = pd.DataFrame(
        {"price": x * 3 + rng.normal(0, 0.01, 3000), "x": x, "z": rng.normal(size=3000)}
    )
    found = run_detector(frame, "target", target="price")
    assert [f.columns[0] for f in by_code(found, "target.suspicious_feature")] == ["x"]


@pytest.mark.parametrize(
    "name",
    ["missingness", "duplicates", "numeric", "categorical", "temporal", "correlation", "target"],
)
def test_detectors_on_degenerate_frames(name: str) -> None:
    for frame in [
        pd.DataFrame(),
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [None, None]}),
        pd.DataFrame({"a": ["é", "日本"], "b": [1, 2]}),
    ]:
        run_detector(
            frame, name, config={"target": "a"} if name == "target" and "a" in frame else {}
        )
