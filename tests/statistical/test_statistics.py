"""Statistical correctness tests: known distributions, known answers."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hs
from scipy import stats as sps

from datasi.statistics import association as assoc
from datasi.statistics import changepoint as cp
from datasi.statistics import descriptive as desc
from datasi.statistics import distribution as dist
from datasi.statistics import outliers as out
from datasi.statistics.streaming import KMVSketch, RunningMoments, profile_chunks

RNG = np.random.default_rng(42)


def test_numeric_summary_matches_scipy() -> None:
    x = RNG.gamma(2.0, 2.0, 5000)
    s = desc.numeric_summary(pd.Series(np.append(x, [np.nan, np.inf])))
    assert s["count"] == 5000
    assert s["infinite"] == 1
    assert s["mean"] == pytest.approx(x.mean())
    assert s["std"] == pytest.approx(x.std(ddof=1))
    assert s["skewness"] == pytest.approx(sps.skew(x, bias=False))
    assert s["excess_kurtosis"] == pytest.approx(sps.kurtosis(x, bias=False))
    assert s["iqr"] == pytest.approx(np.subtract(*np.quantile(x, [0.75, 0.25])))


def test_numeric_summary_degenerate() -> None:
    assert desc.numeric_summary(pd.Series([], dtype=float)) == {"count": 0, "infinite": 0}
    s = desc.numeric_summary(pd.Series([5.0, 5.0, 5.0, 5.0]))
    assert s["skewness"] is None
    assert s["excess_kurtosis"] is None


def test_entropy() -> None:
    assert desc.entropy_bits(np.array([1, 1, 1, 1])) == pytest.approx(2.0)
    assert desc.normalized_entropy(np.array([5, 5])) == pytest.approx(1.0)
    assert desc.normalized_entropy(np.array([10])) == 0.0


def test_outlier_methods_on_normal_data_have_expected_false_positive_rate() -> None:
    x = pd.Series(RNG.normal(0, 1, 100_000))
    # Theoretical two-sided rates under normality.
    assert out.zscore_mask(x, 3.0).mean() == pytest.approx(0.0027, abs=0.0008)
    assert out.iqr_mask(x, 1.5).mean() == pytest.approx(0.0070, abs=0.0015)
    assert out.modified_zscore_mask(x, 3.5).mean() < 0.002


def test_outlier_methods_find_planted_values() -> None:
    x = RNG.normal(10, 1, 1000)
    x[[5, 500]] = [60, -40]
    s = pd.Series(x)
    for method in ["iqr", "zscore", "modified_zscore"]:
        mask = out.outlier_mask(s, method)  # type: ignore[arg-type]
        assert mask[5]
        assert mask[500]


def test_iqr_zero_flags_nothing() -> None:
    s = pd.Series([1.0] * 90 + list(range(10)))
    assert not out.iqr_mask(s).any()


def test_isolation_forest_flags_multivariate_outlier() -> None:
    a = RNG.normal(0, 1, 2000)
    frame = pd.DataFrame({"a": a, "b": a + RNG.normal(0, 0.05, 2000)})
    frame.loc[7, ["a", "b"]] = [3.0, -3.0]  # each value plausible alone, jointly odd
    mask = out.isolation_forest_mask(frame, contamination=0.01)
    assert mask[7]


def test_ks_and_psi_identical_vs_shifted() -> None:
    a = pd.Series(RNG.normal(0, 1, 5000))
    b = pd.Series(RNG.normal(0, 1, 5000))
    c = pd.Series(RNG.normal(1, 1, 5000))
    same = dist.compare_numeric(a, b)
    shifted = dist.compare_numeric(a, c)
    assert same["ks_statistic"] < 0.05
    assert same["psi"] < 0.02
    assert shifted["ks_statistic"] > 0.3
    assert shifted["psi"] > 0.25
    # Mean shift of 1 SD of a normal is 1/1.349 IQRs.
    assert shifted["wasserstein_iqr"] == pytest.approx(1 / 1.349, rel=0.1)


def test_jsd_bounds() -> None:
    assert dist.jensen_shannon(np.array([1, 0]), np.array([0, 1])) == pytest.approx(1.0)
    assert dist.jensen_shannon(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    res = dist.compare_categorical(pd.Series(list("aab")), pd.Series(list("abc")))
    assert res["unseen_categories"] == 1


def test_cramers_v() -> None:
    x = pd.Series(RNG.choice(list("abc"), 3000))
    assert assoc.cramers_v(x, x) == pytest.approx(1.0, abs=0.01)
    y = pd.Series(RNG.choice(list("xyz"), 3000))
    v = assoc.cramers_v(x, y)
    assert v is not None
    assert v < 0.05


def test_correlation_ratio_and_nmi() -> None:
    c = pd.Series(RNG.choice(list("ab"), 2000))
    v = pd.Series(np.where(c == "a", 0.0, 10.0) + RNG.normal(0, 0.1, 2000))
    eta = assoc.correlation_ratio(c, v)
    assert eta is not None
    assert eta > 0.99
    nmi = assoc.normalized_mutual_information(c, v)
    assert nmi is not None
    assert nmi > 0.5


def test_cusum_detects_level_shift_and_not_noise() -> None:
    noise = RNG.normal(0, 1, 60)
    assert cp.binary_segmentation(noise) == []
    shifted = np.concatenate([RNG.normal(0, 1, 30), RNG.normal(3, 1, 30)])
    found = cp.binary_segmentation(shifted)
    assert found
    assert abs(found[0].index - 30) <= 2
    assert found[0].shift == pytest.approx(3, abs=0.8)


def test_cusum_false_alarm_rate_is_controlled() -> None:
    rng = np.random.default_rng(7)
    alarms = sum(bool(cp.binary_segmentation(rng.normal(0, 1, 40))) for _ in range(300))
    assert alarms / 300 < 0.05


def test_robust_spikes() -> None:
    x = RNG.normal(100, 5, 50)
    x[20] = 400
    assert 20 in cp.robust_spikes(x)


@settings(max_examples=50, deadline=None)
@given(hs.lists(hs.floats(-1e6, 1e6), min_size=2, max_size=200), hs.integers(1, 199))
def test_running_moments_merge_is_exact(values: list[float], split: int) -> None:
    arr = np.array(values)
    split = min(split, len(arr) - 1)
    a, b = RunningMoments(), RunningMoments()
    a.update(arr[:split])
    b.update(arr[split:])
    a.merge(b)
    assert a.n == len(arr)
    assert a.mean == pytest.approx(arr.mean(), rel=1e-9, abs=1e-6)
    assert a.variance == pytest.approx(arr.var(ddof=1), rel=1e-6, abs=1e-3)


def test_kmv_sketch() -> None:
    small = KMVSketch(k=256)
    small.update(range(100))
    assert small.exact
    assert small.estimate() == 100
    big = KMVSketch(k=1024)
    big.update(range(100_000))
    assert big.estimate() == pytest.approx(100_000, rel=0.1)  # ~3% standard error


def test_profile_chunks_matches_full_pass() -> None:
    frame = pd.DataFrame({"x": RNG.normal(0, 1, 3000), "c": RNG.choice(list("abcd"), 3000)})
    frame.loc[::10, "x"] = np.nan
    prof = profile_chunks(frame.iloc[i : i + 700] for i in range(0, 3000, 700))
    x = prof["columns"]["x"]
    assert prof["rows"] == 3000
    assert x["missing"] == 300
    assert x["mean"] == pytest.approx(frame["x"].mean())
    assert x["std"] == pytest.approx(frame["x"].std())
    assert prof["columns"]["c"]["distinct_estimate"] == 4
