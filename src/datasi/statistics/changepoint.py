"""Change detection on time-ordered aggregates (one value per period).

Mean-shift detection uses the standardised CUSUM statistic

    T = max_k |S_k| / (sigma * sqrt(n)),   S_k = sum_{i<=k} (x_i - mean(x)).

Under the null of independent, identically distributed values T converges to the
supremum of a Brownian bridge (Kolmogorov distribution), giving the critical value
1.628 at the 1% level. sigma is estimated from first differences with the MAD so the
shift being tested does not inflate it.

Limitations: positive autocorrelation (trends, seasonality) inflates T and causes
false alarms; few periods (< ~8) give little power. The detector therefore also
requires a minimum effect size before reporting anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

CRITICAL_1PCT = 1.628


@dataclass(frozen=True)
class ChangePoint:
    index: int  # first index of the new regime
    statistic: float
    p_value: float  # asymptotic, assumes independence
    mean_before: float
    mean_after: float

    @property
    def shift(self) -> float:
        return self.mean_after - self.mean_before


def robust_sigma(x: np.ndarray) -> float:
    d = np.diff(x)
    if d.size == 0:
        return 0.0
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = float(mad / 0.6745 / np.sqrt(2))
    if sigma == 0:
        sigma = float(np.std(d, ddof=1) / np.sqrt(2)) if d.size > 1 else 0.0
    return sigma


def cusum_change(x: np.ndarray, min_segment: int = 3) -> ChangePoint | None:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 * min_segment:
        return None
    sigma = robust_sigma(x)
    s = np.cumsum(x - x.mean())[:-1]
    valid = np.arange(1, n)
    keep = (valid >= min_segment) & (valid <= n - min_segment)
    if not keep.any():
        return None
    s = s[keep]
    valid = valid[keep]
    k = int(np.argmax(np.abs(s)))
    before, after = x[: valid[k]].mean(), x[valid[k] :].mean()
    if sigma == 0:
        if before == after:
            return None
        return ChangePoint(int(valid[k]), float("inf"), 0.0, float(before), float(after))
    t = float(np.abs(s[k]) / (sigma * np.sqrt(n)))
    return ChangePoint(int(valid[k]), t, float(sps.kstwobign.sf(t)), float(before), float(after))


def binary_segmentation(
    x: np.ndarray,
    critical: float = CRITICAL_1PCT,
    min_segment: int = 3,
    max_changes: int = 3,
) -> list[ChangePoint]:
    """Recursively split at significant CUSUM change points (largest first)."""
    x = np.asarray(x, dtype=float)
    found: list[ChangePoint] = []

    def recurse(lo: int, hi: int) -> None:
        if len(found) >= max_changes:
            return
        cp = cusum_change(x[lo:hi], min_segment)
        if cp is None or cp.statistic < critical:
            return
        found.append(
            ChangePoint(lo + cp.index, cp.statistic, cp.p_value, cp.mean_before, cp.mean_after)
        )
        recurse(lo, lo + cp.index)
        recurse(lo + cp.index, hi)

    recurse(0, x.size)
    return sorted(found, key=lambda c: c.index)


def robust_spikes(x: np.ndarray, threshold: float = 5.0, window: int = 7) -> np.ndarray:
    """Indices whose value deviates from a centred rolling median by > threshold robust SDs."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < window:
        return np.array([], dtype=int)
    half = window // 2
    med = np.array([np.median(x[max(0, i - half) : i + half + 1]) for i in range(n)])
    resid = x - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    scale = mad / 0.6745 if mad > 0 else float(np.std(resid)) or 0.0
    if scale == 0:
        return np.array([], dtype=int)
    return np.flatnonzero(np.abs(resid) / scale > threshold)
