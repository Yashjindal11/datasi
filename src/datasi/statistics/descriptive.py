"""Descriptive statistics and information measures for single columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps


def numeric_summary(values: pd.Series) -> dict[str, Any]:
    """Moments, quantiles and shape statistics of a numeric column (NaN/inf excluded).

    Skewness and kurtosis use the bias-corrected sample estimators (Fisher's
    definition: a normal distribution has excess kurtosis 0). They are undefined for
    fewer than 3 / 4 values or zero variance and are then reported as None.
    """
    s = pd.to_numeric(values, errors="coerce").astype("float64")
    arr = s.to_numpy()
    n_inf = int(np.isinf(arr).sum())
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: dict[str, Any] = {"count": n, "infinite": n_inf}
    if n == 0:
        return out
    q = np.quantile(arr, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    out.update(
        mean=float(arr.mean()),
        std=std,
        variance=std**2,
        min=float(q[0]),
        p01=float(q[1]),
        p05=float(q[2]),
        q1=float(q[3]),
        median=float(q[4]),
        q3=float(q[5]),
        p95=float(q[6]),
        p99=float(q[7]),
        max=float(q[8]),
        iqr=float(q[5] - q[3]),
        zeros=int((arr == 0).sum()),
        negatives=int((arr < 0).sum()),
        distinct=int(np.unique(arr).size),
        integer_valued=bool(np.all(np.mod(arr, 1) == 0)),
    )
    if n >= 3 and std > 0:
        out["skewness"] = float(sps.skew(arr, bias=False))
    else:
        out["skewness"] = None
    if n >= 4 and std > 0:
        out["excess_kurtosis"] = float(sps.kurtosis(arr, fisher=True, bias=False))
    else:
        out["excess_kurtosis"] = None
    return out


def histogram(values: pd.Series, bins: int = 30) -> dict[str, list[float]]:
    """Histogram on the 0.5-99.5% range so a single extreme value does not flatten it."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"edges": [], "counts": []}
    lo, hi = np.quantile(arr, [0.005, 0.995])
    if lo == hi:
        lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return {"edges": [float(lo), float(hi)], "counts": [float(arr.size)]}
    counts, edges = np.histogram(np.clip(arr, lo, hi), bins=bins, range=(lo, hi))
    return {"edges": edges.tolist(), "counts": counts.astype(float).tolist()}


def entropy_bits(counts: pd.Series | np.ndarray) -> float:
    """Shannon entropy in bits of a frequency vector."""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def normalized_entropy(counts: pd.Series | np.ndarray) -> float:
    """Entropy divided by its maximum log2(k); 1 = uniform, 0 = one value."""
    c = np.asarray(counts, dtype=float)
    k = int((c > 0).sum())
    return entropy_bits(c) / np.log2(k) if k > 1 else 0.0
