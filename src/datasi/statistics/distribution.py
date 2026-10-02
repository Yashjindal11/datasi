"""Two-sample distribution comparison, choosing metrics by data type.

Numeric columns: Kolmogorov-Smirnov statistic D (max CDF gap, scale-free, in [0, 1]),
Wasserstein-1 distance normalised by the reference IQR (so it reads as "shift in IQR
units"), and PSI on reference-quantile bins.
Categorical columns: Jensen-Shannon divergence (base 2, in [0, 1]) and PSI on the
category frequencies.

Decisions use effect sizes, not p-values: with large samples every trivial
difference is "significant", and with small samples real shifts are not. The KS
p-value is still reported as context.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

_EPS = 1e-4


def _finite(values: pd.Series) -> np.ndarray:
    arr: np.ndarray = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite: np.ndarray = arr[np.isfinite(arr)]
    return finite


def psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index: sum((a - e) * ln(a / e)), with empty bins floored."""
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref, cur = _finite(reference), _finite(current)
    if ref.size < bins or cur.size == 0:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return None
    # Open-ended outer bins; a mass point at the minimum keeps its own bin.
    edges = np.r_[-np.inf, edges[:-1], np.inf]
    inner = edges[1:-1]
    k = edges.size - 1
    e = np.bincount(np.digitize(ref, inner, right=True), minlength=k) / ref.size
    a = np.bincount(np.digitize(cur, inner, right=True), minlength=k) / cur.size
    return psi_from_proportions(e, a)


def category_proportions(reference: pd.Series, current: pd.Series) -> tuple[pd.Series, pd.Series]:
    r = reference.dropna().astype(str).value_counts(normalize=True)
    c = current.dropna().astype(str).value_counts(normalize=True)
    idx = r.index.union(c.index)
    return r.reindex(idx, fill_value=0.0), c.reindex(idx, fill_value=0.0)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence in bits (0 = identical, 1 = disjoint support)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_numeric(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    ref, cur = _finite(reference), _finite(current)
    out: dict[str, Any] = {"kind": "numeric", "n_reference": ref.size, "n_current": cur.size}
    if ref.size < 2 or cur.size < 2:
        return out
    ks = sps.ks_2samp(ref, cur)
    iqr = float(np.subtract(*np.quantile(ref, [0.75, 0.25])))
    scale = iqr if iqr > 0 else float(ref.std()) or 1.0
    out.update(
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        wasserstein=float(sps.wasserstein_distance(ref, cur)),
        wasserstein_iqr=float(sps.wasserstein_distance(ref, cur) / scale),
        psi=numeric_psi(reference, current),
        mean_reference=float(ref.mean()),
        mean_current=float(cur.mean()),
        median_reference=float(np.median(ref)),
        median_current=float(np.median(cur)),
        range_reference=[float(ref.min()), float(ref.max())],
        range_current=[float(cur.min()), float(cur.max())],
        out_of_range_share=float(((cur < ref.min()) | (cur > ref.max())).mean()),
    )
    return out


def compare_categorical(reference: pd.Series, current: pd.Series, top: int = 10) -> dict[str, Any]:
    p, q = category_proportions(reference, current)
    out: dict[str, Any] = {
        "kind": "categorical",
        "n_reference": int(reference.notna().sum()),
        "n_current": int(current.notna().sum()),
    }
    if p.empty:
        return out
    diff = (q - p).abs().sort_values(ascending=False)
    unseen = q[(p == 0) & (q > 0)]
    vanished = p[(q == 0) & (p > 0)]
    out.update(
        jsd=jensen_shannon(p.to_numpy(), q.to_numpy()),
        psi=psi_from_proportions(p.to_numpy(), q.to_numpy()),
        unseen_categories=int(unseen.size),
        unseen_share=float(unseen.sum()),
        vanished_categories=int(vanished.size),
        largest_changes=[
            {"label": str(k), "reference": float(p[k]), "current": float(q[k])}
            for k in diff.index[:top]
        ],
    )
    return out
