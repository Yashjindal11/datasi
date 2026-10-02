"""Association measures between pairs of columns. None of these imply causation.

* Pearson r: linear association between numeric columns.
* Spearman rho: monotonic association (Pearson on ranks), robust to outliers.
* Cramér's V (bias-corrected, Bergsma 2013): association between categoricals, in [0, 1].
* Correlation ratio eta: how much of a numeric column's variance a categorical explains.
* Normalised mutual information: general (non-linear) dependence on discretised data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num = frame.apply(pd.to_numeric, errors="coerce").astype("float64")
    num = num.replace([np.inf, -np.inf], np.nan)
    return num.corr(method=method, min_periods=10)


def cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    table = pd.crosstab(x, y)
    n = float(table.to_numpy().sum())
    r, k = table.shape
    if n < 2 or r < 2 or k < 2:
        return None
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / n
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2c / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    v = pd.to_numeric(values, errors="coerce")
    ok = categories.notna() & v.notna()
    if ok.sum() < 3:
        return None
    df = pd.DataFrame({"c": categories[ok].astype(str), "v": v[ok].astype(float)})
    total = ((df["v"] - df["v"].mean()) ** 2).sum()
    if total == 0:
        return None
    g = df.groupby("c", observed=True)["v"].agg(["mean", "count"])
    between = (g["count"] * (g["mean"] - df["v"].mean()) ** 2).sum()
    return float(np.sqrt(between / total))


def discretize(values: pd.Series, bins: int = 10) -> pd.Series:
    """Quantile bins for numeric data; string labels for everything else."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        v = pd.to_numeric(values, errors="coerce")
        if v.nunique() > bins:
            return pd.qcut(v, bins, labels=False, duplicates="drop").astype("float64")
        return v.astype("float64")
    return values.astype("string")


def normalized_mutual_information(x: pd.Series, y: pd.Series) -> float | None:
    """MI(X;Y) / sqrt(H(X) H(Y)) on discretised values, rows with missing values dropped.

    Plug-in MI is biased upward for high-cardinality variables, which is why
    identifier-like columns are excluded before calling this.
    """
    from sklearn.metrics import normalized_mutual_info_score

    dx, dy = discretize(x), discretize(y)
    ok = dx.notna() & dy.notna()
    if ok.sum() < 10 or dx[ok].nunique() < 2 or dy[ok].nunique() < 2:
        return None
    return float(
        normalized_mutual_info_score(
            dx[ok].astype(str), dy[ok].astype(str), average_method="geometric"
        )
    )


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation of two binary indicators."""
    a = a.astype(float)
    b = b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
