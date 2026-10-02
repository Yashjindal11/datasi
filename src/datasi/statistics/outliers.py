"""Univariate and multivariate outlier scoring.

All methods return a boolean mask aligned to the input. A flagged value is a
*statistical outlier*: unusual relative to the rest of the column. It is not evidence
that the value is wrong; only domain rules (see ``ColumnRule``) can establish that.

| method            | assumes                         | robust to the outliers it seeks |
|-------------------|---------------------------------|---------------------------------|
| iqr               | nothing beyond ordinal scale    | yes (quartiles)                 |
| zscore            | roughly normal, light tails     | no (mean/std are pulled)        |
| modified_zscore   | roughly symmetric               | yes (median/MAD, Iglewicz-Hoaglin) |
| isolation_forest  | outliers are few and different  | yes; multivariate, randomised   |
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Method = Literal["iqr", "zscore", "modified_zscore", "isolation_forest"]


def _finite(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return arr, np.isfinite(arr)


def iqr_bounds(values: pd.Series, k: float = 1.5) -> tuple[float, float] | None:
    arr, ok = _finite(values)
    if ok.sum() < 4:
        return None
    q1, q3 = np.quantile(arr[ok], [0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def iqr_mask(values: pd.Series, k: float = 1.5) -> np.ndarray:
    """Tukey fences. With IQR = 0 (more than half the values identical) nothing is
    flagged, because every deviation would otherwise be an 'outlier'."""
    arr, ok = _finite(values)
    bounds = iqr_bounds(values, k)
    if bounds is None or bounds[0] == bounds[1]:
        return np.zeros(arr.size, dtype=bool)
    lo, hi = bounds
    return ok & ((arr < lo) | (arr > hi))


def zscore_mask(values: pd.Series, threshold: float = 3.0) -> np.ndarray:
    arr, ok = _finite(values)
    if ok.sum() < 3:
        return np.zeros(arr.size, dtype=bool)
    mean = arr[ok].mean()
    std = arr[ok].std(ddof=1)
    if std == 0:
        return np.zeros(arr.size, dtype=bool)
    z = np.zeros_like(arr)
    z[ok] = (arr[ok] - mean) / std
    return ok & (np.abs(z) > threshold)


def modified_zscore_mask(values: pd.Series, threshold: float = 3.5) -> np.ndarray:
    """Iglewicz & Hoaglin (1993): M = 0.6745 (x - median) / MAD, flag |M| > 3.5.

    Falls back to the mean absolute deviation (scale 0.7979) when MAD is zero.
    """
    arr, ok = _finite(values)
    if ok.sum() < 3:
        return np.zeros(arr.size, dtype=bool)
    x = arr[ok]
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 0:
        m = 0.6745 * (arr - med) / mad
    else:
        meanad = np.mean(np.abs(x - med))
        if meanad == 0:
            return np.zeros(arr.size, dtype=bool)
        m = (arr - med) / (1.253314 * meanad)
    result: np.ndarray = ok & (np.abs(np.nan_to_num(m)) > threshold)
    return result


def isolation_forest_mask(
    frame: pd.DataFrame,
    contamination: float | Literal["auto"] = "auto",
    random_state: int = 0,
    max_rows: int | None = 200_000,
) -> np.ndarray:
    """Multivariate outliers over the numeric columns of ``frame``.

    Rows with any missing value are never flagged (they are not scored). Columns are
    robust-scaled first so no single unit dominates. When the frame is larger than
    ``max_rows`` the forest is fitted on a deterministic sample and scores all rows.
    """
    from sklearn.ensemble import IsolationForest

    num = frame.apply(pd.to_numeric, errors="coerce").astype("float64")
    num = num.replace([np.inf, -np.inf], np.nan)
    complete = num.notna().all(axis=1).to_numpy()
    mask = np.zeros(len(frame), dtype=bool)
    if complete.sum() < 50 or num.shape[1] == 0:
        return mask
    x = num.to_numpy()[complete]
    med = np.median(x, axis=0)
    scale = np.subtract(*np.quantile(x, [0.75, 0.25], axis=0))
    scale[scale == 0] = 1.0
    x = (x - med) / scale
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=1)
    fit_x = x
    if max_rows is not None and x.shape[0] > max_rows:
        idx = np.random.default_rng(random_state).choice(x.shape[0], max_rows, replace=False)
        fit_x = x[idx]
    model.fit(fit_x)
    mask[complete] = model.predict(x) == -1
    return mask


def outlier_mask(
    values: pd.Series,
    method: Method,
    *,
    iqr_k: float = 1.5,
    zscore: float = 3.0,
    modified_zscore: float = 3.5,
    contamination: float | Literal["auto"] = "auto",
    random_state: int = 0,
) -> np.ndarray:
    if method == "iqr":
        return iqr_mask(values, iqr_k)
    if method == "zscore":
        return zscore_mask(values, zscore)
    if method == "modified_zscore":
        return modified_zscore_mask(values, modified_zscore)
    if method == "isolation_forest":
        return isolation_forest_mask(values.to_frame(), contamination, random_state)
    raise ValueError(f"unknown outlier method {method!r}")
