"""Shared helpers for time-aware detectors: period binning and change summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from datasi.detectors.base import Context
from datasi.statistics.changepoint import CRITICAL_1PCT, ChangePoint, binary_segmentation

# (pandas period alias, nominal length, human label)
_FREQS = [
    ("h", pd.Timedelta(hours=1), "hour"),
    ("D", pd.Timedelta(days=1), "day"),
    ("W", pd.Timedelta(days=7), "week"),
    ("M", pd.Timedelta(days=30.44), "month"),
    ("Q", pd.Timedelta(days=91.31), "quarter"),
    ("Y", pd.Timedelta(days=365.25), "year"),
]

MIN_ROWS_PER_PERIOD = 5


@dataclass(frozen=True)
class TimeBins:
    column: str
    freq: str
    unit: str
    labels: pd.Series  # period start per row (NaT where the time is missing)
    index: pd.DatetimeIndex  # every period in range, including empty ones

    def counts(self) -> pd.Series:
        return self.labels.value_counts().reindex(self.index, fill_value=0).astype(int)

    def rate(self, indicator: pd.Series) -> pd.Series:
        """Mean of a boolean indicator per period; NaN for periods with too few rows."""
        df = pd.DataFrame({"p": self.labels, "v": indicator.astype(float)}).dropna(subset=["p"])
        g = df.groupby("p")["v"].agg(["mean", "count"]).reindex(self.index)
        return g["mean"].where(g["count"] >= MIN_ROWS_PER_PERIOD)


def choose_freq(
    times: pd.Series, max_periods: int = 100, min_periods: int = 8
) -> tuple[str, str] | None:
    t = times.dropna()
    if t.empty:
        return None
    span = t.max() - t.min()
    for alias, length, unit in _FREQS:
        n = span / length + 1
        if n <= max_periods:
            return (alias, unit) if n >= min_periods else None
    return ("Y", "year")


def time_bins(ctx: Context) -> TimeBins | None:
    col = ctx.time_column
    if col is None:
        return None

    def build() -> TimeBins | None:
        t = ctx.datetimes(col)
        chosen = choose_freq(t, min_periods=ctx.thresholds.temporal_min_periods)
        if chosen is None:
            return None
        freq, unit = chosen
        labels = t.dt.to_period(freq).dt.start_time
        valid = t.dropna()
        index = pd.period_range(valid.min(), valid.max(), freq=freq).to_timestamp()
        return TimeBins(col, freq, unit, labels, pd.DatetimeIndex(index))

    result: TimeBins | None = ctx.cached("timebins", col, build)
    return result


def strongest_change(series: pd.Series, critical: float = CRITICAL_1PCT) -> ChangePoint | None:
    """Largest-shift significant change point of a per-period series (NaNs dropped)."""
    s = series.dropna()
    if s.size < 6:
        return None
    found = binary_segmentation(s.to_numpy(), critical=critical)
    if not found:
        return None
    return max(found, key=lambda c: abs(c.shift))


def change_date(series: pd.Series, cp: ChangePoint) -> pd.Timestamp:
    ts: pd.Timestamp = series.dropna().index[cp.index]
    return ts


def change_confidence(cp: ChangePoint, n_periods: int) -> float:
    """Heuristic: the CUSUM p-value assumes independent periods, which real data rarely
    satisfies, so confidence is capped well below 1 and grows with evidence strength."""
    conf = 0.5
    if cp.statistic >= 2.5:
        conf += 0.2
    if n_periods >= 20:
        conf += 0.1
    return float(np.clip(conf, 0.0, 0.85))


def fmt_time(ts: pd.Timestamp, unit: str) -> str:
    fmt = {"hour": "%Y-%m-%d %H:00", "day": "%Y-%m-%d", "week": "%Y-%m-%d"}.get(
        unit, "%Y-%m" if unit in {"month", "quarter"} else "%Y"
    )
    return str(ts.strftime(fmt))


def points(s: pd.Series, unit: str) -> list[dict[str, object]]:
    """A per-period series as JSON-friendly points for charts."""
    return [{"t": fmt_time(k, unit), "v": None if pd.isna(v) else float(v)} for k, v in s.items()]
