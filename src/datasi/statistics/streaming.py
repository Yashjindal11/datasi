"""Mergeable streaming statistics for data that does not fit in memory.

* :class:`RunningMoments` - count/mean/variance/min/max via Welford's update and
  Chan et al.'s parallel merge (numerically stable, exact).
* :class:`KMVSketch` - k-minimum-values distinct-count estimator; relative standard
  error is about 1/sqrt(k - 2), exact while fewer than k distinct values are seen.
* :func:`profile_chunks` - one pass over an iterator of DataFrames producing per-column
  missing counts, moments and approximate cardinality.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunningMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        other = RunningMoments(
            n=int(x.size),
            mean=float(x.mean()),
            m2=float(((x - x.mean()) ** 2).sum()),
            min=float(x.min()),
            max=float(x.max()),
        )
        self.merge(other)

    def merge(self, other: RunningMoments) -> None:
        if other.n == 0:
            return
        n = self.n + other.n
        delta = other.mean - self.mean
        self.mean += delta * other.n / n
        self.m2 += other.m2 + delta**2 * self.n * other.n / n
        self.n = n
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


def _hash01(value: str) -> float:
    h = hashlib.blake2b(value.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


@dataclass
class KMVSketch:
    k: int = 1024
    _heap: list[float] = field(default_factory=list)  # max-heap via negation
    _members: set[float] = field(default_factory=set)

    def update(self, values: Iterable[Any]) -> None:
        self.update_hashes(_hash01(str(v)) for v in values)

    def merge(self, other: KMVSketch) -> None:
        self.update_hashes(-x for x in other._heap)

    def update_hashes(self, hashes: Iterable[float]) -> None:
        for h in hashes:
            if h in self._members:
                continue
            if len(self._heap) < self.k:
                heapq.heappush(self._heap, -h)
                self._members.add(h)
            elif h < -self._heap[0]:
                removed = -heapq.heapreplace(self._heap, -h)
                self._members.discard(removed)
                self._members.add(h)

    @property
    def exact(self) -> bool:
        return len(self._heap) < self.k

    def estimate(self) -> float:
        if self.exact:
            return float(len(self._heap))
        kth = -self._heap[0]
        return (self.k - 1) / kth


@dataclass
class ColumnStream:
    rows: int = 0
    missing: int = 0
    moments: RunningMoments = field(default_factory=RunningMoments)
    distinct: KMVSketch = field(default_factory=KMVSketch)
    numeric: bool = True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rows": self.rows,
            "missing": self.missing,
            "missing_ratio": self.missing / self.rows if self.rows else 0.0,
            "distinct_estimate": round(self.distinct.estimate()),
            "distinct_exact": self.distinct.exact,
        }
        if self.numeric and self.moments.n:
            out.update(
                mean=self.moments.mean,
                std=self.moments.std,
                min=self.moments.min,
                max=self.moments.max,
            )
        return out


def profile_chunks(chunks: Iterable[pd.DataFrame], sketch_k: int = 1024) -> dict[str, Any]:
    """Single-pass profile over DataFrame chunks (e.g. ``pd.read_csv(..., chunksize=N)``)."""
    cols: dict[str, ColumnStream] = {}
    rows = 0
    for chunk in chunks:
        rows += len(chunk)
        for name in chunk.columns:
            key = str(name)
            s = chunk[name]
            cs = cols.setdefault(key, ColumnStream(distinct=KMVSketch(sketch_k)))
            cs.rows += len(s)
            cs.missing += int(s.isna().sum())
            present = s.dropna()
            cs.distinct.update(present.astype(str).unique())
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                cs.moments.update(present.to_numpy(dtype=float))
            else:
                cs.numeric = False
    return {"rows": rows, "columns": {k: v.to_dict() for k, v in cols.items()}}
