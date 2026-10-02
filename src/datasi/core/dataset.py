"""The Dataset abstraction: a read-only view of the user's data plus provenance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd


@dataclass(frozen=True)
class SamplingInfo:
    """Recorded whenever rows were dropped before analysis, so reports can disclose it."""

    method: str  # "head" | "random"
    rows_read: int
    rows_total: int | None  # None when the source size is unknown (streamed)


@dataclass(eq=False)
class Dataset:
    """Wraps a DataFrame without modifying it.

    Columns are exposed under string names; duplicate source names are disambiguated
    with a ``__N`` suffix and recorded in :attr:`duplicate_column_names`.
    """

    source_frame: pd.DataFrame
    name: str = "dataset"
    source: str | None = None
    sampling: SamplingInfo | None = None
    file_format: str | None = None
    _frame: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = [str(c) for c in self.source_frame.columns]
        seen: Counter[str] = Counter()
        unique: list[str] = []
        for n in names:
            seen[n] += 1
            unique.append(n if seen[n] == 1 else f"{n}__{seen[n]}")
        self.duplicate_column_names = sorted(n for n, c in seen.items() if c > 1)
        # Shallow copy: relabelled columns, shared data. Analyses never write in place.
        frame = self.source_frame.copy(deep=False)
        frame.columns = pd.Index(unique, dtype=object)
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(drop=True)
        self._frame = frame

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    @property
    def n_rows(self) -> int:
        return len(self._frame)

    @property
    def n_columns(self) -> int:
        return int(self._frame.shape[1])

    @property
    def columns(self) -> list[str]:
        return list(self._frame.columns)

    @cached_property
    def memory_bytes(self) -> int:
        return int(self._frame.memory_usage(deep=True).sum())

    def sample(self, n: int | None, random_state: int = 0) -> pd.DataFrame:
        """Deterministic row sample (or the full frame when n is None or larger)."""
        if n is None or n >= self.n_rows:
            return self._frame
        return self._frame.sample(n=n, random_state=random_state).sort_index()

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, rows={self.n_rows}, columns={self.n_columns})"
