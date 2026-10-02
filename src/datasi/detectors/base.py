"""Detector protocol, shared investigation context and registry."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from datasi.configuration import Config, Thresholds
from datasi.core.dataset import Dataset
from datasi.findings import Category, Finding, Severity
from datasi.schema import ColumnType, Schema, StorageHint
from datasi.statistics import strings as st


@dataclass
class Context:
    """Everything a detector may read, plus caches so work is shared between detectors.

    Detectors must treat ``frame`` as read-only. They may add report material through
    :meth:`profile` (per column) and :meth:`section` (dataset level).
    """

    dataset: Dataset
    schema: Schema
    config: Config
    now: pd.Timestamp
    time_column: str | None = None
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    sections: dict[str, Any] = field(default_factory=dict)
    _cache: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)

    @property
    def frame(self) -> pd.DataFrame:
        return self.dataset.frame

    @property
    def thresholds(self) -> Thresholds:
        return self.config.thresholds

    @property
    def n_rows(self) -> int:
        return self.dataset.n_rows

    def cached(self, kind: str, key: str, fn: Callable[[], Any]) -> Any:
        k = (kind, key)
        if k not in self._cache:
            self._cache[k] = fn()
        return self._cache[k]

    def columns_of(self, *types: ColumnType) -> list[str]:
        return self.schema.of_type(*types)

    def counts(self, column: str) -> pd.Series:
        """value_counts of non-null values as strings (cached)."""
        result: pd.Series = self.cached(
            "counts", column, lambda: st.string_counts(self.frame[column])
        )
        return result

    def numeric(self, column: str) -> pd.Series:
        """Column as float64, parsing numeric-looking strings when the schema says so."""

        def build() -> pd.Series:
            s = self.frame[column]
            if pd.api.types.is_bool_dtype(s):
                return s.astype("float64")
            if pd.api.types.is_numeric_dtype(s):
                return s.astype("float64")
            return st.to_number(s)

        result: pd.Series = self.cached("numeric", column, build)
        return result

    def datetimes(self, column: str) -> pd.Series:
        result: pd.Series = self.cached(
            "datetime", column, lambda: st.parse_datetimes(self.frame[column])
        )
        return result

    def sample(self) -> pd.DataFrame:
        perf = self.config.performance
        result: pd.DataFrame = self.cached(
            "sample", "", lambda: self.dataset.sample(perf.sample_rows, perf.random_state)
        )
        return result

    @property
    def sampled(self) -> bool:
        n = self.config.performance.sample_rows
        return n is not None and n < self.n_rows

    def profile(self, column: str) -> dict[str, Any]:
        return self.profiles.setdefault(column, {})

    def section(self, name: str) -> dict[str, Any]:
        result: dict[str, Any] = self.sections.setdefault(name, {})
        return result

    def label(self, value: Any, limit: int = 60) -> str:
        """A category label as it may appear in a report (redacted when configured)."""
        text = str(value)
        if self.config.privacy.redact_labels:
            digest = hashlib.sha256(text.encode()).hexdigest()[:8]
            return f"<label:{digest}>"
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def examples(self, values: Iterable[Any]) -> list[str] | None:
        """Raw example values, only when the user opted in via privacy.include_examples."""
        p = self.config.privacy
        if not p.include_examples or p.max_examples == 0:
            return None
        out = []
        for v in values:
            out.append(self.label(v, 40))
            if len(out) >= p.max_examples:
                break
        return out

    def row_examples(self, mask: np.ndarray | pd.Series, limit: int = 10) -> list[int]:
        """Row positions (not values) of the first matching rows, safe to include in reports."""
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))[:limit]
        return [int(i) for i in idx]

    def is_numeric_like(self, column: str) -> bool:
        c = self.schema[column]
        return c.column_type == ColumnType.NUMERIC or (
            c.column_type == ColumnType.BOOLEAN and StorageHint.BOOLEAN_AS_NUMBER in c.hints
        )


@runtime_checkable
class Detector(Protocol):
    @property
    def name(self) -> str: ...

    def analyze(self, ctx: Context) -> list[Finding]: ...


class BaseDetector:
    """Convenience base: metadata used in reports and a finding factory.

    Subclasses document what they measure, their assumptions and limitations; this
    text is shown in the HTML report next to every finding they produce.
    """

    name: ClassVar[str]
    category: ClassVar[Category]
    description: ClassVar[str] = ""
    assumptions: ClassVar[str] = ""
    limitations: ClassVar[str] = ""

    def analyze(self, ctx: Context) -> list[Finding]:  # pragma: no cover - abstract
        raise NotImplementedError

    def finding(
        self,
        code: str,
        severity: Severity,
        title: str,
        observation: str,
        *,
        columns: Iterable[str] = (),
        evidence: dict[str, Any] | None = None,
        interpretation: str | None = None,
        suggestion: str | None = None,
        confidence: float | None = None,
        rule: str | None = None,
        category: Category | None = None,
    ) -> Finding:
        return Finding(
            detector=self.name,
            code=code,
            category=category or self.category,
            severity=severity,
            title=title,
            observation=observation,
            columns=tuple(columns),
            evidence=evidence or {},
            interpretation=interpretation,
            suggestion=suggestion,
            confidence=None if confidence is None else float(np.clip(confidence, 0.0, 1.0)),
            rule=rule,
        )

    @classmethod
    def info(cls) -> dict[str, str]:
        return {
            "name": cls.name,
            "category": cls.category.value,
            "description": cls.description,
            "assumptions": cls.assumptions,
            "limitations": cls.limitations,
        }


def tiered(value: float, tiers: list[tuple[float, Severity]], default: Severity) -> Severity:
    """Highest severity whose threshold ``value`` reaches (tiers sorted high to low)."""
    for threshold, sev in tiers:
        if value >= threshold:
            return sev
    return default


def pct(x: float) -> str:
    if x == 0:
        return "0%"
    if x < 0.001:
        return "<0.1%"
    return f"{x:.1%}"


_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_detector(cls: type[BaseDetector]) -> type[BaseDetector]:
    _REGISTRY[cls.name] = cls
    return cls


def registry() -> dict[str, type[BaseDetector]]:
    from datasi import detectors  # noqa: F401  (populates the registry)

    return dict(_REGISTRY)
