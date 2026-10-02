"""Structured findings: the unit of output for every detector.

A finding keeps *observation* (what was measured, stated as fact) separate from
*interpretation* (what it might mean, stated as a possibility). Severity comes from
explicit, documented rules; confidence expresses how much a heuristic should be trusted.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    @classmethod
    def at_least(cls, level: Severity) -> list[Severity]:
        return [s for s in cls if s.rank >= level.rank]


_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


class Category(StrEnum):
    """Report section a finding belongs to."""

    SCHEMA = "schema"
    MISSINGNESS = "missingness"
    DUPLICATES = "duplicates"
    TYPES = "types"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    STRINGS = "strings"
    TEMPORAL = "temporal"
    RELATIONSHIPS = "relationships"
    DISTRIBUTION = "distribution"
    TARGET = "target"


class Finding(BaseModel):
    """One detected issue, with the evidence needed to verify it."""

    model_config = ConfigDict(frozen=True)

    detector: str
    code: str = Field(description="Stable machine-readable identifier, e.g. 'missing.high'.")
    category: Category
    severity: Severity
    title: str
    observation: str = Field(description="What was measured. Factual, no speculation.")
    columns: tuple[str, ...] = ()
    evidence: dict[str, Any] = Field(default_factory=dict)
    interpretation: str | None = Field(
        default=None, description="Possible explanations. Always phrased as a possibility."
    )
    suggestion: str | None = Field(default=None, description="What the user could investigate.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Heuristic trust in the interpretation; None when the finding is a pure measurement.",
    )
    rule: str | None = Field(default=None, description="The explicit rule that set the severity.")

    @field_validator("evidence", mode="before")
    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        return to_jsonable(value)

    @property
    def column(self) -> str | None:
        return self.columns[0] if self.columns else None

    @property
    def id(self) -> str:
        raw = "|".join([self.detector, self.code, *self.columns])
        return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["id"] = self.id
        return data

    def explain(self) -> str:
        """Plain-text answer to: what happened, why, evidence, how serious, what next."""
        lines = [
            f"[{self.severity.value.upper()}] {self.title}",
            f"Observation: {self.observation}",
        ]
        if self.rule:
            lines.append(f"Why flagged: {self.rule}")
        if self.evidence:
            shown = ", ".join(f"{k}={_short(v)}" for k, v in list(self.evidence.items())[:8])
            lines.append(f"Evidence: {shown}")
        if self.interpretation:
            lines.append(f"Possible interpretation: {self.interpretation}")
        if self.confidence is not None:
            lines.append(f"Confidence: {self.confidence:.2f}")
        if self.suggestion:
            lines.append(f"Suggested investigation: {self.suggestion}")
        return "\n".join(lines)


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers into JSON-safe Python values."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if hasattr(value, "isoformat"):  # pandas Timestamp / Timedelta-like
        return str(value.isoformat())
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    return str(value)
