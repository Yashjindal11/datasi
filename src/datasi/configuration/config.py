"""Investigation configuration: which detectors run and the thresholds they use.

Every threshold that influences severity lives here, so severity is always the result
of an explicit, user-adjustable rule. See docs/thresholds.md for the rationale of each
default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

KNOWN_DETECTORS: tuple[str, ...] = (
    "schema",
    "missingness",
    "duplicates",
    "constant",
    "identifier",
    "types",
    "numeric",
    "outliers",
    "categorical",
    "strings",
    "datetime",
    "temporal",
    "correlation",
    "redundancy",
    "target",
)

OutlierMethod = Literal["iqr", "zscore", "modified_zscore", "isolation_forest"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Thresholds(_Strict):
    # Missingness (fraction of rows)
    missing_warning: float = Field(0.10, ge=0, le=1)
    missing_high: float = Field(0.50, ge=0, le=1)
    missing_critical: float = Field(1.0, ge=0, le=1)
    missing_group_min_gap: float = Field(
        0.20, ge=0, le=1, description="Absolute gap between a group's missing rate and the rest."
    )
    missing_cooccurrence_phi: float = Field(0.5, ge=0, le=1)
    # Duplicates (fraction of rows)
    duplicate_warning: float = Field(0.001, ge=0, le=1)
    duplicate_high: float = Field(0.05, ge=0, le=1)
    # Constant / dominant values
    near_constant_ratio: float = Field(0.99, ge=0.5, le=1)
    dominant_category_ratio: float = Field(0.95, ge=0.5, le=1)
    # Identifiers and cardinality
    identifier_unique_ratio: float = Field(0.95, ge=0, le=1)
    high_cardinality_ratio: float = Field(0.5, ge=0, le=1)
    high_cardinality_min: int = Field(100, ge=2)
    rare_category_ratio: float = Field(0.005, ge=0, le=1)
    # Types
    type_parse_ratio: float = Field(
        0.95, ge=0.5, le=1, description="Share of values that must parse to infer a stored type."
    )
    # Numeric shape
    skew_warning: float = Field(2.0, ge=0)
    excess_kurtosis_warning: float = Field(10.0, ge=0)
    zero_inflation_ratio: float = Field(0.5, ge=0, le=1)
    value_spike_ratio: float = Field(
        0.10,
        ge=0,
        le=1,
        description="Share of rows on one value in an otherwise continuous column.",
    )
    # Outliers
    iqr_k: float = Field(1.5, gt=0)
    iqr_extreme_k: float = Field(3.0, gt=0)
    zscore: float = Field(3.0, gt=0)
    modified_zscore: float = Field(3.5, gt=0)
    isolation_contamination: float | Literal["auto"] = "auto"
    outlier_warning_ratio: float = Field(0.05, ge=0, le=1)
    # Correlation / association (absolute coefficient)
    correlation_warning: float = Field(0.90, ge=0, le=1)
    correlation_high: float = Field(0.98, ge=0, le=1)
    cramers_v_warning: float = Field(0.90, ge=0, le=1)
    # Distribution shift (effect sizes, not p-values)
    psi_warning: float = Field(0.10, ge=0)
    psi_high: float = Field(0.25, ge=0)
    ks_warning: float = Field(0.10, ge=0, le=1)
    ks_high: float = Field(0.25, ge=0, le=1)
    jsd_warning: float = Field(0.05, ge=0, le=1)
    jsd_high: float = Field(0.15, ge=0, le=1)
    # Temporal
    temporal_min_periods: int = Field(8, ge=4)
    temporal_shift_gap: float = Field(
        0.15, ge=0, le=1, description="Minimum absolute change in a rate across a change point."
    )
    volume_change_ratio: float = Field(
        0.5, ge=0, description="Relative change in rows/period across a change point."
    )
    gap_factor: float = Field(
        5.0, gt=1, description="An interval this many times the median interval is a gap."
    )
    # Target
    imbalance_warning: float = Field(0.10, ge=0, le=0.5)
    imbalance_high: float = Field(0.01, ge=0, le=0.5)
    target_association_warning: float = Field(
        0.95, ge=0, le=1, description="Single-feature predictive score that looks too good."
    )

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        pairs = [
            ("missing_warning", "missing_high"),
            ("missing_high", "missing_critical"),
            ("duplicate_warning", "duplicate_high"),
            ("iqr_k", "iqr_extreme_k"),
            ("correlation_warning", "correlation_high"),
            ("psi_warning", "psi_high"),
            ("ks_warning", "ks_high"),
            ("jsd_warning", "jsd_high"),
            ("imbalance_high", "imbalance_warning"),
        ]
        for low, high in pairs:
            if getattr(self, low) > getattr(self, high):
                raise ValueError(f"thresholds.{low} must be <= thresholds.{high}")
        return self


class ColumnRule(_Strict):
    """Domain knowledge for one column. Violations are reported as confirmed invalid values."""

    min: float | None = None
    max: float | None = None
    allowed: list[str] | None = None
    pattern: str | None = Field(default=None, description="Regex every non-null value must match.")
    nullable: bool = True
    unique: bool = False


class Privacy(_Strict):
    include_examples: bool = Field(
        False, description="Include sampled raw cell values in finding evidence."
    )
    max_examples: int = Field(5, ge=0, le=100)
    redact_labels: bool = Field(
        False, description="Replace category labels in reports with stable hashes."
    )


class Performance(_Strict):
    sample_rows: int | None = Field(
        200_000,
        ge=1000,
        description="Row cap for expensive pairwise/multivariate analyses. Counts stay exact.",
    )
    max_pairwise_columns: int = Field(60, ge=2)
    max_group_levels: int = Field(50, ge=2)
    random_state: int = 0


class Config(_Strict):
    detectors: dict[str, bool] = Field(default_factory=dict)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    outlier_methods: list[OutlierMethod] = Field(
        default_factory=lambda: ["iqr", "modified_zscore"]  # type: ignore[arg-type]
    )
    target: str | None = None
    time_column: str | None = None
    id_columns: list[str] = Field(default_factory=list)
    rules: dict[str, ColumnRule] = Field(default_factory=dict)
    privacy: Privacy = Field(default_factory=Privacy)
    performance: Performance = Field(default_factory=Performance)
    reference_time: str | None = Field(
        default=None,
        description="ISO timestamp treated as 'now' for future-date checks (default: run time).",
    )

    @model_validator(mode="after")
    def _known_detectors(self) -> Config:
        unknown = sorted(set(self.detectors) - set(KNOWN_DETECTORS))
        if unknown:
            raise ValueError(f"unknown detector(s) {unknown}; known: {', '.join(KNOWN_DETECTORS)}")
        return self

    def enabled(self, name: str) -> bool:
        return self.detectors.get(name, True)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> Config:
        return cls.model_validate(data or {})

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if data is not None and not isinstance(data, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        try:
            return cls.from_mapping(data)
        except ValidationError as exc:
            raise ConfigError(f"{path}: {exc}") from exc

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        data["detectors"] = {name: self.enabled(name) for name in KNOWN_DETECTORS}
        return yaml.safe_dump(data, sort_keys=False)


class ConfigError(ValueError):
    pass
