"""The investigation report: findings plus the profile material needed to verify them."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datasi._version import __version__
from datasi.findings import Category, Finding, Severity, to_jsonable

SECTION_ORDER = [
    Category.SCHEMA,
    Category.MISSINGNESS,
    Category.DUPLICATES,
    Category.TYPES,
    Category.NUMERIC,
    Category.CATEGORICAL,
    Category.STRINGS,
    Category.TEMPORAL,
    Category.RELATIONSHIPS,
    Category.DISTRIBUTION,
    Category.TARGET,
]


class DatasetInfo(BaseModel):
    name: str
    source: str | None = None
    format: str | None = None
    rows: int
    columns: int
    memory_bytes: int
    sampling: dict[str, Any] | None = Field(
        default=None, description="Set when only part of the source was read."
    )
    analysis_sample_rows: int | None = Field(
        default=None, description="Row cap used for pairwise/multivariate analyses, if applied."
    )


class DetectorRun(BaseModel):
    name: str
    status: Literal["ok", "error", "skipped"]
    seconds: float = 0.0
    findings: int = 0
    error: str | None = None
    info: dict[str, str] = Field(default_factory=dict)


class Summary(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    rows: int
    columns: int
    findings: int
    by_severity: dict[str, int]
    by_category: dict[str, int]
    columns_affected: int
    headline: list[str]

    def __str__(self) -> str:
        sev = ", ".join(f"{k}: {v}" for k, v in self.by_severity.items() if v)
        lines = [
            f"DataSI report for {self.dataset} ({self.rows:,} rows x {self.columns} columns)",
            f"{self.findings} findings ({sev or 'none'}) affecting {self.columns_affected} column(s)",
        ]
        lines.extend(f"  - {h}" for h in self.headline)
        return "\n".join(lines)


class Report(BaseModel):
    kind: Literal["inspection", "comparison"] = "inspection"
    datasi_version: str = __version__
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    dataset: DatasetInfo
    reference: DatasetInfo | None = None
    time_column: str | None = None
    target: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    findings: list[Finding] = Field(default_factory=list)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sections: dict[str, Any] = Field(default_factory=dict)
    detectors: list[DetectorRun] = Field(default_factory=list)
    runtime_seconds: float = 0.0

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("profiles", "sections", mode="before")
    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        return to_jsonable(value)

    # ----- querying -------------------------------------------------------------
    def filter(
        self,
        *,
        severity: Severity | str | None = None,
        min_severity: Severity | str | None = None,
        category: Category | str | None = None,
        column: str | None = None,
        detector: str | None = None,
    ) -> list[Finding]:
        out = self.findings
        if severity is not None:
            out = [f for f in out if f.severity == Severity(severity)]
        if min_severity is not None:
            floor = Severity(min_severity).rank
            out = [f for f in out if f.severity.rank >= floor]
        if category is not None:
            out = [f for f in out if f.category == Category(category)]
        if column is not None:
            out = [f for f in out if column in f.columns]
        if detector is not None:
            out = [f for f in out if f.detector == detector]
        return out

    @property
    def summary(self) -> Summary:
        sev = Counter(f.severity.value for f in self.findings)
        cat = Counter(f.category.value for f in self.findings)
        cols = {c for f in self.findings if f.severity != Severity.INFO for c in f.columns}
        top = [f for f in self.findings if f.severity.rank >= Severity.WARNING.rank][:5]
        return Summary(
            dataset=self.dataset.name,
            rows=self.dataset.rows,
            columns=self.dataset.columns,
            findings=len(self.findings),
            by_severity={s.value: sev.get(s.value, 0) for s in reversed(list(Severity))},
            by_category={c.value: cat[c.value] for c in SECTION_ORDER if cat.get(c.value)},
            columns_affected=len(cols),
            headline=[f"[{f.severity.value}] {f.title}" for f in top],
        )

    @property
    def recommendations(self) -> list[dict[str, Any]]:
        """Suggestions drawn from warning-or-worse findings, most severe first.

        Each recommendation cites the finding that motivates it; none are generated
        without evidence.
        """
        seen: set[str] = set()
        out = []
        for f in self.findings:
            if f.severity.rank < Severity.WARNING.rank or not f.suggestion:
                continue
            key = f"{f.code}|{f.suggestion}"
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "severity": f.severity.value,
                    "suggestion": f.suggestion,
                    "because": f.title,
                    "finding_id": f.id,
                    "columns": list(f.columns),
                }
            )
        return out

    # ----- serialisation ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", by_alias=True)
        data["findings"] = [f.to_dict() for f in self.findings]
        data["summary"] = self.summary.model_dump()
        data["recommendations"] = self.recommendations
        return data

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Report:
        data = json.loads(text)
        data.pop("summary", None)
        data.pop("recommendations", None)
        for f in data.get("findings", []):
            f.pop("id", None)
        return cls.model_validate(data)

    def to_markdown(self) -> str:
        from datasi.reports.markdown import render_markdown

        return render_markdown(self)

    def to_html(self) -> str:
        from datasi.reports.html import render_html

        return render_html(self)

    def save(self, path: str | Path, format: str | None = None) -> Path:
        """Write the report; format is inferred from the extension (.json/.md/.html)."""
        p = Path(path)
        fmt = (format or p.suffix.lstrip(".") or "html").lower()
        renderers = {
            "json": self.to_json,
            "md": self.to_markdown,
            "markdown": self.to_markdown,
            "html": self.to_html,
            "htm": self.to_html,
        }
        if fmt not in renderers:
            raise ValueError(f"unsupported report format {fmt!r}; use json, md or html")
        p.write_text(renderers[fmt](), encoding="utf-8")
        return p

    def __str__(self) -> str:
        return str(self.summary)

    def _repr_html_(self) -> str:  # Jupyter
        return self.to_html()
