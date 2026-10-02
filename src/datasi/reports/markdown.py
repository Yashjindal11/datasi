"""Markdown rendering of a report (for PRs, wikis and terminals)."""

from __future__ import annotations

from typing import Any

from datasi.findings import Finding, Severity
from datasi.reports.model import SECTION_ORDER, Report

_TITLES = {
    "schema": "Schema & Structure",
    "missingness": "Missingness",
    "duplicates": "Duplicates",
    "types": "Data Types & Rules",
    "numeric": "Numeric Analysis",
    "categorical": "Categorical Analysis",
    "strings": "Text Hygiene",
    "temporal": "Temporal Analysis",
    "relationships": "Relationships",
    "distribution": "Distribution Analysis",
    "target": "Target Analysis",
}


def _esc(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _bytes(n: int) -> str:
    size = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _finding_md(f: Finding) -> list[str]:
    cols = f" — `{'`, `'.join(f.columns)}`" if f.columns else ""
    lines = [
        f"#### [{f.severity.value.upper()}] {f.title}{cols}",
        "",
        f"**Observation.** {f.observation}",
        "",
    ]
    if f.interpretation:
        conf = f" _(confidence {f.confidence:.2f})_" if f.confidence is not None else ""
        lines += [f"**Possible interpretation.** {f.interpretation}{conf}", ""]
    if f.suggestion:
        lines += [f"**Suggested investigation.** {f.suggestion}", ""]
    if f.rule:
        lines += [f"<sub>Rule: {f.rule} · detector `{f.detector}` · code `{f.code}`</sub>", ""]
    return lines


def render_markdown(report: Report) -> str:
    s = report.summary
    d = report.dataset
    title = (
        "DataSI Comparison Report" if report.kind == "comparison" else "DataSI Investigation Report"
    )
    out = [f"# {title}: {d.name}", ""]
    out += ["## Executive Summary", "", f"- **Rows x columns:** {d.rows:,} x {d.columns}"]
    if report.reference is not None:
        out.append(f"- **Reference:** {report.reference.name} ({report.reference.rows:,} rows)")
    out.append(
        "- **Findings:** "
        + ", ".join(f"{v} {k}" for k, v in s.by_severity.items() if v)
        + (f" ({s.findings} total)" if s.findings else "none")
    )
    out.append(f"- **Columns with warnings or worse:** {s.columns_affected}")
    if d.sampling:
        out.append(
            f"- **Note:** only {d.sampling['rows_read']:,} rows were read ({d.sampling['method']})."
        )
    if d.analysis_sample_rows:
        out.append(
            f"- **Note:** pairwise/multivariate analyses used a {d.analysis_sample_rows:,}-row sample; counts are exact."
        )
    out += ["", *[f"> {h}" for h in s.headline], ""] if s.headline else [""]

    out += ["## Dataset Overview", "", "| Property | Value |", "|---|---|"]
    out += [
        f"| Name | {_esc(d.name)} |",
        f"| Format | {d.format or '-'} |",
        f"| Rows | {d.rows:,} |",
        f"| Columns | {d.columns} |",
        f"| Memory | {_bytes(d.memory_bytes)} |",
        f"| Time column | {report.time_column or '-'} |",
        f"| Target | {report.target or '-'} |",
        f"| DataSI | {report.datasi_version} · {report.created_at} |",
        "",
    ]
    fields = report.schema_.get("fields", [])
    if fields:
        out += [
            "## Schema",
            "",
            "| Column | Type | dtype | Missing | Unique | Notes |",
            "|---|---|---|---|---|---|",
        ]
        for c in fields:
            notes = ", ".join(c.get("hints", []))
            out.append(
                f"| {_esc(c['name'])} | {c['type']} | {c['dtype']} | {c['missing_ratio']:.1%} | "
                f"{c['unique']:,} | {_esc(notes)} |"
            )
        out.append("")

    crit = [f for f in report.findings if f.severity.rank >= Severity.HIGH.rank]
    out += ["## Critical and High Findings", ""]
    if crit:
        for f in crit:
            out += _finding_md(f)
    else:
        out += ["_None._", ""]

    for cat in SECTION_ORDER:
        fs = [
            f for f in report.findings if f.category == cat and f.severity.rank < Severity.HIGH.rank
        ]
        if not fs:
            continue
        out += [f"## {_TITLES[cat.value]}", ""]
        for f in fs:
            out += _finding_md(f)

    recs = report.recommendations
    out += ["## Recommendations", ""]
    if recs:
        out += ["Suggestions based on the evidence above (most severe first):", ""]
        for i, r in enumerate(recs, 1):
            out.append(f"{i}. **{r['suggestion']}** — because: {r['because']} ({r['severity']}).")
    else:
        out.append("_No warning-level findings; no recommendations._")
    out += [
        "",
        "---",
        "",
        "_Generated locally by DataSI. Observations are measurements; interpretations are possibilities, not conclusions._",
        "",
    ]
    return "\n".join(out)
