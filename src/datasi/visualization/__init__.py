"""Optional matplotlib charts drawn from a :class:`~datasi.reports.Report`.

Install with ``pip install 'datasi[plot]'``. Every function takes a report (never raw
data) and returns a matplotlib ``Figure``; nothing is shown or saved implicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasi.findings import Severity

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

    from datasi.reports.model import Report

_SEV_COLORS = {"critical": "#b42318", "high": "#d4572a", "warning": "#b7791f", "info": "#5f6b7d"}


def _subplots(figsize: tuple[float, float]) -> tuple[Figure, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("plotting needs matplotlib: pip install 'datasi[plot]'") from exc
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_severity(report: Report) -> Figure:
    sev = report.summary.by_severity
    fig, ax = _subplots(figsize=(5, 2.5))
    names = [s.value for s in reversed(list(Severity))]
    ax.barh(names, [sev.get(n, 0) for n in names], color=[_SEV_COLORS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("findings")
    ax.set_title("Findings by severity")
    fig.tight_layout()
    return fig


def plot_missingness(report: Report, top: int = 30) -> Figure:
    cols = report.sections.get("missingness", {}).get("columns", {})
    items = sorted(
        ((k, v["ratio"]) for k, v in cols.items() if v["ratio"] > 0), key=lambda kv: -kv[1]
    )[:top]
    fig, ax = _subplots(figsize=(6, 0.3 * max(len(items), 3) + 1))
    if items:
        ax.barh([k for k, _ in items], [v for _, v in items], color="#2f5bea")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
    ax.set_xlabel("share missing")
    ax.set_title("Missing values per column")
    fig.tight_layout()
    return fig


def plot_distribution(report: Report, column: str) -> Figure:
    prof = report.profiles.get(column, {})
    fig, ax = _subplots(figsize=(6, 3))
    if "histogram" in prof and prof["histogram"]["counts"]:
        h = prof["histogram"]
        edges, counts = h["edges"], h["counts"]
        widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
        ax.bar(edges[:-1], counts, width=widths, align="edge", color="#2f5bea", edgecolor="white")
        ax.set_ylabel("rows")
    elif "categorical" in prof:
        top = prof["categorical"]["top"]
        ax.barh([t["label"] for t in top], [t["share"] for t in top], color="#2f5bea")
        ax.invert_yaxis()
        ax.set_xlabel("share of rows")
    else:
        raise ValueError(f"no distribution profile for column {column!r}")
    ax.set_title(column)
    fig.tight_layout()
    return fig


def plot_correlation(report: Report, method: str = "pearson") -> Figure:
    sec = report.sections.get("correlation", {})
    labels = sec.get("numeric_columns") or []
    if len(labels) < 2:
        raise ValueError("report has no correlation matrix (fewer than two numeric columns)")
    import numpy as np

    m = np.array([[np.nan if v is None else v for v in row] for row in sec[method]], dtype=float)
    fig, ax = _subplots(figsize=(0.4 * len(labels) + 2.5, 0.4 * len(labels) + 2))
    im = ax.imshow(m, vmin=-1, vmax=1, cmap="coolwarm_r")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(im, ax=ax, shrink=0.8, label=method)
    ax.set_title(f"{method.title()} correlation (not causation)")
    fig.tight_layout()
    return fig


def plot_temporal(report: Report) -> Figure:
    sec = report.sections.get("temporal")
    if not sec:
        raise ValueError("report has no temporal section (no usable time column)")
    pts = sec["volume"]
    fig, ax = _subplots(figsize=(8, 3))
    ax.plot([p["t"] for p in pts], [p["v"] for p in pts], color="#2f5bea")
    step = max(1, len(pts) // 10)
    ax.set_xticks(
        range(0, len(pts), step),
        [pts[i]["t"] for i in range(0, len(pts), step)],
        rotation=45,
        ha="right",
    )
    for f in report.findings:
        at = f.evidence.get("change_at") if f.code == "temporal.volume_shift" else None
        if at:
            idx = next((i for i, p in enumerate(pts) if str(at).startswith(p["t"])), None)
            if idx is not None:
                ax.axvline(idx, color="#d4572a", linestyle="--", label="change point")
    ax.set_ylabel(f"rows per {sec['period']}")
    ax.set_title(f"Volume over time ({sec['time_column']})")
    fig.tight_layout()
    return fig


def plot_drift(report: Report) -> Figure:
    sec = report.sections.get("drift")
    if not sec:
        raise ValueError("not a comparison report")
    rows = [r for r in sec["columns"] if r.get("kind") in ("numeric", "categorical")]
    names = [r["column"] for r in rows]
    vals = [r.get("ks_statistic") if r["kind"] == "numeric" else r.get("jsd") for r in rows]
    fig, ax = _subplots(figsize=(6, 0.3 * max(len(rows), 3) + 1))
    ax.barh(
        names,
        [v or 0 for v in vals],
        color=[_SEV_COLORS.get(r.get("severity", "info"), "#5f6b7d") for r in rows],
    )
    ax.invert_yaxis()
    ax.set_xlabel("KS D (numeric) / JSD (categorical)")
    ax.set_title("Distribution shift by column")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_correlation",
    "plot_distribution",
    "plot_drift",
    "plot_missingness",
    "plot_severity",
    "plot_temporal",
]
