"""The investigation pipeline: load -> schema -> detectors (in order) -> report."""

from __future__ import annotations

import time
import traceback
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from datasi.configuration import Config
from datasi.core.dataset import Dataset
from datasi.detectors.base import BaseDetector, Context, Detector, registry
from datasi.findings import Category, Finding, Severity
from datasi.loaders import load
from datasi.reports.model import SECTION_ORDER, DatasetInfo, DetectorRun, Report
from datasi.schema import ColumnType, Schema, infer_schema

_TIME_NAME_HINTS = (
    "date",
    "time",
    "timestamp",
    "created",
    "updated",
    "_at",
    "period",
    "day",
    "month",
)
_TIME_NAME_AVOID = ("birth", "dob", "expir", "due", "valid")


def build_config(config: Config | dict[str, Any] | str | None, **overrides: Any) -> Config:
    if isinstance(config, Config):
        base = config
    elif isinstance(config, str):
        base = Config.from_file(config)
    else:
        base = Config.from_mapping(config)
    if not any(v is not None for v in overrides.values()):
        return base
    data = base.model_dump()
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return Config.model_validate(data)


def choose_time_column(schema: Schema, config: Config) -> str | None:
    """Configured time column, else the best-populated datetime column with a time-like name."""
    if config.time_column:
        return config.time_column if config.time_column in schema.columns else None
    candidates = []
    for name in schema.of_type(ColumnType.DATETIME):
        c = schema[name]
        if c.missing_ratio > 0.5 or c.n_unique < 10:
            continue
        lname = name.lower()
        if any(a in lname for a in _TIME_NAME_AVOID):
            continue
        named = any(h in lname for h in _TIME_NAME_HINTS)
        candidates.append((named, 1 - c.missing_ratio, c.n_unique, -c.position, name))
    if not candidates:
        return None
    return max(candidates)[-1]


def _resolve_detectors(
    config: Config,
    only: Sequence[str] | None,
    extra: Iterable[Detector] | None,
) -> list[Detector]:
    reg = registry()
    if only is not None:
        unknown = [d for d in only if d not in reg]
        if unknown:
            raise ValueError(f"unknown detector(s) {unknown}; known: {', '.join(reg)}")
        names = [n for n in reg if n in set(only)]
    else:
        names = [n for n in reg if config.enabled(n)]
    dets: list[Detector] = [reg[n]() for n in names]
    dets.extend(extra or [])
    return dets


def _sort_key(f: Finding) -> tuple[int, int, str]:
    return (-f.severity.rank, SECTION_ORDER.index(f.category), f.columns[0] if f.columns else "")


def dataset_info(ds: Dataset, config: Config) -> DatasetInfo:
    cap = config.performance.sample_rows
    return DatasetInfo(
        name=ds.name,
        source=ds.source,
        format=ds.file_format,
        rows=ds.n_rows,
        columns=ds.n_columns,
        memory_bytes=ds.memory_bytes,
        sampling=None if ds.sampling is None else ds.sampling.__dict__,
        analysis_sample_rows=cap if cap is not None and cap < ds.n_rows else None,
    )


def investigate(
    data: Any,
    *,
    config: Config | dict[str, Any] | str | None = None,
    detectors: Sequence[str] | None = None,
    extra_detectors: Iterable[Detector] | None = None,
    target: str | None = None,
    time_column: str | None = None,
    name: str | None = None,
    nrows: int | None = None,
    **load_kwargs: Any,
) -> Report:
    """Investigate a dataset and return a :class:`Report`.

    ``data`` may be a pandas/Polars DataFrame, a PyArrow table, a path to a CSV,
    Parquet or JSON file, or a :class:`Dataset`. The input is never modified.

    ``detectors`` restricts the run to the named detectors; otherwise every detector
    enabled in ``config`` runs. ``extra_detectors`` adds custom detector instances.
    """
    started = time.perf_counter()
    cfg = build_config(config, target=target, time_column=time_column)
    ds = load(data, name=name, nrows=nrows, **load_kwargs)
    schema = infer_schema(
        ds.frame,
        parse_ratio=cfg.thresholds.type_parse_ratio,
        unique_threshold=cfg.thresholds.identifier_unique_ratio,
        id_columns=cfg.id_columns,
    )
    now = (
        pd.Timestamp(cfg.reference_time).tz_localize(None)
        if cfg.reference_time
        else pd.Timestamp.now(tz="UTC").tz_localize(None)
    )
    ctx = Context(
        dataset=ds,
        schema=schema,
        config=cfg,
        now=now,
        time_column=choose_time_column(schema, cfg),
    )
    findings: list[Finding] = []
    runs: list[DetectorRun] = []
    for det in _resolve_detectors(cfg, detectors, extra_detectors):
        t0 = time.perf_counter()
        info = det.info() if isinstance(det, BaseDetector) else {"name": det.name}
        try:
            produced = det.analyze(ctx)
        except Exception as exc:
            runs.append(
                DetectorRun(
                    name=det.name,
                    status="error",
                    seconds=time.perf_counter() - t0,
                    error="".join(traceback.format_exception_only(exc)).strip(),
                    info=info,
                )
            )
            findings.append(
                Finding(
                    detector=det.name,
                    code="datasi.detector_error",
                    category=Category.SCHEMA,
                    severity=Severity.INFO,
                    title=f"Detector '{det.name}' failed",
                    observation=f"The detector raised {type(exc).__name__}: {exc}",
                    suggestion="Please report this with the dataset schema (not the data).",
                )
            )
            continue
        findings.extend(produced)
        runs.append(
            DetectorRun(
                name=det.name,
                status="ok",
                seconds=round(time.perf_counter() - t0, 4),
                findings=len(produced),
                info=info,
            )
        )
    findings.sort(key=_sort_key)
    for col in schema.columns.values():
        ctx.profile(col.name)["schema"] = col.to_dict()
    ordered_profiles = {c: ctx.profiles[c] for c in schema.columns}
    return Report(
        dataset=dataset_info(ds, cfg),
        time_column=ctx.time_column,
        target=cfg.target,
        config=cfg.model_dump(mode="json"),
        schema=schema.to_dict(),
        findings=findings,
        profiles=ordered_profiles,
        sections=ctx.sections,
        detectors=runs,
        runtime_seconds=round(time.perf_counter() - started, 4),
    )


def inspect(data: Any, **kwargs: Any) -> Report:
    """Alias of :func:`investigate`: ``report = inspect(df)``."""
    return investigate(data, **kwargs)


def inspect_schema(data: Any, config: Config | dict[str, Any] | None = None, **kw: Any) -> Schema:
    cfg = build_config(config)
    ds = load(data, **kw)
    return infer_schema(
        ds.frame,
        parse_ratio=cfg.thresholds.type_parse_ratio,
        unique_threshold=cfg.thresholds.identifier_unique_ratio,
        id_columns=cfg.id_columns,
    )
