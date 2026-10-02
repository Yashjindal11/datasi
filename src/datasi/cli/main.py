"""Command-line interface: ``datasi inspect data.csv --output report.html``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from datasi._version import __version__
from datasi.configuration import Config
from datasi.findings import Severity
from datasi.reports.model import Report

_COLORS = {"critical": "\033[1;31m", "high": "\033[31m", "warning": "\033[33m", "info": "\033[2m"}
_RESET = "\033[0m"


def _color(text: str, sev: str, enabled: bool) -> str:
    return f"{_COLORS[sev]}{text}{_RESET}" if enabled else text


def _print_report(report: Report, *, min_severity: str, color: bool, limit: int) -> None:
    print(report.summary)
    shown = report.filter(min_severity=min_severity)
    if not shown:
        return
    print()
    for f in shown[:limit]:
        tag = _color(f"{f.severity.value.upper():8}", f.severity.value, color)
        cols = f" [{', '.join(f.columns)}]" if f.columns else ""
        print(f"{tag} {f.title}{cols}")
        print(f"         {f.observation}")
    if len(shown) > limit:
        print(f"... {len(shown) - limit} more (use --limit or write a report with --output)")


def _config(args: argparse.Namespace) -> Config:
    cfg = Config.from_file(args.config) if getattr(args, "config", None) else Config()
    data = cfg.model_dump()
    if getattr(args, "disable", None):
        for name in args.disable.split(","):
            data["detectors"][name.strip()] = False
    if getattr(args, "outlier_methods", None):
        data["outlier_methods"] = [m.strip() for m in args.outlier_methods.split(",")]
    if getattr(args, "include_examples", False):
        data["privacy"]["include_examples"] = True
    if getattr(args, "reference_time", None):
        data["reference_time"] = args.reference_time
    return Config.model_validate(data)


def _write_outputs(report: Report, outputs: Sequence[str] | None) -> None:
    for out in outputs or []:
        path = report.save(out)
        print(f"wrote {path}", file=sys.stderr)


def _exit_code(report: Report, fail_on: str | None) -> int:
    if fail_on and report.filter(min_severity=fail_on):
        return 1
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from datasi.core.pipeline import investigate

    report = investigate(
        args.path,
        config=_config(args),
        detectors=args.detectors.split(",") if args.detectors else None,
        target=args.target,
        time_column=args.time_column,
        nrows=args.nrows,
    )
    if args.json:
        print(report.to_json())
    else:
        _print_report(
            report, min_severity=args.min_severity, color=_use_color(args), limit=args.limit
        )
    _write_outputs(report, args.output)
    return _exit_code(report, args.fail_on)


def cmd_report(args: argparse.Namespace) -> int:
    from datasi.core.pipeline import investigate

    report = investigate(
        args.path,
        config=_config(args),
        detectors=args.detectors.split(",") if args.detectors else None,
        target=args.target,
        time_column=args.time_column,
        nrows=args.nrows,
    )
    outputs = args.output or [f"datasi-report-{Path(args.path).stem}.html"]
    _write_outputs(report, outputs)
    print(report.summary, file=sys.stderr)
    return _exit_code(report, args.fail_on)


def cmd_compare(args: argparse.Namespace) -> int:
    from datasi.core.compare import inspect_train_test

    report = inspect_train_test(
        args.reference, args.current, config=_config(args), target=args.target
    )
    if args.json:
        print(report.to_json())
    else:
        _print_report(
            report, min_severity=args.min_severity, color=_use_color(args), limit=args.limit
        )
    _write_outputs(report, args.output)
    return _exit_code(report, args.fail_on)


def cmd_schema(args: argparse.Namespace) -> int:
    if args.chunksize:
        import pandas as pd

        from datasi.statistics.streaming import profile_chunks

        prof = profile_chunks(pd.read_csv(args.path, chunksize=args.chunksize, low_memory=False))
        print(json.dumps(prof, indent=2, default=str))
        return 0
    from datasi.core.pipeline import inspect_schema

    schema = inspect_schema(args.path, config=_config(args), nrows=args.nrows)
    if args.json:
        print(json.dumps(schema.to_dict(), indent=2))
        return 0
    print(
        f"{schema.n_rows:,} rows x {schema.n_columns} columns, {schema.memory_bytes / 1e6:.1f} MB in memory"
    )
    width = max([len(c) for c in schema.columns] + [6])
    print(f"{'column':{width}}  {'type':11}  {'dtype':14}  {'missing':>8}  {'unique':>9}  hints")
    for c in schema.columns.values():
        print(
            f"{c.name:{width}}  {c.column_type.value:11}  {c.dtype:14}  {c.missing_ratio:8.1%}  "
            f"{c.n_unique:9,}  {', '.join(h.value for h in c.hints)}"
        )
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    if args.config_command == "validate":
        cfg = Config.from_file(args.file)
        disabled = [k for k, v in cfg.detectors.items() if not v]
        print(f"{args.file}: valid" + (f" (disabled: {', '.join(disabled)})" if disabled else ""))
        return 0
    text = Config().to_yaml()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


def cmd_detectors(args: argparse.Namespace) -> int:
    from datasi.detectors import registry

    for name, cls in registry().items():
        print(f"{name:12} {cls.description}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from datasi.server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def _use_color(args: argparse.Namespace) -> bool:
    return not args.no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _add_common(p: argparse.ArgumentParser, *, single: bool = True) -> None:
    p.add_argument("--config", "-c", help="YAML configuration file")
    p.add_argument("--target", help="target column for target analysis")
    p.add_argument(
        "--output", "-o", action="append", help="write report (.html/.md/.json); repeatable"
    )
    p.add_argument(
        "--fail-on",
        choices=[s.value for s in Severity],
        help="exit 1 if a finding at or above this severity exists",
    )
    p.add_argument("--disable", help="comma-separated detectors to disable")
    p.add_argument(
        "--outlier-methods", help="comma-separated: iqr,zscore,modified_zscore,isolation_forest"
    )
    p.add_argument(
        "--include-examples", action="store_true", help="include sampled raw values in evidence"
    )
    p.add_argument("--reference-time", help="ISO timestamp treated as 'now' for future-date checks")
    if single:
        p.add_argument("--time-column", help="primary time column (auto-detected by default)")
        p.add_argument("--detectors", help="run only these detectors (comma-separated)")
        p.add_argument("--nrows", type=int, help="read only the first N rows")


def _add_display(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="print the JSON report to stdout")
    p.add_argument("--min-severity", default="warning", choices=[s.value for s in Severity])
    p.add_argument("--limit", type=int, default=30, help="max findings to print")
    p.add_argument("--no-color", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasi",
        description="Find what is wrong with your data before your model does. Runs locally; no data leaves your machine.",
    )
    parser.add_argument("--version", action="version", version=f"datasi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="investigate a dataset and print findings")
    p.add_argument("path")
    _add_common(p)
    _add_display(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("report", help="investigate a dataset and write a report file")
    p.add_argument("path")
    _add_common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "compare", help="compare a reference dataset (e.g. train) with a current one (e.g. test)"
    )
    p.add_argument("reference")
    p.add_argument("current")
    _add_common(p, single=False)
    _add_display(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("schema", help="print the inferred schema")
    p.add_argument("path")
    p.add_argument("--config", "-c")
    p.add_argument("--json", action="store_true")
    p.add_argument("--nrows", type=int)
    p.add_argument(
        "--chunksize",
        type=int,
        help="stream a CSV in chunks (bounded memory, approximate cardinality)",
    )
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("config", help="validate or generate configuration")
    csub = p.add_subparsers(dest="config_command", required=True)
    v = csub.add_parser("validate", help="validate a configuration file")
    v.add_argument("file")
    i = csub.add_parser("init", help="print (or write) the default configuration")
    i.add_argument("--output", "-o")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("detectors", help="list detectors")
    p.set_defaults(func=cmd_detectors)

    p = sub.add_parser("serve", help="start the local web UI (binds to 127.0.0.1)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code: int = args.func(args)
        return code
    except (ValueError, FileNotFoundError) as exc:  # includes LoadError and ConfigError
        print(f"datasi: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
