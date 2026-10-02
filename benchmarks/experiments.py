"""Reproducible research experiments on detector reliability.

Questions:
  E1  How accurately do the detectors find synthetic data-quality issues?
  E2  Which detectors raise warnings on clean data (false positives)?
  E3  How sensitive are outlier methods to their thresholds and to skew?
  E4  How does the analysis sample size affect pairwise/target detection?
  E5  How does dataset size affect detection of subtle issues?
  E6  How sensitive is drift detection to shift size and sample size?

Every run is seeded; results go to benchmarks/results/experiments.json and the tables
in docs/research.md are generated from that file by --write-docs (never typed by hand).

    python benchmarks/experiments.py            # all experiments
    python benchmarks/experiments.py --quick    # fewer seeds, smaller sizes (CI smoke)
    python benchmarks/experiments.py --only E3 E6
    python benchmarks/experiments.py --write-docs
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import datasi
from datasi import compare, inspect
from datasi.evaluation import Confusion, evaluate_findings, outlier_row_scores, summarize
from datasi.findings import Severity
from datasi.synthetic import generate_dataset, make_drift_pair

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "experiments.json"
DOC = ROOT / "docs" / "research.md"
CFG = {"reference_time": "2026-10-01"}

ALL_ISSUES: dict[str, Any] = {
    "missingness": 0.15,
    "duplicates": 0.03,
    "outliers": 0.01,
    "category_variants": 0.02,
    "numeric_as_text": True,
    "constant_column": True,
    "redundant_columns": True,
    "sentinels": 0.005,
    "placeholders": 0.01,
    "missing_after": 0.2,
    "missing_by_group": 0.7,
    "leakage": True,
}
SUBTLE: dict[str, Any] = {
    "missing_by_group": 0.3,
    "missing_after": 0.1,
    "category_variants": 0.003,
    "sentinels": 0.002,
    "placeholders": 0.002,
    "outliers": 0.002,
}


def _table(df: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = df.to_dict(orient="records")
    return out


def e1_accuracy(seeds: int, rows: int) -> dict[str, Any]:
    any_sev, warn_sev = [], []
    for seed in range(seeds):
        s = generate_dataset(rows, seed=seed, **ALL_ISSUES)
        r = inspect(s.frame, target=s.target, config=CFG)
        any_sev.append(evaluate_findings(r.findings, s.truth))
        warn_sev.append(evaluate_findings(r.findings, s.truth, min_severity=Severity.WARNING))
    return {
        "params": {"seeds": seeds, "rows": rows, "issues": ALL_ISSUES},
        "any_severity": _table(summarize(any_sev)),
        "warning_or_worse": _table(summarize(warn_sev)),
    }


def e2_clean(seeds: int, sizes: list[int]) -> dict[str, Any]:
    out = []
    for rows in sizes:
        warn: dict[str, int] = defaultdict(int)
        info: dict[str, int] = defaultdict(int)
        for seed in range(seeds):
            s = generate_dataset(rows, seed=1000 + seed)
            r = inspect(s.frame, target=s.target, config=CFG)
            for f in r.findings:
                if f.code == "identifier.detected":
                    continue  # true statements about real ID columns, not alarms
                (warn if f.severity.rank >= Severity.WARNING.rank else info)[f.code] += 1
        out.append(
            {
                "rows": rows,
                "datasets": seeds,
                "warning_per_dataset": {k: v / seeds for k, v in sorted(warn.items())},
                "info_per_dataset": {k: v / seeds for k, v in sorted(info.items())},
            }
        )
    return {"params": {"seeds": seeds, "sizes": sizes}, "results": out}


def e3_outliers(seeds: int, rows: int) -> dict[str, Any]:
    grid = [
        ("iqr", {"iqr_k": 1.5}),
        ("iqr", {"iqr_k": 3.0}),
        ("zscore", {"zscore": 3.0}),
        ("zscore", {"zscore": 4.0}),
        ("modified_zscore", {"modified_zscore": 3.5}),
        ("modified_zscore", {"modified_zscore": 5.0}),
        ("isolation_forest", {"contamination": "auto"}),
        ("isolation_forest", {"contamination": 0.01}),
    ]
    results = []
    for col in ["amount", "age"]:  # amount is log-normal (skewed), age is near-normal
        for method, th in grid:
            total = Confusion(0, 0, 0, 0)
            for seed in range(seeds):
                s = generate_dataset(rows, outliers=0.01, seed=seed)
                truth = s.truth
                truth.issues = [i for i in truth.issues if i.columns == (col,)]
                total = total + outlier_row_scores(s.frame, truth, methods=[method], **th)[method]
            results.append({"column": col, "method": method, "threshold": th, **total.to_dict()})
    return {"params": {"seeds": seeds, "rows": rows, "outlier_rate": 0.01}, "results": results}


def e4_sampling(seeds: int, rows: int, caps: list[int]) -> dict[str, Any]:
    kinds = ["leakage", "redundant", "sum", "missing_group"]
    out = []
    for cap in caps:
        pooled: list[dict[str, Confusion]] = []
        secs = []
        for seed in range(seeds):
            s = generate_dataset(
                rows, seed=seed, leakage=True, redundant_columns=True, missing_by_group=0.5
            )
            t0 = time.perf_counter()
            r = inspect(
                s.frame, target=s.target, config={**CFG, "performance": {"sample_rows": cap}}
            )
            secs.append(time.perf_counter() - t0)
            pooled.append(evaluate_findings(r.findings, s.truth, kinds=kinds))
        out.append(
            {
                "sample_rows": cap,
                "mean_seconds": float(np.mean(secs)),
                "metrics": _table(summarize(pooled)),
            }
        )
    return {"params": {"seeds": seeds, "rows": rows, "caps": caps}, "results": out}


def e5_size(seeds: int, sizes: list[int]) -> dict[str, Any]:
    out = []
    for rows in sizes:
        pooled = []
        for seed in range(seeds):
            s = generate_dataset(rows, seed=seed, **SUBTLE)
            r = inspect(s.frame, config=CFG)
            pooled.append(evaluate_findings(r.findings, s.truth))
        out.append({"rows": rows, "metrics": _table(summarize(pooled))})
    return {"params": {"seeds": seeds, "sizes": sizes, "issues": SUBTLE}, "results": out}


def e6_drift(seeds: int, sizes: list[int], shifts: list[float]) -> dict[str, Any]:
    out = []
    for rows in sizes:
        for shift in shifts:
            hits = 0
            for seed in range(seeds):
                ref, cur, _ = make_drift_pair(rows, shift=shift, category_shift=0.0, seed=seed)
                r = compare(ref, cur)
                hits += any(
                    f.code == "drift.numeric" and "amount" in f.columns
                    for f in r.filter(min_severity="warning")
                )
            out.append({"rows": rows, "shift_log_sd": shift, "detection_rate": hits / seeds})
    return {
        "params": {
            "seeds": seeds,
            "sizes": sizes,
            "shifts": shifts,
            "note": "shift 0 = false alarm rate",
        },
        "results": out,
    }


def run(only: list[str] | None, quick: bool) -> None:
    seeds = 3 if quick else 20
    plan = {
        "E1": lambda: e1_accuracy(seeds, 2000 if quick else 5000),
        "E2": lambda: e2_clean(
            2 if quick else 10, [1000, 10_000] if quick else [1000, 10_000, 100_000]
        ),
        "E3": lambda: e3_outliers(2 if quick else 10, 5000 if quick else 10_000),
        "E4": lambda: e4_sampling(
            1 if quick else 3,
            20_000 if quick else 200_000,
            [2000, 10_000] if quick else [2000, 10_000, 50_000, 200_000],
        ),
        "E5": lambda: e5_size(
            2 if quick else 10, [500, 2000] if quick else [500, 2000, 10_000, 50_000]
        ),
        "E6": lambda: e6_drift(3 if quick else 20, [500, 5000], [0.0, 0.05, 0.1, 0.25, 0.5]),
    }
    existing = json.loads(RESULTS.read_text()) if RESULTS.exists() and not quick else {}
    experiments = existing.get("experiments", {})
    for key in only or list(plan):
        t0 = time.perf_counter()
        experiments[key] = plan[key]()
        experiments[key]["seconds"] = round(time.perf_counter() - t0, 1)
        print(f"{key} done in {experiments[key]['seconds']}s")
    payload = {
        "quick": quick,
        "date": time.strftime("%Y-%m-%d"),
        "versions": {"datasi": datasi.__version__, "python": platform.python_version()},
        "platform": platform.platform(),
        "experiments": experiments,
    }
    target = RESULTS if not quick else RESULTS.with_name("experiments-quick.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {target}")


def _f(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_docs() -> None:
    data = json.loads(RESULTS.read_text())
    ex = data["experiments"]
    L: list[str] = [
        "# Research: how reliable are DataSI's detectors?",
        "",
        "> Generated by `python benchmarks/experiments.py --write-docs` from",
        f"> `benchmarks/results/experiments.json` (DataSI {data['versions']['datasi']}, {data['date']}).",
        "> Do not edit the numbers by hand; rerun the experiments instead.",
        "",
        "All data is synthetic (`datasi.synthetic`), so ground truth is known exactly. Metrics are",
        "pooled over seeds (micro-averaged). Column-level: the universe is every column of the",
        "dataset. n/a means the ratio is undefined (0/0).",
        "",
    ]
    if "E1" in ex:
        e = ex["E1"]
        L += [
            f"## E1 - Detection accuracy ({e['params']['seeds']} datasets x {e['params']['rows']:,} rows, all issue types injected)",
            "",
        ]
        for key, title in [
            ("any_severity", "Any severity"),
            ("warning_or_worse", "Warning or worse"),
        ]:
            L += [
                f"**{title}**",
                "",
                "| issue | TP | FP | FN | precision | recall | FPR |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in e[key]:
                L.append(
                    f"| {r['issue']} | {r['tp']} | {r['fp']} | {r['fn']} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['fpr'])} |"
                )
            L.append("")
    if "E2" in ex:
        e = ex["E2"]
        L += [
            "## E2 - Findings on clean data (false-positive behaviour)",
            "",
            "Mean number of findings per clean dataset, by code (identifier facts excluded).",
            "",
        ]
        for r in e["results"]:
            L += [
                f"**{r['rows']:,} rows ({r['datasets']} datasets)** - warning or worse: "
                + (
                    ", ".join(f"`{k}` {v:.1f}" for k, v in r["warning_per_dataset"].items())
                    or "none"
                )
                + "; info: "
                + (", ".join(f"`{k}` {v:.1f}" for k, v in r["info_per_dataset"].items()) or "none"),
                "",
            ]
    if "E3" in ex:
        e = ex["E3"]
        L += [
            f"## E3 - Outlier methods: thresholds and skew ({e['params']['seeds']} seeds x {e['params']['rows']:,} rows, 1% injected per column)",
            "",
            "Row-level, raw scale. `amount` is log-normal (skewed); `age` is near-normal.",
            "",
            "| column | method | threshold | precision | recall | FPR |",
            "|---|---|---|---|---|---|",
        ]
        for r in e["results"]:
            th = ", ".join(f"{k}={v}" for k, v in r["threshold"].items())
            L.append(
                f"| {r['column']} | {r['method']} | {th} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['fpr'])} |"
            )
        L.append("")
    if "E4" in ex:
        e = ex["E4"]
        L += [
            f"## E4 - Analysis sample size ({e['params']['seeds']} datasets x {e['params']['rows']:,} rows)",
            "",
            "| sample_rows | mean seconds | "
            + " | ".join(f"{m['issue']} recall" for m in e["results"][0]["metrics"])
            + " |",
            "|---|---|" + "---|" * len(e["results"][0]["metrics"]),
        ]
        for r in e["results"]:
            L.append(
                f"| {r['sample_rows']:,} | {r['mean_seconds']:.2f} | "
                + " | ".join(_f(m["recall"]) for m in r["metrics"])
                + " |"
            )
        L.append("")
    if "E5" in ex:
        e = ex["E5"]
        kinds = [m["issue"] for m in e["results"][0]["metrics"]]
        L += [
            f"## E5 - Dataset size vs subtle issues ({e['params']['seeds']} datasets per size)",
            "",
            "Recall (precision in brackets) per issue type.",
            "",
            "| rows | " + " | ".join(kinds) + " |",
            "|---|" + "---|" * len(kinds),
        ]
        for r in e["results"]:
            cells = {m["issue"]: f"{_f(m['recall'])} ({_f(m['precision'])})" for m in r["metrics"]}
            L.append(f"| {r['rows']:,} | " + " | ".join(cells.get(k, "n/a") for k in kinds) + " |")
        L.append("")
    if "E6" in ex:
        e = ex["E6"]
        shifts = e["params"]["shifts"]
        L += [
            f"## E6 - Drift detection rate ({e['params']['seeds']} pairs per cell)",
            "",
            "Share of reference/current pairs where `amount` drift is reported at warning or worse.",
            "Shift is in log-SD units of the log-normal; shift 0 is the false-alarm rate.",
            "",
            "| rows per dataset | " + " | ".join(str(s) for s in shifts) + " |",
            "|---|" + "---|" * len(shifts),
        ]
        for rows in e["params"]["sizes"]:
            rates = {
                r["shift_log_sd"]: r["detection_rate"] for r in e["results"] if r["rows"] == rows
            }
            L.append(f"| {rows:,} | " + " | ".join(_f(rates[s]) for s in shifts) + " |")
        L.append("")
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")
    print(f"wrote {DOC}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=["E1", "E2", "E3", "E4", "E5", "E6"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--write-docs", action="store_true")
    args = p.parse_args()
    if args.write_docs:
        write_docs()
        return
    run(args.only, args.quick)


if __name__ == "__main__":
    main()
