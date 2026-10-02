"""Runtime and peak memory of `datasi.inspect` on 10K / 100K / 1M rows.

Each size runs in a fresh subprocess so peak RSS is not polluted by earlier runs.
Results (with machine/software details) are written to benchmarks/results/performance.json.

    python benchmarks/bench_inspect.py              # 10k, 100k, 1m
    python benchmarks/bench_inspect.py --sizes 10000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

CHILD = r"""
import json, resource, sys, time
from datasi import inspect
from datasi.synthetic import generate_dataset
rows = int(sys.argv[1])
s = generate_dataset(rows, missingness=0.05, duplicates=0.01, outliers=0.005, category_variants=0.01,
                     redundant_columns=True, missing_after=0.1, seed=0)
frame = s.frame
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
report = inspect(frame, target=s.target, config={"reference_time": "2026-10-01"})
elapsed = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1 if sys.platform == "darwin" else 1024   # ru_maxrss: bytes on macOS, KiB on Linux
print(json.dumps({
    "rows": len(frame), "columns": frame.shape[1],
    "frame_mb": frame.memory_usage(deep=True).sum() / 1e6,
    "seconds": elapsed,
    "peak_rss_mb": peak * scale / 1e6,
    "rss_before_inspect_mb": before * scale / 1e6,
    "findings": len(report.findings),
    "detector_seconds": {d.name: d.seconds for d in report.detectors},
}))
"""


def run(rows: int) -> dict[str, object]:
    out = subprocess.run(
        [sys.executable, "-c", CHILD, str(rows)], capture_output=True, text=True, check=True
    )
    result: dict[str, object] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


def main() -> None:
    import numpy
    import pandas

    import datasi

    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    args = p.parse_args()
    results = []
    for n in args.sizes:
        r = run(n)
        print(
            f"{r['rows']:>9,} rows: {r['seconds']:7.2f}s  peak RSS {r['peak_rss_mb']:8.0f} MB  ({r['findings']} findings)"
        )
        results.append(r)
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "benchmark": "inspect() on generate_dataset(...) with all default detectors and a target",
        "date": time.strftime("%Y-%m-%d"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "versions": {
            "datasi": datasi.__version__,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
        },
        "results": results,
    }
    (RESULTS / "performance.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
