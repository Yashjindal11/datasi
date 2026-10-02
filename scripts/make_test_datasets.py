"""Regenerate the small CSV fixtures in tests/data (deterministic).

Run: python scripts/make_test_datasets.py
Each file has known problems; tests/data/expected.json lists the finding codes that
DataSI must report for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasi.synthetic import generate_dataset, make_drift_pair

OUT = Path(__file__).resolve().parents[1] / "tests" / "data"


def temporal(
    rows_per_day_before: int = 30, rows_per_day_after: int = 10, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", "2025-08-31", freq="D")
    counts = np.where(days < "2025-05-01", rows_per_day_before, rows_per_day_after)
    ts = np.repeat(days.values, counts)
    n = ts.size
    late = ts >= np.datetime64("2025-05-01")
    return pd.DataFrame(
        {
            "event_date": pd.to_datetime(ts).strftime("%Y-%m-%d"),
            "sensor": rng.choice(["s1", "s2", "s3"], n),
            "status": np.where(
                late,
                rng.choice(["ok", "warn", "fault"], n),
                rng.choice(["ok", "warn", "legacy"], n),
            ),
            "reading": np.round(np.where(late, rng.normal(30, 2, n), rng.normal(20, 2, n)), 3),
        }
    )


def types(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    fmt = rng.choice(["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"], n, p=[0.8, 0.15, 0.05])
    carrier = rng.choice(["United", "Delta", "American"], n).astype(object)
    carrier[:20] = "UNITED"
    carrier[20:30] = "united"
    status = rng.choice(["active", "closed"], n).astype(object)
    status[:40] = "unknown"
    return pd.DataFrame(
        {
            "booking_ref": [f"BK-{i:06d}" for i in range(n)],
            "booked_on": [d.strftime(f) for d, f in zip(dates, fmt, strict=True)],
            "fare": [f"${x:,.2f}" for x in rng.uniform(80, 2500, n)],
            "refundable": rng.choice(["Yes", "Y", "No"], n),
            "carrier": carrier,
            "status": status,
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected: dict[str, list[str]] = {}

    s = generate_dataset(1500, missingness=0.15, missing_by_group=0.7, missing_after=0.25, seed=1)
    s.frame.to_csv(OUT / "dataset_missing.csv", index=False)
    expected["dataset_missing.csv"] = ["missing.high", "missing.by_group", "missing.over_time"]

    s = generate_dataset(1500, duplicates=0.05, seed=2)
    s.frame.to_csv(OUT / "dataset_duplicates.csv", index=False)
    expected["dataset_duplicates.csv"] = ["duplicates.exact", "duplicates.key"]

    s = generate_dataset(1500, outliers=0.01, sentinels=0.005, seed=3)
    s.frame.to_csv(OUT / "dataset_outliers.csv", index=False)
    expected["dataset_outliers.csv"] = ["outliers.univariate", "numeric.sentinel"]

    temporal().to_csv(OUT / "dataset_temporal.csv", index=False)
    expected["dataset_temporal.csv"] = [
        "temporal.volume_shift",
        "temporal.distribution_shift",
        "temporal.new_category",
        "temporal.vanished_category",
    ]

    types().to_csv(OUT / "dataset_types.csv", index=False)
    expected["dataset_types.csv"] = [
        "types.numeric_text",
        "types.date_formats",
        "types.boolean_text",
        "categorical.variants",
        "strings.placeholder",
    ]

    ref, cur, _ = make_drift_pair(1500, shift=0.6, category_shift=0.25, seed=4)
    ref.to_csv(OUT / "dataset_drift_train.csv", index=False)
    cur.to_csv(OUT / "dataset_drift_test.csv", index=False)
    expected["compare:dataset_drift_train.csv:dataset_drift_test.csv"] = [
        "drift.numeric",
        "drift.categorical",
    ]

    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    for p in sorted(OUT.glob("*.csv")):
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
