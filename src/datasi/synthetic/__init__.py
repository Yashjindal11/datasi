"""Synthetic datasets with known, controllable data-quality problems.

Every injected problem is recorded in a :class:`GroundTruth`, so detector output can
be scored (see :mod:`datasi.evaluation`). Generation is fully determined by ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Issue:
    kind: str
    columns: tuple[str, ...] = ()
    rows: tuple[int, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruth:
    issues: list[Issue] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    def add(self, kind: str, columns: tuple[str, ...] = (), rows: Any = (), **detail: Any) -> None:
        self.issues.append(Issue(kind, tuple(columns), tuple(int(r) for r in rows), detail))

    def of_kind(self, kind: str) -> list[Issue]:
        return [i for i in self.issues if i.kind == kind]

    def kinds(self) -> list[str]:
        return sorted({i.kind for i in self.issues})

    def rows(self, kind: str, column: str | None = None) -> set[int]:
        return {
            r for i in self.of_kind(kind) if column is None or column in i.columns for r in i.rows
        }


@dataclass
class SyntheticDataset:
    frame: pd.DataFrame
    truth: GroundTruth
    target: str = "churned"
    time_column: str = "order_ts"


def clean_frame(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """A realistic, problem-free orders table (problems are injected separately)."""
    n_customers = max(20, rows // 5)
    start = pd.Timestamp("2025-01-01")
    seconds = np.sort(rng.uniform(0, 365 * 86400, rows))
    age = np.clip(rng.normal(40, 12, rows), 18, 90).round()
    score = rng.uniform(0, 1, rows)
    logit = -1.0 + 2.0 * score - 0.02 * (age - 40)
    churned = rng.uniform(0, 1, rows) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "order_id": np.arange(100_000, 100_000 + rows),
            "customer_id": rng.integers(1, n_customers + 1, rows),
            "order_ts": start + pd.to_timedelta(seconds, unit="s"),
            "region": rng.choice(
                ["North", "South", "East", "West", "Central"], rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]
            ),
            "channel": rng.choice(["web", "store", "phone"], rows, p=[0.6, 0.3, 0.1]),
            "amount": np.round(rng.lognormal(3.5, 0.4, rows), 2),
            "quantity": rng.poisson(2, rows) + 1,
            "age": age,
            "score": np.round(score, 4),
            "is_member": rng.uniform(0, 1, rows) < 0.4,
            "churned": churned.astype(int),
        }
    )


def generate_dataset(
    rows: int = 10_000,
    *,
    missingness: float = 0.0,
    duplicates: float = 0.0,
    outliers: float = 0.0,
    category_variants: float = 0.0,
    numeric_as_text: bool = False,
    constant_column: bool = False,
    redundant_columns: bool = False,
    sentinels: float = 0.0,
    placeholders: float = 0.0,
    missing_after: float | None = None,
    missing_by_group: float | None = None,
    leakage: bool = False,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a dataset with controlled corruption.

    Rates are fractions of rows (or cells of the affected columns):

    * ``missingness`` - MCAR nulls in ``amount``, ``age`` and ``region``.
    * ``duplicates`` - exact copies of random rows appended (then shuffled in place).
    * ``outliers`` - cells of ``amount`` and ``age`` replaced by values 8-20 robust SDs away.
    * ``category_variants`` - ``region`` labels rewritten with case/whitespace variants.
    * ``sentinels`` - ``age`` cells set to -999.
    * ``placeholders`` - ``channel`` cells set to 'N/A'.
    * ``missing_after`` - ``score`` becomes missing for this share of rows at the end of
      the time range (a temporal missingness change).
    * ``missing_by_group`` - ``quantity`` missing at this rate only where region = 'East'.
    * ``numeric_as_text`` / ``constant_column`` / ``redundant_columns`` / ``leakage`` add
      columns with those properties.
    """
    rng = np.random.default_rng(seed)
    df = clean_frame(rows, rng)
    truth = GroundTruth()

    if redundant_columns:
        df["amount_cents"] = np.round(df["amount"] * 100, 0)
        df["shipping"] = np.round(rng.uniform(0, 15, rows), 2)
        df["total"] = df["amount"] + df["shipping"]
        truth.add("redundant", ("amount", "amount_cents"))
        truth.add("sum", ("total", "amount", "shipping"))
    if constant_column:
        df["source_system"] = "ERP"
        truth.add("constant", ("source_system",))
    if leakage:
        df["refund_issued"] = (df["churned"] == 1) & (rng.uniform(0, 1, rows) < 0.995)
        df["refund_issued"] = df["refund_issued"].astype(int) * np.round(
            rng.uniform(5, 50, rows), 2
        )
        truth.add("leakage", ("refund_issued",))

    def pick(rate: float, n: int = rows) -> np.ndarray:
        k = round(rate * n)
        return np.sort(rng.choice(n, size=k, replace=False)) if k else np.array([], dtype=int)

    if outliers > 0:
        for col in ["amount", "age"]:
            idx = pick(outliers)
            x = df[col].to_numpy(dtype=float)
            med = np.median(x)
            scale = 1.4826 * np.median(np.abs(x - med))
            signs = rng.choice([-1, 1], idx.size) if col == "age" else np.ones(idx.size)
            vals = med + signs * rng.uniform(8, 20, idx.size) * scale
            if col == "age":
                vals = np.round(vals)
            df[col] = df[col].astype(float)
            df.loc[idx, col] = vals
            truth.add("outliers", (col,), idx)
    if sentinels > 0:
        idx = pick(sentinels)
        df["age"] = df["age"].astype(float)
        df.loc[idx, "age"] = -999.0
        truth.add("sentinel", ("age",), idx)
    if category_variants > 0:
        idx = pick(category_variants)
        variants = df.loc[idx, "region"].map(
            lambda v: rng.choice([v.upper(), v.lower(), f" {v}", f"{v} "])
        )
        df.loc[idx, "region"] = variants.to_numpy()
        truth.add("category_variants", ("region",), idx)
    if placeholders > 0:
        idx = pick(placeholders)
        df.loc[idx, "channel"] = "N/A"
        truth.add("placeholder", ("channel",), idx)
    if numeric_as_text:
        df["discount_text"] = [f"{x:.2f}" for x in rng.uniform(0, 30, rows)]
        truth.add("numeric_as_text", ("discount_text",))
    if missingness > 0:
        for col in ["amount", "age", "region"]:
            idx = pick(missingness)
            df.loc[idx, col] = None
            truth.add("missing", (col,), idx)
    if missing_after is not None and missing_after > 0:
        k = round(missing_after * rows)
        idx = np.arange(rows - k, rows)
        df.loc[idx, "score"] = np.nan
        truth.add("missing_temporal", ("score",), idx)
    if missing_by_group is not None and missing_by_group > 0:
        east = np.flatnonzero(
            (df["region"].astype(str).str.strip().str.lower() == "east").to_numpy()
        )
        k = round(missing_by_group * east.size)
        idx = np.sort(rng.choice(east, size=k, replace=False)) if k else np.array([], dtype=int)
        df["quantity"] = df["quantity"].astype(float)
        df.loc[idx, "quantity"] = np.nan
        truth.add("missing_group", ("quantity", "region"), idx)
    if duplicates > 0:
        k = round(duplicates * rows)
        src = rng.choice(rows, size=k, replace=False)
        copies = df.iloc[src]
        df = pd.concat([df, copies], ignore_index=True)
        order = np.argsort(np.concatenate([np.arange(rows), src + 0.5]), kind="stable")
        df = df.iloc[order].reset_index(drop=True)
        # Positions of the later copy in each pair after interleaving.
        is_copy = np.concatenate([np.zeros(rows, bool), np.ones(k, bool)])[order]
        dup_rows = np.flatnonzero(is_copy)
        for issue in list(truth.issues):
            if issue.rows:
                remap = np.flatnonzero(~is_copy)
                truth.issues[truth.issues.index(issue)] = Issue(
                    issue.kind,
                    issue.columns,
                    tuple(int(remap[r]) for r in issue.rows),
                    issue.detail,
                )
        truth.add("duplicates", (), dup_rows)
    truth.columns = list(df.columns)
    return SyntheticDataset(df, truth)


def make_drift_pair(
    rows: int = 5000, *, shift: float = 0.5, category_shift: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:
    """Reference/current frames where ``amount`` shifts by ``shift`` log-SDs and the
    ``channel`` mix moves by ``category_shift`` toward 'phone'. Other columns are i.i.d."""
    rng = np.random.default_rng(seed)
    ref = clean_frame(rows, rng)
    cur = clean_frame(rows, rng)
    truth = GroundTruth()
    if shift:
        cur["amount"] = np.round(cur["amount"] * np.exp(shift * 0.4), 2)
        truth.add("drift", ("amount",), shift=shift)
    if category_shift:
        idx = rng.choice(rows, int(category_shift * rows), replace=False)
        cur.loc[idx, "channel"] = "phone"
        truth.add("drift", ("channel",), shift=category_shift)
    truth.columns = list(ref.columns)
    return ref, cur, truth
