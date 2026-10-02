"""Schema inference: physical dtype, semantic column type and storage hints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from datasi.statistics import strings as st


class ColumnType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


class StorageHint(StrEnum):
    """The values mean one type but are stored as another."""

    NUMERIC_AS_STRING = "numeric_as_string"
    CURRENCY_AS_STRING = "currency_as_string"
    PERCENT_AS_STRING = "percent_as_string"
    DATETIME_AS_STRING = "datetime_as_string"
    BOOLEAN_AS_TEXT = "boolean_as_text"
    BOOLEAN_AS_NUMBER = "boolean_as_number"
    MIXED_TYPES = "mixed_types"


_ID_NAME_RE = re.compile(
    r"(^|[_\s.-])(id|ids|uuid|guid|key|pk|identifier|number|no|num|nr|code|ref|hash)$"
    r"|^(id|uuid|guid|pk|key)([_\s.-]|$)"
    r"|[a-z](Id|ID|Key|Uuid|UUID|Number|No)$",
)


def name_suggests_identifier(name: str) -> bool:
    return bool(_ID_NAME_RE.search(name) or _ID_NAME_RE.search(name.lower()))


@dataclass
class ColumnSchema:
    name: str
    position: int
    dtype: str
    column_type: ColumnType
    n_rows: int
    n_missing: int
    n_unique: int
    memory_bytes: int
    hints: list[StorageHint] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    python_types: dict[str, int] = field(default_factory=dict)

    @property
    def nullable(self) -> bool:
        return self.n_missing > 0

    @property
    def n_present(self) -> int:
        return self.n_rows - self.n_missing

    @property
    def missing_ratio(self) -> float:
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    @property
    def unique_ratio(self) -> float:
        return self.n_unique / self.n_present if self.n_present else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "position": self.position,
            "dtype": self.dtype,
            "type": self.column_type.value,
            "nullable": self.nullable,
            "missing": self.n_missing,
            "missing_ratio": round(self.missing_ratio, 6),
            "unique": self.n_unique,
            "unique_ratio": round(self.unique_ratio, 6),
            "memory_bytes": self.memory_bytes,
            "hints": [h.value for h in self.hints],
            "reasons": self.reasons,
        }


@dataclass
class Schema:
    n_rows: int
    n_columns: int
    memory_bytes: int
    columns: dict[str, ColumnSchema]

    def of_type(self, *types: ColumnType) -> list[str]:
        return [c.name for c in self.columns.values() if c.column_type in types]

    def __getitem__(self, name: str) -> ColumnSchema:
        return self.columns[name]

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.columns.values():
            counts[c.column_type.value] = counts.get(c.column_type.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.n_rows,
            "columns": self.n_columns,
            "memory_bytes": self.memory_bytes,
            "type_counts": self.type_counts(),
            "fields": [c.to_dict() for c in self.columns.values()],
        }


def _is_stringish(s: pd.Series) -> bool:
    return bool(pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s))


def python_type_counts(s: pd.Series, limit: int = 20000) -> dict[str, int]:
    """Counts of Python types among non-null values of an object column (sampled)."""
    if not pd.api.types.is_object_dtype(s):
        return {}
    vals = s.dropna()
    if len(vals) > limit:
        vals = vals.sample(limit, random_state=0)
    counts = vals.map(lambda v: type(v).__name__).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def identifier_signals(
    s: pd.Series,
    name: str,
    unique_ratio: float,
    counts: pd.Series | None = None,
    unique_threshold: float = 0.95,
) -> list[str]:
    """Reasons a column looks like an identifier (empty list when it does not)."""
    reasons: list[str] = []
    present = s.dropna()
    if present.empty:
        return reasons
    if name_suggests_identifier(name):
        reasons.append(f"name '{name}' follows an identifier naming pattern")
    if unique_ratio >= unique_threshold and len(present) >= 20:
        reasons.append(f"{unique_ratio:.1%} of non-null values are unique")
    is_int = pd.api.types.is_integer_dtype(s) or (
        pd.api.types.is_float_dtype(s) and bool(np.all(np.mod(present.to_numpy(float), 1) == 0))
    )
    if is_int and len(present) >= 20 and not pd.api.types.is_bool_dtype(s):
        arr = present.to_numpy(dtype=float)
        diffs = np.diff(arr)
        if len(diffs) and (np.all(diffs > 0) or np.all(diffs < 0)):
            step_one = float(np.mean(np.abs(diffs) == 1))
            reasons.append(
                "values are strictly monotonic"
                + (f" ({step_one:.0%} of steps are exactly 1)" if step_one >= 0.5 else "")
            )
    if counts is not None and _is_stringish(s):
        if st.uuid_ratio(counts) >= 0.95:
            reasons.append("values are UUID-formatted")
        elif len(counts) >= 20:
            shapes = pd.Series([st.shape(str(v)) for v in counts.index[:2000]]).value_counts()
            has_space = any(" " in str(v) for v in counts.index[:2000])
            if shapes.iloc[0] / shapes.sum() >= 0.95 and not has_space and "9" in shapes.index[0]:
                reasons.append(f"values share one fixed format ('{shapes.index[0]}')")
    if counts is not None and len(counts) >= 20 and unique_ratio >= 0.5 and counts.iloc[0] <= 2:
        spaced = np.mean([" " in str(v).strip() for v in counts.index[:2000]])
        if spaced < 0.1:
            reasons.append("no value occurs more than twice (low reuse)")
    return reasons


def _looks_like_identifier(reasons: list[str], unique_ratio: float, n_unique: int) -> bool:
    if not reasons:
        return False
    named = any(r.startswith("name ") for r in reasons)
    unique = any("unique" in r for r in reasons)
    structural = any(
        k in r for r in reasons for k in ("monotonic", "UUID", "fixed format", "low reuse")
    )
    if any("UUID" in r for r in reasons):
        return True
    if unique and (named or structural):
        return True
    # Repeated entity keys (customer_id in a transactions table) are identifiers too.
    return named and n_unique >= 20 and unique_ratio >= 0.01


def infer_column(
    s: pd.Series,
    name: str,
    position: int,
    *,
    parse_ratio: float = 0.95,
    unique_threshold: float = 0.95,
    forced_identifier: bool = False,
) -> ColumnSchema:
    n_rows = len(s)
    n_missing = int(s.isna().sum())
    n_present = n_rows - n_missing
    counts = st.string_counts(s) if _is_stringish(s) else None
    try:
        n_unique = int(s.nunique(dropna=True))
    except TypeError:  # unhashable objects such as lists/dicts
        n_unique = int(s.dropna().map(repr).nunique())
    col = ColumnSchema(
        name=name,
        position=position,
        dtype=str(s.dtype),
        column_type=ColumnType.UNKNOWN,
        n_rows=n_rows,
        n_missing=n_missing,
        n_unique=n_unique,
        memory_bytes=int(s.memory_usage(deep=True, index=False)),
    )
    unique_ratio = col.unique_ratio
    col.python_types = python_type_counts(s)
    non_str = {k: v for k, v in col.python_types.items() if k != "str"}
    if col.python_types and non_str and len(col.python_types) > 1:
        col.hints.append(StorageHint.MIXED_TYPES)
        col.reasons.append(f"object column mixes Python types {sorted(col.python_types)}")

    def done(t: ColumnType, reason: str) -> ColumnSchema:
        col.column_type = t
        col.reasons.insert(0, reason)
        return col

    if n_present == 0:
        return done(ColumnType.UNKNOWN, "all values are missing")
    if forced_identifier:
        return done(ColumnType.IDENTIFIER, "declared as an identifier in configuration")
    if n_unique == 1:
        return done(ColumnType.CONSTANT, "a single distinct non-null value")
    if pd.api.types.is_bool_dtype(s):
        return done(ColumnType.BOOLEAN, "boolean dtype")
    if pd.api.types.is_datetime64_any_dtype(s):
        return done(ColumnType.DATETIME, "datetime dtype")
    if isinstance(s.dtype, pd.CategoricalDtype):
        return done(ColumnType.CATEGORICAL, "pandas categorical dtype")
    if pd.api.types.is_numeric_dtype(s):
        uniq = pd.unique(s.dropna())
        if n_unique == 2 and set(np.asarray(uniq, dtype=float).tolist()) <= {0.0, 1.0}:
            col.hints.append(StorageHint.BOOLEAN_AS_NUMBER)
            return done(ColumnType.BOOLEAN, "numeric column containing only 0 and 1")
        reasons = identifier_signals(s, name, unique_ratio, unique_threshold=unique_threshold)
        if _looks_like_identifier(reasons, unique_ratio, n_unique):
            col.reasons.extend(reasons)
            return done(ColumnType.IDENTIFIER, "numeric values with identifier characteristics")
        return done(ColumnType.NUMERIC, "numeric dtype")
    if counts is None:
        return done(ColumnType.UNKNOWN, f"unsupported dtype {s.dtype}")
    if non_str and sum(non_str.values()) / max(sum(col.python_types.values()), 1) > 0.5:
        return done(ColumnType.UNKNOWN, "majority of values are non-string Python objects")

    if st.boolean_like(counts):
        col.hints.append(StorageHint.BOOLEAN_AS_TEXT)
        return done(ColumnType.BOOLEAN, "text values are boolean tokens (e.g. yes/no, true/false)")
    num = st.numeric_parse(counts)
    if num.ratio >= parse_ratio:
        hint = {
            "currency": StorageHint.CURRENCY_AS_STRING,
            "percent": StorageHint.PERCENT_AS_STRING,
        }.get(num.kind, StorageHint.NUMERIC_AS_STRING)
        col.hints.append(hint)
        reasons = identifier_signals(s, name, unique_ratio, counts, unique_threshold)
        if num.kind == "plain" and _looks_like_identifier(reasons, unique_ratio, n_unique):
            col.reasons.extend(reasons)
            return done(ColumnType.IDENTIFIER, "numeric-looking strings with identifier traits")
        return done(ColumnType.NUMERIC, f"{num.ratio:.1%} of text values parse as numbers")
    if st.datetime_parse(counts) >= parse_ratio:
        col.hints.append(StorageHint.DATETIME_AS_STRING)
        return done(ColumnType.DATETIME, "text values parse as dates/times")
    reasons = identifier_signals(s, name, unique_ratio, counts, unique_threshold)
    if _looks_like_identifier(reasons, unique_ratio, n_unique):
        col.reasons.extend(reasons)
        return done(ColumnType.IDENTIFIER, "text values with identifier characteristics")
    lengths = st.length_stats(counts)
    sample_vals = [str(v) for v in counts.index[:500]]
    spaced = np.mean([v.count(" ") >= 3 for v in sample_vals]) if sample_vals else 0.0
    if lengths.get("mean", 0) >= 40 or (spaced >= 0.5 and unique_ratio >= 0.5):
        return done(ColumnType.TEXT, "long or multi-word free-form values")
    return done(ColumnType.CATEGORICAL, "text values with repeated levels")


def infer_schema(
    frame: pd.DataFrame,
    *,
    parse_ratio: float = 0.95,
    unique_threshold: float = 0.95,
    id_columns: list[str] | None = None,
) -> Schema:
    forced = set(id_columns or [])
    cols = {
        name: infer_column(
            frame[name],
            name,
            i,
            parse_ratio=parse_ratio,
            unique_threshold=unique_threshold,
            forced_identifier=name in forced,
        )
        for i, name in enumerate(frame.columns)
    }
    return Schema(
        n_rows=len(frame),
        n_columns=frame.shape[1],
        memory_bytes=sum(c.memory_bytes for c in cols.values()),
        columns=cols,
    )
