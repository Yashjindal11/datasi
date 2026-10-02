"""Pattern utilities for string values.

All functions operate on *distinct* values with their counts (a value_counts Series), so
cost scales with cardinality instead of row count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

BOOLEAN_TOKENS = frozenset(
    {"true", "false", "t", "f", "yes", "no", "y", "n", "0", "1", "on", "off"}
)
_TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1", "on"})

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)
_PLAIN_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS_NUMBER_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_CURRENCY_RE = re.compile(
    r"^\(?[+-]?\s*[$€£¥₹]\s*[+-]?[\d,]*\.?\d+\)?$|^[+-]?[\d,]*\.?\d+\s*[$€£¥₹]$"
)
_PERCENT_RE = re.compile(r"^[+-]?\d*\.?\d+\s*%$")

# Date shapes after digits -> 9; ordered most specific first.
_DATE_SHAPES: dict[str, str] = {
    r"^9999-99-99[T ]99:99(:99(\.9+)?)?(Z|[+-]99:?99)?$": "ISO datetime (YYYY-MM-DD hh:mm)",
    r"^9999-99-99$": "ISO date (YYYY-MM-DD)",
    r"^9999/99/99$": "YYYY/MM/DD",
    r"^9999\.99\.99$": "YYYY.MM.DD",
    r"^99?/99?/9999$": "MM/DD/YYYY or DD/MM/YYYY",
    r"^99?/99?/99$": "MM/DD/YY or DD/MM/YY",
    r"^99?-99?-9999$": "MM-DD-YYYY or DD-MM-YYYY",
    r"^99?\.99?\.9999$": "DD.MM.YYYY",
    r"^99?/99?/9999 99?:99(:99)?( ?[AaPp][Mm])?$": "MM/DD/YYYY hh:mm",
    r"^99999999$": "YYYYMMDD",
    r"^99? [A-Za-z]{3,9},? 9999$": "D Month YYYY",
    r"^[A-Za-z]{3,9} 99?,? 9999$": "Month D, YYYY",
}
_DATE_SHAPE_RES = [(re.compile(k), v) for k, v in _DATE_SHAPES.items()]
_TZ_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def shape(value: str, max_len: int = 40) -> str:
    """Collapse a string into its character-class shape: digits->9, letters->A, rest kept."""
    out = []
    for ch in value[:max_len]:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A")
        else:
            out.append(ch)
    return "".join(out) + ("…" if len(value) > max_len else "")


def compact_shape(value: str) -> str:
    """Shape with runs collapsed (A+ / 9+), used to group 'same format, different length'."""
    return re.sub(r"(A)A+|(9)9+", lambda m: (m.group(1) or m.group(2)) + "+", shape(value))


def string_counts(series: pd.Series) -> pd.Series:
    """value_counts of non-null values as strings (no stripping, no case folding)."""
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype="int64")
    counts = s.value_counts(dropna=True)
    counts.index = counts.index.map(str)
    return counts.groupby(level=0, sort=False).sum()


@dataclass(frozen=True)
class ParseResult:
    ratio: float  # share of non-null rows that parse
    kind: str  # plain | thousands | currency | percent | mixed | none
    kinds: dict[str, int]  # rows per kind


def number_kind(raw: str) -> str | None:
    """'plain' | 'thousands' | 'currency' | 'percent' for numeric-looking text, else None."""
    v = raw.strip()
    if _PLAIN_NUMBER_RE.match(v):
        return "plain"
    if _THOUSANDS_NUMBER_RE.match(v):
        return "thousands"
    if _CURRENCY_RE.match(v):
        return "currency"
    if _PERCENT_RE.match(v):
        return "percent"
    return None


def numeric_parse(counts: pd.Series) -> ParseResult:
    """How many values look like numbers, and in which textual convention."""
    total = int(counts.sum())
    if total == 0:
        return ParseResult(0.0, "none", {})
    kinds: dict[str, int] = {}
    for raw, n in counts.items():
        k = number_kind(str(raw))
        if k is None:
            continue
        kinds[k] = kinds.get(k, 0) + int(n)
    parsed = sum(kinds.values())
    if not kinds:
        kind = "none"
    elif len(kinds) == 1:
        kind = next(iter(kinds))
    else:
        kind = (
            max(kinds, key=lambda k: kinds[k]) if max(kinds.values()) / parsed > 0.99 else "mixed"
        )
    return ParseResult(parsed / total, kind, kinds)


def to_number(values: pd.Series) -> pd.Series:
    """Parse numeric-looking strings (currency, thousands, percent) into floats; others NaN."""
    s = values.astype("string").str.strip()
    neg = s.str.match(r"^\(.*\)$").fillna(False)
    pct = s.str.endswith("%").fillna(False)
    cleaned = s.str.replace(r"[$€£¥₹,%()\s]", "", regex=True)
    out = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    out = out.where(~neg.astype(bool), -out)
    return out.where(~pct.astype(bool), out / 100.0)


def date_shapes(counts: pd.Series) -> dict[str, int]:
    """Rows per recognised date format (unrecognised values are not counted)."""
    found: dict[str, int] = {}
    for raw, n in counts.items():
        sh = shape(str(raw).strip())
        for rx, label in _DATE_SHAPE_RES:
            if rx.match(sh):
                found[label] = found.get(label, 0) + int(n)
                break
    return found


def datetime_parse(counts: pd.Series, max_distinct: int = 5000) -> float:
    """Share of non-null rows whose value parses as a date/time.

    Bare numbers are excluded (an integer column is not a date column just because
    some integers are valid epoch offsets). Only the most frequent distinct values
    are tested when cardinality is very large, so the ratio is a weighted estimate.
    """
    total = int(counts.sum())
    if total == 0:
        return 0.0
    top = counts.iloc[:max_distinct]
    values = pd.Series(top.index.astype(str), index=top.index)
    shaped = values.map(lambda v: shape(v.strip()))
    candidate = ~shaped.str.fullmatch(r"[+-]?9*\.?9+") & shaped.str.contains("9", regex=False)
    if not candidate.any():
        return 0.0
    vals = values[candidate].str.strip()
    parsed = pd.to_datetime(vals, errors="coerce", format="mixed", utc=True)
    ok = parsed.notna().to_numpy()
    weights = top[candidate].to_numpy()
    return float(weights[ok].sum() / top.sum())


def parse_datetimes(series: pd.Series) -> pd.Series:
    """Best-effort datetime parsing of a column (strings or datetimes) to tz-naive UTC."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("datetime64[ns]")
    s = series.astype("string").str.strip()
    parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_localize(None)


def timezone_styles(counts: pd.Series) -> dict[str, int]:
    """Rows per timezone notation among datetime strings: none / Z / offset."""
    out: dict[str, int] = {}
    for raw, n in counts.items():
        m = _TZ_RE.search(str(raw).strip())
        key = "none" if not m else ("Z" if m.group(1) == "Z" else "offset")
        out[key] = out.get(key, 0) + int(n)
    return out


def boolean_like(counts: pd.Series) -> bool:
    """True if every distinct value is a boolean token and there are at most two meanings."""
    if counts.empty:
        return False
    tokens = {str(v).strip().lower() for v in counts.index}
    if not tokens <= BOOLEAN_TOKENS:
        return False
    meanings = {t in _TRUE_TOKENS for t in tokens}
    return len(meanings) == 2 and tokens != {"0", "1"}


def uuid_ratio(counts: pd.Series) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    hits = sum(int(n) for v, n in counts.items() if UUID_RE.match(str(v).strip()))
    return hits / total


def normalize_label(value: str) -> str:
    """Normalisation used to group spelling variants: case, whitespace, punctuation."""
    v = re.sub(r"[\s_\-./]+", " ", value.strip().lower())
    return re.sub(r"[^\w ]", "", v).strip()


def unusual_characters(value: str) -> list[str]:
    """Control characters, replacement characters and zero-width/invisible code points."""
    bad = []
    for ch in value:
        o = ord(ch)
        if (
            (o < 32 and ch not in "\t\n\r")
            or o == 127
            or ch == "\ufffd"
            or ch in "\u200b\u200c\u200d\u2060\ufeff\u00a0"
        ):
            bad.append(f"U+{o:04X}")
    return bad


def length_stats(counts: pd.Series) -> dict[str, float]:
    lengths = np.array([len(str(v)) for v in counts.index], dtype=float)
    weights = counts.to_numpy(dtype=float)
    if weights.sum() == 0:
        return {}
    mean = float(np.average(lengths, weights=weights))
    order = np.argsort(lengths)
    cum = np.cumsum(weights[order]) / weights.sum()
    median = float(lengths[order][np.searchsorted(cum, 0.5)])
    return {
        "min": float(lengths.min()),
        "max": float(lengths.max()),
        "mean": mean,
        "median": median,
    }
