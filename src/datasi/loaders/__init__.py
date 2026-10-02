"""Input adapters. Each adapter turns one kind of source into a pandas DataFrame.

Built in: pandas DataFrame, CSV/TSV (optionally compressed), Parquet, JSON / JSON Lines,
Polars DataFrame and PyArrow Table. New sources (e.g. SQL) plug in through
:func:`register_adapter` without touching the rest of the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from datasi.core.dataset import Dataset, SamplingInfo


class LoadError(ValueError):
    pass


@runtime_checkable
class SourceAdapter(Protocol):
    name: str

    def can_load(self, source: Any) -> bool: ...

    def load(self, source: Any, *, nrows: int | None = None, **kwargs: Any) -> pd.DataFrame: ...


def _suffixes(source: Any) -> list[str]:
    if isinstance(source, str | os.PathLike):
        return [s.lower() for s in Path(source).suffixes]
    return []


_COMPRESSION = {".gz", ".bz2", ".zip", ".xz", ".zst"}


def _ext(source: Any) -> str | None:
    sfx = [s for s in _suffixes(source) if s not in _COMPRESSION]
    return sfx[-1] if sfx else None


class DataFrameAdapter:
    name = "pandas"

    def can_load(self, source: Any) -> bool:
        return isinstance(source, pd.DataFrame)

    def load(self, source: Any, *, nrows: int | None = None, **kwargs: Any) -> pd.DataFrame:
        return source if nrows is None else source.head(nrows)


class CSVAdapter:
    name = "csv"

    def can_load(self, source: Any) -> bool:
        return _ext(source) in {".csv", ".tsv", ".txt"}

    def load(self, source: Any, *, nrows: int | None = None, **kwargs: Any) -> pd.DataFrame:
        if _ext(source) == ".tsv":
            kwargs.setdefault("sep", "\t")
        kwargs.setdefault("low_memory", False)
        return pd.read_csv(source, nrows=nrows, **kwargs)


class ParquetAdapter:
    name = "parquet"

    def can_load(self, source: Any) -> bool:
        return _ext(source) in {".parquet", ".pq"}

    def load(self, source: Any, *, nrows: int | None = None, **kwargs: Any) -> pd.DataFrame:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LoadError("Parquet support needs pyarrow: pip install 'datasi[parquet]'") from exc
        if nrows is None:
            return pd.read_parquet(source, **kwargs)
        pf = pq.ParquetFile(source)
        batches = []
        remaining = nrows
        for batch in pf.iter_batches(batch_size=min(65536, nrows)):
            batches.append(batch.slice(0, remaining))
            remaining -= min(remaining, batch.num_rows)
            if remaining == 0:
                break
        import pyarrow as pa

        if not batches:
            return pf.schema_arrow.empty_table().to_pandas()
        return pa.Table.from_batches(batches).to_pandas()


class JSONAdapter:
    name = "json"

    def can_load(self, source: Any) -> bool:
        return _ext(source) in {".json", ".jsonl", ".ndjson"}

    def load(self, source: Any, *, nrows: int | None = None, **kwargs: Any) -> pd.DataFrame:
        lines = _ext(source) in {".jsonl", ".ndjson"}
        if lines:
            return pd.read_json(source, lines=True, nrows=nrows, **kwargs)
        frame = pd.read_json(source, **kwargs)
        return frame if nrows is None else frame.head(nrows)


class PolarsAdapter:
    name = "polars"

    def can_load(self, source: Any) -> bool:
        return type(source).__module__.startswith("polars") and hasattr(source, "to_pandas")

    def load(self, source: Any, *, nrows: int | None = None, **kwargs: Any) -> pd.DataFrame:
        frame = source.head(nrows) if nrows is not None else source
        result: pd.DataFrame = frame.to_pandas()
        return result


class ArrowAdapter:
    name = "arrow"

    def can_load(self, source: Any) -> bool:
        return type(source).__module__.startswith("pyarrow") and hasattr(source, "to_pandas")

    def load(self, source: Any, *, nrows: int | None = None, **kwargs: Any) -> pd.DataFrame:
        table = source.slice(0, nrows) if nrows is not None else source
        result: pd.DataFrame = table.to_pandas()
        return result


_ADAPTERS: list[SourceAdapter] = [
    DataFrameAdapter(),
    PolarsAdapter(),
    ArrowAdapter(),
    CSVAdapter(),
    ParquetAdapter(),
    JSONAdapter(),
]


def register_adapter(adapter: SourceAdapter, *, first: bool = True) -> None:
    """Add a source adapter. ``first=True`` lets it take precedence over built-ins."""
    if first:
        _ADAPTERS.insert(0, adapter)
    else:
        _ADAPTERS.append(adapter)


def adapters() -> list[SourceAdapter]:
    return list(_ADAPTERS)


def load(
    source: Any,
    *,
    name: str | None = None,
    nrows: int | None = None,
    **kwargs: Any,
) -> Dataset:
    """Load a source into a :class:`Dataset`. The source itself is never modified.

    ``nrows`` reads only the first N rows; the dataset then records that it was
    truncated so reports can disclose it.
    """
    if isinstance(source, Dataset):
        return source
    if isinstance(source, str | os.PathLike) and not Path(source).exists():
        raise LoadError(f"file not found: {source}")
    for adapter in _ADAPTERS:
        if adapter.can_load(source):
            try:
                frame = adapter.load(source, nrows=nrows, **kwargs)
            except LoadError:
                raise
            except Exception as exc:
                raise LoadError(f"could not read {source!s} as {adapter.name}: {exc}") from exc
            break
    else:
        raise LoadError(
            f"no adapter for {type(source).__name__} {source!r}; "
            "supported: DataFrame, .csv/.tsv, .parquet, .json/.jsonl, polars, pyarrow"
        )
    is_path = isinstance(source, str | os.PathLike)
    sampling = None
    if nrows is not None and len(frame) >= nrows:
        total = len(source) if isinstance(source, pd.DataFrame) else None
        sampling = SamplingInfo(method="head", rows_read=len(frame), rows_total=total)
    return Dataset(
        frame,
        name=name or (Path(source).name if is_path else "dataframe"),
        source=str(source) if is_path else None,
        sampling=sampling,
        file_format=adapter.name,
    )
