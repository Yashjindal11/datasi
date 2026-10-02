import json
import uuid

import numpy as np
import pandas as pd
import pytest

from datasi.core.dataset import Dataset
from datasi.loaders import LoadError, load
from datasi.schema import ColumnType, StorageHint, infer_schema, name_suggests_identifier
from datasi.statistics import strings as st


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "uuid": [str(uuid.UUID(int=int(i) + 1)) for i in range(n)],
            "amount": rng.normal(100, 10, n),
            "amount_text": [f"${x:,.2f}" for x in rng.normal(1000, 100, n)],
            "rate": [f"{x:.1f}%" for x in rng.uniform(0, 100, n)],
            "count_str": [str(i % 17) for i in range(n)],
            "flag": rng.choice(["yes", "no"], n),
            "flag01": rng.integers(0, 2, n),
            "city": rng.choice(["Chicago", "Denver", "Houston"], n),
            "when": pd.date_range("2026-01-01", periods=n, freq="D"),
            "when_text": [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-01-01", periods=n)],
            "comment": [f"customer said the delivery number {i} was late again" for i in range(n)],
            "const": ["x"] * n,
            "empty": [None] * n,
        }
    )


def test_schema_inference_types(mixed_frame: pd.DataFrame) -> None:
    schema = infer_schema(mixed_frame)
    t = {c: schema[c].column_type for c in mixed_frame.columns}
    assert t["customer_id"] == ColumnType.IDENTIFIER
    assert t["uuid"] == ColumnType.IDENTIFIER
    assert t["amount"] == ColumnType.NUMERIC
    assert t["amount_text"] == ColumnType.NUMERIC
    assert StorageHint.CURRENCY_AS_STRING in schema["amount_text"].hints
    assert StorageHint.PERCENT_AS_STRING in schema["rate"].hints
    assert StorageHint.NUMERIC_AS_STRING in schema["count_str"].hints
    assert t["flag"] == ColumnType.BOOLEAN
    assert t["flag01"] == ColumnType.BOOLEAN
    assert t["city"] == ColumnType.CATEGORICAL
    assert t["when"] == ColumnType.DATETIME
    assert t["when_text"] == ColumnType.DATETIME
    assert StorageHint.DATETIME_AS_STRING in schema["when_text"].hints
    assert t["comment"] == ColumnType.TEXT
    assert t["const"] == ColumnType.CONSTANT
    assert t["empty"] == ColumnType.UNKNOWN
    assert schema["empty"].missing_ratio == 1.0
    assert schema["customer_id"].reasons  # explains why


def test_repeated_entity_key_is_identifier() -> None:
    frame = pd.DataFrame({"account_id": np.repeat(np.arange(50), 4), "v": np.arange(200)})
    assert infer_schema(frame)["account_id"].column_type == ColumnType.IDENTIFIER


def test_unique_measurement_is_not_identifier() -> None:
    frame = pd.DataFrame({"price": np.random.default_rng(1).normal(50, 5, 500)})
    assert infer_schema(frame)["price"].column_type == ColumnType.NUMERIC


def test_forced_identifier() -> None:
    frame = pd.DataFrame({"sku": ["a", "b", "a"]})
    assert infer_schema(frame, id_columns=["sku"])["sku"].column_type == ColumnType.IDENTIFIER


def test_mixed_python_types_hint() -> None:
    frame = pd.DataFrame({"m": pd.Series(["a", 1, "b", 2.5, "c"] * 10, dtype=object)})
    assert StorageHint.MIXED_TYPES in infer_schema(frame)["m"].hints


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("customer_id", True),
        ("customerId", True),
        ("uuid", True),
        ("record_number", True),
        ("id", True),
        ("idea", False),
        ("valid", False),
        ("paid", False),
        ("amount", False),
    ],
)
def test_identifier_names(name: str, expected: bool) -> None:
    assert name_suggests_identifier(name) is expected


def test_string_helpers() -> None:
    counts = pd.Series({"2026-01-01": 5, "01/02/2026": 3, "2026/03/01": 2, "x": 1})
    shapes = st.date_shapes(counts)
    assert shapes["ISO date (YYYY-MM-DD)"] == 5
    assert shapes["MM/DD/YYYY or DD/MM/YYYY"] == 3
    assert shapes["YYYY/MM/DD"] == 2
    assert st.datetime_parse(counts) == pytest.approx(10 / 11)
    assert st.datetime_parse(pd.Series({"12": 3, "13": 4})) == 0.0
    assert st.numeric_parse(pd.Series({"1,234.5": 1, "12": 1})).kind == "mixed"
    parsed = st.to_number(pd.Series(["$1,200.50", "(3)", "45%", "abc"]))
    assert parsed.iloc[0] == 1200.5
    assert parsed.iloc[1] == -3
    assert parsed.iloc[2] == pytest.approx(0.45)
    assert np.isnan(parsed.iloc[3])
    assert st.boolean_like(pd.Series({"Yes": 1, "no": 2}))
    assert not st.boolean_like(pd.Series({"0": 1, "1": 2}))
    assert st.unusual_characters("a\u200bb\x00") == ["U+200B", "U+0000"]
    assert st.compact_shape("AB-0012") == "A+-9+"
    assert st.normalize_label(" United-Airlines ") == st.normalize_label("united airlines")


def test_dataset_does_not_modify_source() -> None:
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"], index=[10, 20])
    before = frame.copy()
    ds = Dataset(frame)
    assert ds.columns == ["a", "a__2"]
    assert ds.duplicate_column_names == ["a"]
    pd.testing.assert_frame_equal(frame, before)
    assert list(frame.columns) == ["a", "a"]


def test_load_formats(tmp_path, mixed_frame: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    small = mixed_frame[["customer_id", "amount", "city"]]
    small.to_csv(tmp_path / "d.csv", index=False)
    small.to_csv(tmp_path / "d.tsv", sep="\t", index=False)
    small.to_csv(tmp_path / "d.csv.gz", index=False)
    small.to_parquet(tmp_path / "d.parquet")
    small.to_json(tmp_path / "d.jsonl", orient="records", lines=True)
    (tmp_path / "d.json").write_text(json.dumps(small.to_dict(orient="records")))
    for f in ["d.csv", "d.tsv", "d.csv.gz", "d.parquet", "d.jsonl", "d.json"]:
        ds = load(tmp_path / f)
        assert ds.n_rows == 200, f
        assert ds.columns == ["customer_id", "amount", "city"], f
    head = load(tmp_path / "d.parquet", nrows=50)
    assert head.n_rows == 50
    assert head.sampling is not None
    assert head.sampling.method == "head"


def test_load_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(LoadError, match="not found"):
        load(tmp_path / "missing.csv")
    (tmp_path / "x.xlsx").write_text("nope")
    with pytest.raises(LoadError, match="no adapter"):
        load(tmp_path / "x.xlsx")


def test_load_arrow_table() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": [1, 2, 3]})
    assert load(table).n_rows == 3
