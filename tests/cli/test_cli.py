from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasi.cli.main import main

DATA = Path(__file__).resolve().parents[1] / "data"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "datasi" in capsys.readouterr().out


def test_inspect_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    html, md, js = tmp_path / "r.html", tmp_path / "r.md", tmp_path / "r.json"
    code = main(
        [
            "inspect",
            str(DATA / "dataset_duplicates.csv"),
            "-o",
            str(html),
            "-o",
            str(md),
            "-o",
            str(js),
            "--no-color",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Exact duplicate rows" in out
    assert html.read_text().startswith("<!doctype html>")
    assert "# DataSI Investigation Report" in md.read_text()
    assert json.loads(js.read_text())["summary"]["findings"] > 0


def test_fail_on(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(DATA / "dataset_duplicates.csv"), "--fail-on", "warning"]) == 1
    assert main(["inspect", str(DATA / "dataset_duplicates.csv"), "--fail-on", "critical"]) == 0


def test_inspect_json_and_detector_selection(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["inspect", str(DATA / "dataset_missing.csv"), "--json", "--detectors", "missingness"])
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert {f["detector"] for f in data["findings"]} == {"missingness"}
    assert [d["name"] for d in data["detectors"]] == ["missingness"]


def test_report_default_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["report", str(DATA / "dataset_types.csv")]) == 0
    assert (tmp_path / "datasi-report-dataset_types.html").is_file()


def test_compare(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "compare",
            str(DATA / "dataset_drift_train.csv"),
            str(DATA / "dataset_drift_test.csv"),
            "--no-color",
        ]
    )
    assert code == 0
    assert "Distribution shift in 'amount'" in capsys.readouterr().out


def test_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", str(DATA / "dataset_types.csv")]) == 0
    out = capsys.readouterr().out
    assert "currency_as_string" in out
    assert main(["schema", str(DATA / "dataset_types.csv"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 1200
    assert main(["schema", str(DATA / "dataset_types.csv"), "--chunksize", "100"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 1200


def test_config_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "c.yaml"
    assert main(["config", "init", "-o", str(path)]) == 0
    assert main(["config", "validate", str(path)]) == 0
    assert "valid" in capsys.readouterr().out
    path.write_text("detectors:\n  nope: true\n")
    assert main(["config", "validate", str(path)]) == 2
    assert "unknown detector" in capsys.readouterr().err


def test_detectors_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["detectors"]) == 0
    assert "missingness" in capsys.readouterr().out


def test_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(tmp_path / "nope.csv")]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["inspect", str(DATA / "dataset_types.csv"), "--detectors", "bogus"]) == 2
    assert main(["inspect", str(DATA / "dataset_types.csv"), "-o", str(tmp_path / "x.pdf")]) == 2


def test_disable_and_options(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "inspect",
                str(DATA / "dataset_outliers.csv"),
                "--json",
                "--disable",
                "outliers,correlation",
                "--outlier-methods",
                "iqr",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    names = {d["name"] for d in data["detectors"]}
    assert "outliers" not in names and "correlation" not in names
