from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")

from datasi import compare, inspect
from datasi import visualization as viz

DATA = Path(__file__).resolve().parents[1] / "data"


def test_inspection_plots(tmp_path: Path) -> None:
    report = inspect(DATA / "dataset_missing.csv")
    figs = [
        viz.plot_severity(report),
        viz.plot_missingness(report),
        viz.plot_distribution(report, "amount"),
        viz.plot_distribution(report, "channel"),
        viz.plot_correlation(report, "spearman"),
        viz.plot_temporal(report),
    ]
    for i, fig in enumerate(figs):
        fig.savefig(tmp_path / f"{i}.png")
    assert len(list(tmp_path.glob("*.png"))) == len(figs)
    with pytest.raises(ValueError):
        viz.plot_drift(report)


def test_drift_plot() -> None:
    report = compare(DATA / "dataset_drift_train.csv", DATA / "dataset_drift_test.csv")
    viz.plot_drift(report)
