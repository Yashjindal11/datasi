"""Measure detector precision/recall on synthetic data with known problems.

Run: python examples/evaluate_detectors.py
"""

from datasi import inspect
from datasi.evaluation import evaluate_findings, summarize
from datasi.synthetic import generate_dataset

trials = []
for seed in range(5):
    data = generate_dataset(
        3000,
        missingness=0.1,
        duplicates=0.02,
        outliers=0.01,
        category_variants=0.01,
        sentinels=0.005,
        redundant_columns=True,
        seed=seed,
    )
    report = inspect(data.frame, config={"reference_time": "2026-10-01"})
    trials.append(evaluate_findings(report.findings, data.truth))

table = summarize(trials)
print(table[["issue", "tp", "fp", "fn", "precision", "recall", "fpr"]].to_string(index=False))
