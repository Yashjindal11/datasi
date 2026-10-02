"""Compare a training set with a test set and list distribution shifts.

Run: python examples/train_test_drift.py
"""

from datasi import inspect_train_test
from datasi.synthetic import make_drift_pair

train, test, _truth = make_drift_pair(5000, shift=0.5, category_shift=0.2, seed=1)

report = inspect_train_test(train, test, target="churned")
print(report.summary)
print()
for f in report.findings:
    print(f"[{f.severity.value}] {f.title}: {f.observation}")

drift = report.sections["drift"]
print(f"\nDomain classifier AUC: {drift['domain_classifier_auc']:.3f} (0.5 = indistinguishable)")
report.save("datasi-report-drift.html")
