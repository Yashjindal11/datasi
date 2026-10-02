"""Quickstart: investigate a DataFrame and read the findings.

Run: python examples/quickstart.py
"""

from datasi import inspect
from datasi.synthetic import generate_dataset

# A synthetic orders table with a few planted problems (see datasi.synthetic).
data = generate_dataset(
    5000, missingness=0.12, duplicates=0.02, outliers=0.01, category_variants=0.02, seed=7
)
df = data.frame

report = inspect(df, target="churned", config={"reference_time": "2026-10-01"})

print(report.summary)
print()

# Every finding separates what was measured from what it might mean.
for finding in report.filter(min_severity="warning")[:5]:
    print(finding.explain())
    print()

# Structured access for automation.
high = report.filter(min_severity="high")
print(
    f"{len(high)} high-or-critical findings; columns affected: {sorted({c for f in high for c in f.columns})}"
)

report.save("datasi-report-quickstart.html")
print("wrote datasi-report-quickstart.html (open it in a browser; works offline)")
