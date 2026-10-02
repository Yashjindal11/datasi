# Configuration

Configuration can be passed as a `Config` object, a dict or a YAML file path:

```python
from datasi import inspect

report = inspect("orders.parquet", config="datasi.yaml")
report = inspect(df, config={"thresholds": {"missing_warning": 0.05}}, target="churned")
```

Generate the full default file with `datasi config init -o datasi.yaml` and validate
edits with `datasi config validate datasi.yaml`.

```yaml
detectors:            # every detector is on unless set to false
  outliers: true
  temporal: true
  correlation: false

thresholds:           # see thresholds.md
  missing_warning: 0.1
  missing_high: 0.5
  correlation_warning: 0.9

outlier_methods: [iqr, modified_zscore, isolation_forest]

target: churned       # enables target analysis
time_column: order_ts # otherwise auto-detected
id_columns: [sku]     # force identifier typing

rules:                # domain knowledge -> confirmed-invalid findings
  age: {min: 0, max: 120, nullable: false}
  status: {allowed: [active, closed, suspended]}
  order_ref: {pattern: "ORD-\\d{8}", unique: true}

privacy:
  include_examples: false   # raw values never appear in reports unless true
  max_examples: 5
  redact_labels: false      # hash category labels in reports

performance:
  sample_rows: 200000       # cap for pairwise/multivariate analyses
  max_pairwise_columns: 60
  max_group_levels: 50
  random_state: 0

reference_time: "2026-10-01T00:00:00"  # 'now' for future-date checks
```

Validation is strict: unknown keys or detector names, values outside their range and
inconsistent pairs (e.g. `ks_warning > ks_high`) raise a readable `ConfigError`.
