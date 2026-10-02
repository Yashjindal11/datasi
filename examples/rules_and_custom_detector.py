"""Add domain knowledge (rules) and a custom detector.

Statistical detectors can only say a value is *unusual*. Rules let DataSI say a value
is *invalid*. Custom detectors plug into the same pipeline and report format.

Run: python examples/rules_and_custom_detector.py
"""

import numpy as np
import pandas as pd

from datasi import Severity, investigate
from datasi.detectors import BaseDetector, Context
from datasi.findings import Category, Finding

rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame(
    {
        "flight_id": [f"UA{1000 + i}" for i in range(n)],
        "dep_delay_min": rng.gamma(1.5, 12, n).round(),
        "arr_delay_min": rng.gamma(1.5, 12, n).round(),
        "seats_sold": rng.integers(80, 180, n),
        "capacity": np.full(n, 180),
        "status": rng.choice(["ON_TIME", "DELAYED", "CANCELLED"], n, p=[0.7, 0.27, 0.03]),
    }
)
df.loc[[10, 20], "seats_sold"] = [200, 215]  # oversold beyond capacity
df.loc[[5], "status"] = "LANDED?"


class SeatsWithinCapacity(BaseDetector):
    name = "seats_within_capacity"
    category = Category.NUMERIC
    description = "Seats sold must not exceed aircraft capacity."

    def analyze(self, ctx: Context) -> list[Finding]:
        bad = ctx.numeric("seats_sold") > ctx.numeric("capacity")
        if not bad.any():
            return []
        return [
            self.finding(
                "custom.oversold",
                Severity.HIGH,
                "Seats sold exceed capacity",
                f"{int(bad.sum())} flights sold more seats than the aircraft has.",
                columns=["seats_sold", "capacity"],
                evidence={"rows": int(bad.sum()), "row_examples": ctx.row_examples(bad)},
                suggestion="Check overbooking records or capacity data for these flights.",
                confidence=1.0,
            )
        ]


report = investigate(
    df,
    config={
        "rules": {
            "dep_delay_min": {"min": -60, "max": 1440},
            "status": {"allowed": ["ON_TIME", "DELAYED", "CANCELLED"]},
            "flight_id": {"pattern": r"UA\d{4}", "unique": True},
        }
    },
    extra_detectors=[SeatsWithinCapacity()],
)

for f in report.filter(min_severity="high"):
    print(f"[{f.severity.value}] {f.title} - {f.observation}")
