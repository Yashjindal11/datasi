"""Run every example in a temporary directory (used by CI to keep examples working)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    examples = sorted((ROOT / "examples").glob("*.py"))
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for ex in examples:
            print(f"--- {ex.name}")
            result = subprocess.run(
                [sys.executable, str(ex)], cwd=tmp, capture_output=True, text=True
            )
            if result.returncode != 0:
                failed.append(ex.name)
                print(result.stdout[-2000:], result.stderr[-4000:], sep="\n")
            else:
                print("ok")
        check = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasi",
                "config",
                "validate",
                str(ROOT / "examples" / "datasi.yaml"),
            ],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            failed.append("datasi.yaml")
            print(check.stderr)
    if failed:
        print(f"FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
