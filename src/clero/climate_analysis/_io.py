"""small file writers for demo outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_csv(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of dict records to CSV, creating parent dirs if needed. Columns are the sorted union of all keys."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in records for key in row})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    return out
