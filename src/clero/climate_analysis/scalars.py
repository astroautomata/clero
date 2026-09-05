"""table-friendly scalar summaries for predicted fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._io import write_csv
from .grid import latitude_weights
from .summary_stats import dayside_mean, global_mean, nightside_mean, summarize_outputs


def summary_table(outputs: dict[str, Any], lat: np.ndarray) -> list[dict[str, float | str]]:
    """Scalar-only summary rows (name, metric, value) over a prediction dict."""
    rows: list[dict[str, float | str]] = []
    for name, stats in summarize_outputs(outputs, lat).items():
        rows.extend(
            {"name": name, "metric": metric, "value": float(value)}
            for metric, value in stats.items()
            if np.asarray(value).ndim == 0
        )
    return rows


__all__ = [
    "dayside_mean",
    "global_mean",
    "latitude_weights",
    "nightside_mean",
    "summarize_outputs",
    "summary_table",
    "write_csv",
]
