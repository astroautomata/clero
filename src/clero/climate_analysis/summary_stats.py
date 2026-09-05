"""small summary stats for predicted climate fields and variable stacks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .grid import latitude_weights


def summarize_outputs(
    outputs: dict[str, Any],
    lat: np.ndarray,
) -> dict[str, dict[str, float | np.ndarray]]:
    """Per-output stats: min/max, plus map means for 2D fields and profile stats for 3D variables."""
    summary: dict[str, dict[str, float | np.ndarray]] = {}
    for name, value in outputs.items():
        array = np.asarray(value, dtype=float)
        stats: dict[str, float | np.ndarray] = {
            "min": float(array.min()),
            "max": float(array.max()),
        }
        if array.ndim == 2:
            stats.update({
                "dayside_mean": dayside_mean(array, lat),
                "nightside_mean": nightside_mean(array, lat),
                "global_mean": global_mean(array, lat),
            })
        if array.ndim == 3:
            stats.update(profile_stats(array, lat))
        summary[name] = stats
    return summary


def global_mean(field: np.ndarray, lat: np.ndarray) -> float:
    """Area-weighted global mean of a 2D (lat, lon) field."""
    array = np.asarray(field, dtype=float)
    return float(np.average(array.mean(axis=1), weights=latitude_weights(lat)))


def dayside_mean(field: np.ndarray, lat: np.ndarray) -> float:
    """Area-weighted mean over the dayside longitudes of a 2D (lat, lon) field."""
    array = np.asarray(field, dtype=float)
    q1, q3 = array.shape[1] // 4, array.shape[1] * 3 // 4
    if q1 == q3:
        return global_mean(array, lat)
    return global_mean(array[:, q1:q3], lat)


def nightside_mean(field: np.ndarray, lat: np.ndarray) -> float:
    """Area-weighted mean over the nightside longitudes of a 2D (lat, lon) field."""
    array = np.asarray(field, dtype=float)
    q1, q3 = array.shape[1] // 4, array.shape[1] * 3 // 4
    if q1 == q3:
        return global_mean(array, lat)
    return global_mean(np.concatenate([array[:, :q1], array[:, q3:]], axis=1), lat)


def profile_stats(variable: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Substellar, antistellar, both terminator, and global-mean profiles for a 3D variable stack."""
    array = np.asarray(variable, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"expected (level, lat, lon), got shape {array.shape}")
    lat_idx = array.shape[1] // 2
    mean_over_lon = array.mean(axis=2)
    return {
        "substellar_profile": _equatorial_profile(array, lat_idx, array.shape[2] // 2),
        "antistellar_profile": _equatorial_profile(array, lat_idx, 0),
        "west_terminator_profile": _equatorial_profile(array, lat_idx, array.shape[2] // 4),
        "east_terminator_profile": _equatorial_profile(array, lat_idx, array.shape[2] * 3 // 4),
        "global_mean_profile": np.average(mean_over_lon, axis=1, weights=latitude_weights(lat)),
    }


def _equatorial_profile(variable: np.ndarray, lat_idx: int, lon_center: int) -> np.ndarray:
    return np.take(variable[:, lat_idx], _longitude_indices(variable.shape[2], lon_center), axis=1).mean(axis=1)


def _longitude_indices(n_lon: int, lon_center: int) -> np.ndarray:
    if n_lon % 2:
        return np.array([lon_center])
    return np.array([(lon_center - 1) % n_lon, lon_center % n_lon])
