"""Latitude and longitude axes, area weights, and grid-cell export for the 32×64 grid."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def latitude_centers(n_lat: int) -> np.ndarray:
    """Latitude of each row centre in degrees (Gaussian latitudes; CLERO uses `n_lat=32`)."""
    mu, _ = leggauss(n_lat)
    return np.degrees(np.arcsin(mu))


def longitude_centers(n_lon: int) -> np.ndarray:
    """Longitude of each column centre in degrees, with 0 at the substellar point and ±180 at the antistellar point."""
    return np.linspace(-180.0 + 180.0 / n_lon, 180.0 - 180.0 / n_lon, n_lon)


def latitude_edges(n_lat: int) -> np.ndarray:
    """Latitude cell edges in degrees, tiling [-90, 90] (`n_lat + 1` values).

    Area-conserving edges consistent with `latitude_weights`. Pass to `pcolormesh` so the
    plotted cells reach the poles.
    """
    _, weights = leggauss(n_lat)
    mu = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    return np.degrees(np.arcsin(np.clip(mu, -1.0, 1.0)))


def longitude_edges(n_lon: int) -> np.ndarray:
    """Longitude cell edges in degrees, tiling [-180, 180] (`n_lon + 1` values)."""
    return np.linspace(-180.0, 180.0, n_lon + 1)


def latitude_weights(lat: np.ndarray) -> np.ndarray:
    """Area weight of each latitude row, normalised to sum to 1. Exact Gaussian-quadrature weights on the CLERO grid, cos(lat) for any other latitude axis."""
    lat = np.asarray(lat, dtype=float)
    if np.allclose(lat, latitude_centers(lat.size)):
        _, weights = leggauss(lat.size)
    else:
        weights = np.cos(np.deg2rad(lat))
    return weights / weights.sum()


def grid_records(
    values: np.ndarray,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    levels: np.ndarray | None = None,
    value_name: str = "value",
) -> list[dict[str, float]]:
    """Flatten a field or a `(level, lat, lon)` stack into one row per grid cell, for CSV or JSON export.

    Args:
        values: a `(lat, lon)` field or a `(level, lat, lon)` stack.
        lat: latitudes in degrees; defaults to `latitude_centers`.
        lon: longitudes in degrees; defaults to `longitude_centers`.
        levels: level coordinates for a stack; defaults to 0, 1, ....
        value_name: key under which the cell value is stored in each row.

    Returns:
        A list of dicts with keys lat, lon, (level,) and `value_name`.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        lat = _axis(lat, array.shape[0], latitude_centers)
        lon = _axis(lon, array.shape[1], longitude_centers)
        return [
            {"lat": float(lat_idx), "lon": float(lon_idx), value_name: float(array[i, j])}
            for i, lat_idx in enumerate(lat)
            for j, lon_idx in enumerate(lon)
        ]
    if array.ndim == 3:
        levels = _axis(levels, array.shape[0], lambda n: np.arange(n, dtype=float), "levels")
        lat = _axis(lat, array.shape[1], latitude_centers)
        lon = _axis(lon, array.shape[2], longitude_centers)
        return [
            {"level": float(level), "lat": float(lat_idx), "lon": float(lon_idx), value_name: float(array[k, i, j])}
            for k, level in enumerate(levels)
            for i, lat_idx in enumerate(lat)
            for j, lon_idx in enumerate(lon)
        ]
    raise ValueError(f"expected 2D field or 3D variable stack, got shape {array.shape}")


def _axis(
    values: np.ndarray | None,
    size: int,
    default,
    name: str = "axis",
) -> np.ndarray:
    axis = default(size) if values is None else np.asarray(values, dtype=float)
    if axis.shape != (size,):
        raise ValueError(f"expected {name} shape {(size,)}, got {axis.shape}")
    return axis
