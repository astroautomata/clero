"""latitude, longitude, and grid-record helpers for climate-analysis outputs."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def latitude_centers(n_lat: int) -> np.ndarray:
    """Cell-centred Gaussian latitudes in degrees for the CLERO/T21 grid."""
    mu, _ = leggauss(n_lat)
    return np.degrees(np.arcsin(mu))


def longitude_centers(n_lon: int) -> np.ndarray:
    """Cell-centred longitudes in degrees, with 0 at the substellar point and antistellar at +/-180."""
    return np.linspace(-180.0 + 180.0 / n_lon, 180.0 - 180.0 / n_lon, n_lon)


def latitude_edges(n_lat: int) -> np.ndarray:
    """Cell *edges* in degrees for the Gaussian latitude grid, tiling [-90, 90].

    Area-conserving boundaries from the quadrature weights (cumulative in sin-latitude),
    consistent with latitude_weights. Pass to pcolormesh so cells reach the poles.
    """
    _, weights = leggauss(n_lat)
    mu = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    return np.degrees(np.arcsin(np.clip(mu, -1.0, 1.0)))


def longitude_edges(n_lon: int) -> np.ndarray:
    """Cell *edges* in degrees, tiling [-180, 180]."""
    return np.linspace(-180.0, 180.0, n_lon + 1)


def latitude_weights(lat: np.ndarray) -> np.ndarray:
    """Normalised latitude weights: Gaussian quadrature on the CLERO grid, cos(lat) otherwise."""
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
    """Flatten a 2D field or 3D variable stack into grid-cell records.

    Args:
        values: (lat, lon) field or (level, lat, lon) variable stack.
        lat: latitudes in degrees; defaults to latitude_centers.
        lon: longitudes in degrees; defaults to longitude_centers.
        levels: level coordinates; defaults to 0..n-1.
        value_name: key for the grid-cell value in each record.

    Returns:
        List of dicts, one per grid cell, suitable for CSV / JSON.
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
