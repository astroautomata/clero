"""physical diagnostics derived from CLERO climate fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from .grid import latitude_centers, latitude_weights
from .profiles import pressure_levels, stack_levels


def ice_fraction(
    outputs_or_surface_temperature: dict[str, Any] | np.ndarray,
    lat: np.ndarray | None = None,
    *,
    threshold: float = 273.15,
) -> float:
    """Area-weighted surface fraction colder than `threshold` K."""
    surface = _field(outputs_or_surface_temperature, "surface_temperature")
    return float(np.average((surface < threshold).mean(axis=1), weights=_weights(surface, lat)))


def bond_albedo(outputs_or_asr: dict[str, Any] | np.ndarray, f_star: float, lat: np.ndarray | None = None) -> float:
    """Bond albedo from absorbed stellar radiation: 1 - 4 * global_mean(ASR) / F_star."""
    return float(np.clip(1.0 - 4.0 * _area_mean(_field(outputs_or_asr, "asr"), lat) / f_star, 0.0, 1.0))


def net_toa_radiation(outputs_or_asr: dict[str, Any] | np.ndarray, olr: np.ndarray | None = None) -> np.ndarray:
    """Net top-of-atmosphere radiation, positive downward: ASR - OLR."""
    if isinstance(outputs_or_asr, dict):
        return np.asarray(outputs_or_asr["asr"], dtype=float) - np.asarray(outputs_or_asr["olr"], dtype=float)
    if olr is None:
        raise ValueError("olr is required when passing ASR as an array")
    return np.asarray(outputs_or_asr, dtype=float) - np.asarray(olr, dtype=float)


def water_vapor_path(outputs: dict[str, Any] | np.ndarray, *, P0: float, gravity: float) -> np.ndarray:
    """Column water vapor path (kg/m^2) for one planet.

    Hydrostatic column integral (1/g) * integral(q dp) over the model levels, by the
    trapezoidal rule; it spans the sampled levels (surface slab and TOA are not
    extrapolated). `P0` (bar) and `gravity` (m/s^2) are the planet's surface pressure
    and gravity (the same values passed to the emulator).

    Args:
        outputs: a prediction dict (uses its `specific_humidity_*` levels) or a
            `(level, lat, lon)` specific-humidity stack.
        P0: surface pressure in bar.
        gravity: surface gravity in m/s^2.

    Returns:
        A `(lat, lon)` water vapor path field.
    """
    q = outputs if isinstance(outputs, np.ndarray) else stack_levels(outputs, "specific_humidity")
    plev = pressure_levels(P0)
    integral = np.trapezoid(q, x=plev, axis=0)
    if plev[0] > plev[-1]:
        integral = -integral
    return integral / gravity


def _area_mean(field: np.ndarray, lat: np.ndarray | None) -> float:
    array = np.asarray(field, dtype=float)
    return float(np.average(array.mean(axis=1), weights=_weights(array, lat)))


def _field(outputs_or_field: dict[str, Any] | np.ndarray, name: str) -> np.ndarray:
    return np.asarray(outputs_or_field[name] if isinstance(outputs_or_field, dict) else outputs_or_field, dtype=float)


def _weights(field: np.ndarray, lat: np.ndarray | None) -> np.ndarray:
    return latitude_weights(latitude_centers(field.shape[0]) if lat is None else lat)
