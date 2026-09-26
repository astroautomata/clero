"""Climate states: the six-way classification of a climate, and the runaway probability over draws."""

from __future__ import annotations

from typing import Any

import numpy as np

from .grid import latitude_centers, latitude_weights
from .profiles import pressure_levels, stack_levels

CLIMATE_STATES = ("snowball", "eyeball", "waterbelt", "globally_temperate", "moist_greenhouse", "runaway_greenhouse")
"""The six climate states, cold to hot, as returned by `climate_state`."""

_MOLAR_MASS = {"H2O": 18.01528, "N2": 28.0134, "CO2": 44.0095, "CH4": 16.043}  # g/mol
_REQUIRED_FIELDS = ("surface_temperature", *(f"specific_humidity_{k}" for k in range(10)))


def climate_state(
    outputs: dict[str, Any],
    inputs: dict[str, Any] | list[dict[str, float]],
    *,
    column_water_threshold: float = 0.30,
    surface_temperature_threshold: float = 400.0,
    h2o_vmr_threshold: float = 1.0e-3,
    ice_threshold: float = 273.15,
) -> dict[str, Any]:
    """Classify a climate into one of `CLIMATE_STATES`.

    The rules are applied in order. Runaway greenhouse: global-mean surface temperature above
    `surface_temperature_threshold` K, or column water mass fraction (the pressure-weighted mean
    of the global-mean specific humidity over the ten model levels) above `column_water_threshold`.
    Moist greenhouse: area-mean water volume mixing ratio on the top level (10 mbar) at or above
    `h2o_vmr_threshold`. The rest by ice cover, a cell being frozen at or below `ice_threshold` K:
    snowball (every cell frozen), globally temperate (no cell frozen), waterbelt (frozen cells
    remain but at least one latitude row is entirely ice-free), otherwise eyeball.

    Args:
        outputs: a climate dict in physical units from `Emulator.predict` or `Emulator.sample`.
            Fields may carry leading axes (draws, planets), which the results keep.
        inputs: the planet dict or batch the climate was predicted for; its P0, CO2 and CH4 are used.
        column_water_threshold, surface_temperature_threshold, h2o_vmr_threshold, ice_threshold:
            the thresholds above (defaults are the paper's).

    Returns:
        A dict with `state` (strings from `CLIMATE_STATES`), `column_water`, `surface_temperature`
        (the global mean), `h2o_vmr_top` and `ice_fraction`, each with the leading axes of `outputs`.
    """
    missing = [name for name in _REQUIRED_FIELDS if name not in outputs]
    if missing:
        raise KeyError(f"climate_state needs the fields {list(_REQUIRED_FIELDS)}; missing {missing}")
    ts = np.asarray(outputs["surface_temperature"], dtype=float)
    q = stack_levels(outputs, "specific_humidity")  # (10, ..., lat, lon)
    if ts.min() <= 0.0 or q.min() < 0.0 or q.max() > 1.0:
        raise ValueError("expected physical-space fields (K and kg/kg): use space='physical' or Emulator.to_physical")
    P0, CO2, CH4 = (_planet_values(inputs, name) for name in ("P0", "CO2", "CH4"))
    if np.ndim(P0) == 1 and (ts.ndim < 3 or ts.shape[-3] != np.size(P0)):
        raise ValueError(f"a batch of {np.size(P0)} planets needs fields shaped (..., {np.size(P0)}, lat, lon), got {ts.shape}")

    surface_temperature = _area_mean(ts)
    q_mean = np.moveaxis(_area_mean(q), 0, -1)  # (..., 10)
    column_water = (q_mean * _column_weights(P0)).sum(axis=-1)
    m_dry = _MOLAR_MASS["N2"] * (1.0 - CO2 - CH4) + _MOLAR_MASS["CO2"] * CO2 + _MOLAR_MASS["CH4"] * CH4
    water = q[-1] / _MOLAR_MASS["H2O"]
    dry = (1.0 - q[-1]) / np.reshape(m_dry, np.shape(m_dry) + (1, 1))
    h2o_vmr_top = _area_mean(water / (water + dry))
    warm = ts > ice_threshold
    ice_fraction = _area_mean(~warm)

    runaway = (column_water > column_water_threshold) | (surface_temperature > surface_temperature_threshold)
    moist = ~runaway & (h2o_vmr_top >= h2o_vmr_threshold)
    rest = ~runaway & ~moist
    state = np.select(
        [runaway, moist, rest & ~warm.any(axis=(-2, -1)), rest & warm.all(axis=(-2, -1)), rest & warm.all(axis=-1).any(axis=-1)],
        ["runaway_greenhouse", "moist_greenhouse", "snowball", "globally_temperate", "waterbelt"],
        default="eyeball",
    )
    result = {
        "state": state,
        "column_water": column_water,
        "surface_temperature": surface_temperature,
        "h2o_vmr_top": h2o_vmr_top,
        "ice_fraction": ice_fraction,
    }
    return {name: value.item() if value.ndim == 0 else value for name, value in result.items()}


def runaway_probability(
    samples: dict[str, Any],
    inputs: dict[str, Any] | list[dict[str, float]],
    *,
    column_water_threshold: float = 0.30,
    surface_temperature_threshold: float = 400.0,
    h2o_vmr_threshold: float = 1.0e-3,
    ice_threshold: float = 273.15,
) -> float | np.ndarray:
    """Fraction of predictive draws in the runaway greenhouse state.

    `samples` is the dict returned by `Emulator.sample`: the first axis of every field holds the
    draws for one planet, so fields are `(n_samples, 32, 64)`, or `(n_samples, n_planets, 32, 64)`
    for a batch. Thresholds are as in `climate_state`.

    Returns:
        A float, or one float per planet for a batch.
    """
    if np.ndim(samples["surface_temperature"]) < 3:
        raise ValueError("samples must have a leading draw axis, as returned by Emulator.sample")
    state = climate_state(
        samples,
        inputs,
        column_water_threshold=column_water_threshold,
        surface_temperature_threshold=surface_temperature_threshold,
        h2o_vmr_threshold=h2o_vmr_threshold,
        ice_threshold=ice_threshold,
    )["state"]
    probability = np.mean(np.asarray(state) == "runaway_greenhouse", axis=0)
    return float(probability) if probability.ndim == 0 else probability


def _area_mean(field: np.ndarray) -> np.ndarray:
    """Area-weighted mean over the trailing (lat, lon) axes."""
    array = np.asarray(field, dtype=float)
    return (array.mean(axis=-1) * latitude_weights(latitude_centers(array.shape[-2]))).sum(axis=-1)


def _column_weights(P0: float | np.ndarray) -> np.ndarray:
    """Trapezoid weights of the ten levels in pressure, summing to 1; one row per planet for a batch."""
    P0 = np.asarray(P0, dtype=float)
    plev = np.stack([pressure_levels(float(p)) for p in P0.reshape(-1)]).reshape(P0.shape + (-1,))
    half = 0.5 * np.abs(np.diff(plev, axis=-1))
    weights = np.zeros_like(plev)
    weights[..., :-1] += half
    weights[..., 1:] += half
    return weights / weights.sum(axis=-1, keepdims=True)


def _planet_values(inputs: dict[str, Any] | list[dict[str, float]], name: str) -> float | np.ndarray:
    """One input as a float for a single planet or a `(n_planets,)` array for a batch."""
    if isinstance(inputs, (list, tuple)):
        return np.asarray([row[name] for row in inputs], dtype=float)
    if isinstance(inputs, dict):
        value = np.asarray(inputs[name], dtype=float)
        return float(value) if value.ndim == 0 else value
    raise TypeError("inputs must be a planet dict, a list of planet dicts, or a column dict")


__all__ = ["CLIMATE_STATES", "climate_state", "runaway_probability"]
