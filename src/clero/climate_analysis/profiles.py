"""Vertical structure: stacking levels, pressure levels and profiles."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._style import _styled
from .grid import latitude_centers
from .summary_stats import profile_stats

# Vertical coordinate: relative-isobar sigma levels (level 0 = surface, 9 = TOA).
_P_TOP = 1000.0                   # Pa (10 mbar)
_BOTTOM_SQUEEZE_FRACTION = 0.925  # lowest level sits at f_bottom * P0
_SIGMA_LEVELS = (lambda x: 0.75 * x + 1.75 * x**3 - 1.5 * x**4)(np.linspace(1.0, 0.0, 10))

# Variables whose profile extends to the surface, and the field holding that value.
_SURFACE_FIELD = {"temperature": "surface_temperature"}


def vertical_profile(
    outputs_or_variable: dict[str, Any] | np.ndarray,
    variable: str | None = None,
    *,
    lat: np.ndarray,
    profile: str = "global_mean",
) -> np.ndarray:
    """One vertical profile of a multi-level variable.

    Args:
        outputs_or_variable: a climate dict (then `variable` is required) or a
            `(level, lat, lon)` stack.
        variable: variable name when passing a climate dict, e.g. "temperature".
        lat: latitudes in degrees, used for the area-weighted global mean.
        profile: which column: "global_mean", "substellar", "antistellar",
            "east_terminator" or "west_terminator".

    Returns:
        A 1D array with one value per level.
    """
    stats = profile_stats(_profile_array(outputs_or_variable, variable), lat)
    key = profile if profile.endswith("_profile") else f"{profile}_profile"
    return np.asarray(stats[key], dtype=float)


def stack_levels(outputs: dict[str, Any], variable: str, *, surface_key: str | None = None) -> np.ndarray:
    """Assemble the per-level fields `variable_0` to `variable_9` into one `(level, lat, lon)` array.

    If `surface_key` names a field in `outputs` (e.g. "surface_temperature"), it is prepended
    as an extra level below level 0, so that a temperature profile reaches the surface.
    """
    levels = sorted(
        (int(name.rsplit("_", 1)[1]), np.asarray(value, dtype=float))
        for name, value in outputs.items()
        if name.startswith(f"{variable}_") and name.rsplit("_", 1)[1].isdigit()
    )
    if not levels:
        raise KeyError(variable)
    arrays = [value for _, value in levels]
    if surface_key is not None and surface_key in outputs:
        arrays.insert(0, np.asarray(outputs[surface_key], dtype=float))
    return np.stack(arrays, axis=0)


def pressure_levels(P0: float, *, surface: bool = False) -> np.ndarray:
    """Pressure in Pa of the ten model levels for surface pressure `P0` in bar, level 0 (lowest) to 9 (top).

    With `surface=True`, `P0` itself is prepended as an extra bottom level, to pair with a
    stack built using `stack_levels(..., surface_key=...)`.

    The levels are the same relative isobars for every planet:
    `sigma_k = (P_k - P_top) / (f_bottom * P0 - P_top)` with `P_top = 10 mbar` and
    `f_bottom = 0.925`, spaced slightly more finely near the surface and the top.
    """
    P0_pa = P0 * 1.0e5
    plev = _SIGMA_LEVELS * (_BOTTOM_SQUEEZE_FRACTION * P0_pa - _P_TOP) + _P_TOP
    return np.concatenate([[P0_pa], plev]) if surface else plev


@_styled
def plot_profile(
    outputs_or_variable: dict[str, Any] | np.ndarray,
    variable: str | None = None,
    *,
    lat: np.ndarray | None = None,
    levels: np.ndarray | None = None,
    P0: float | None = None,
    ax=None,
    title: str | None = None,
    units: str | None = None,
    log_xscale: bool = False,
    include_surface: bool = True,
    **kwargs,
):
    """Plot the global-mean vertical profile of a multi-level variable.

    Args:
        outputs_or_variable: a climate dict (then `variable` is required) or a
            `(level, lat, lon)` stack.
        variable: variable name when passing a climate dict, e.g. "temperature".
        lat: latitudes in degrees; defaults to the CLERO grid.
        levels: vertical coordinate values. If omitted and `P0` is given, `pressure_levels(P0)`.
        P0: surface pressure in bar, giving a pressure axis in hPa.
        ax: existing matplotlib axes to draw on; otherwise a new figure is made.
        title: plot title.
        units: x-axis label.
        log_xscale: use a logarithmic x-axis (useful for humidity).
        include_surface: for temperature with `P0` given, extend the profile down to the
            surface using `surface_temperature` (default True).
        **kwargs: forwarded to `plot`.

    Returns:
        The `(fig, ax)` pair.
    """
    import matplotlib.pyplot as plt

    add_surface = (
        include_surface
        and P0 is not None
        and isinstance(outputs_or_variable, dict)
        and _SURFACE_FIELD.get(variable) in outputs_or_variable
    )
    if add_surface:
        values = stack_levels(outputs_or_variable, variable, surface_key=_SURFACE_FIELD[variable])
        levels = pressure_levels(P0, surface=True)
        P0 = None  # levels now carries the surface row
    else:
        values = _profile_array(outputs_or_variable, variable)
    lat = latitude_centers(values.shape[1]) if lat is None else np.asarray(lat, dtype=float)
    x = vertical_profile(values, lat=lat, profile="global_mean")
    y, ylabel, pressure_axis = _profile_axis(x.size, levels, P0)
    fig, ax = (plt.subplots(figsize=(4, 4), constrained_layout=True) if ax is None else (ax.figure, ax))
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linewidth", 2.0)
    ax.plot(x, y, **kwargs)
    ax.set(xlabel=units or variable or "value", ylabel=ylabel)
    if pressure_axis:
        ax.set_yscale("log")
        ax.invert_yaxis()
    if log_xscale:
        ax.set_xscale("log")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    return fig, ax


def profile_table(
    outputs: dict[str, Any],
    *,
    lat: np.ndarray,
    variables: list[str] | None = None,
    levels: np.ndarray | None = None,
    profiles: tuple[str, ...] = ("global_mean", "substellar", "antistellar"),
) -> list[dict[str, float | str]]:
    """One row per (variable, profile, level) with its value, for the requested multi-level variables and profile types (see `vertical_profile`)."""
    rows: list[dict[str, float | str]] = []
    for variable in variables or list(outputs):
        try:
            values = _profile_array(outputs, variable)
        except (KeyError, ValueError):
            if variables is not None:
                raise
            continue
        if values.ndim != 3:
            continue
        level_values = np.arange(values.shape[0], dtype=float) if levels is None else np.asarray(levels, dtype=float)
        if level_values.shape != (values.shape[0],):
            raise ValueError(f"expected levels shape {(values.shape[0],)}, got {level_values.shape}")
        for name in profiles:
            profile_values = vertical_profile(values, lat=lat, profile=name)
            rows.extend(
                {"variable": variable, "profile": name, "level": float(level), "value": float(value)}
                for level, value in zip(level_values, profile_values)
            )
    return rows


def _profile_axis(size: int, levels: np.ndarray | None, P0: float | None) -> tuple[np.ndarray, str, bool]:
    if levels is None and P0 is not None:
        levels = pressure_levels(P0)
    if levels is None:
        return np.arange(size, dtype=float), "level", False
    y = np.asarray(levels, dtype=float)
    if y.shape != (size,):
        raise ValueError(f"expected levels shape {(size,)}, got {y.shape}")
    return y / 100.0, "pressure / hPa", True


def _profile_array(outputs_or_variable: dict[str, Any] | np.ndarray, variable: str | None) -> np.ndarray:
    if isinstance(outputs_or_variable, dict):
        if variable is None:
            raise ValueError("variable is required when passing a prediction dictionary")
        values = np.asarray(
            outputs_or_variable[variable] if variable in outputs_or_variable else stack_levels(outputs_or_variable, variable),
            dtype=float,
        )
    else:
        values = np.asarray(outputs_or_variable, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"expected a 3D variable stack, got shape {values.shape}")
    return values
