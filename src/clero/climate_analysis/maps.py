"""Maps and plots of predicted climate fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._style import _styled
from .grid import grid_records as _grid_records
from .grid import latitude_centers, latitude_edges, longitude_centers, longitude_edges
from .profiles import pressure_levels, stack_levels

# Per-variable colour and colorbar-label conventions.
_FIELD_CMAP = {
    "surface_temperature": "RdYlBu_r",
    "temperature": "RdYlBu_r",
    "specific_humidity": "viridis",
    "asr": "inferno",
    "olr": "inferno",
    "cloud_fraction": "Blues_r",
    "u": "RdBu_r",
    "v": "RdBu_r",
}
_CBAR_LABEL = {
    "surface_temperature": "surface temperature / K",
    "temperature": "temperature / K",
    "specific_humidity": "specific humidity / kg kg$^{-1}$",
    "cloud_fraction": "cloud fraction",
    "asr": "ASR / W m$^{-2}$",
    "olr": "OLR / W m$^{-2}$",
    "u": "eastward wind / m s$^{-1}$",
    "v": "northward wind / m s$^{-1}$",
}


def surface_map(
    outputs_or_field: dict[str, Any] | np.ndarray,
    field: str | None = None,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """A 2D field together with its latitude and longitude axes, as a dict with keys lat, lon, values. Takes a climate dict plus a field name, or a field array."""
    values = _surface_array(outputs_or_field, field)
    return {
        "lat": _axis(lat, values.shape[0], latitude_centers),
        "lon": _axis(lon, values.shape[1], longitude_centers),
        "values": values,
    }


def map_records(
    outputs_or_field: dict[str, Any] | np.ndarray,
    field: str | None = None,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    value_name: str | None = None,
) -> list[dict[str, float]]:
    """One row per grid cell for a 2D field (see `grid_records`). Takes a climate dict plus a field name, or a field array."""
    name = value_name or field or "value"
    return _grid_records(_surface_array(outputs_or_field, field), lat=lat, lon=lon, value_name=name)


def zonal_mean(field: np.ndarray) -> np.ndarray:
    """Mean over longitude: a `(lat, lon)` field gives `(lat,)`, a `(level, lat, lon)` stack gives `(level, lat)`."""
    array = np.asarray(field, dtype=float)
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2D field or 3D variable stack, got shape {array.shape}")
    return array.mean(axis=-1)


def meridional_mean(field: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Area-weighted (cos lat) mean over latitude of a `(lat, lon)` field, giving `(lon,)`."""
    values = _as_surface(field)
    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))
    return np.average(values, axis=0, weights=weights / weights.sum())


@_styled
def field_map(
    outputs_or_field: dict[str, Any] | np.ndarray,
    field: str | None = None,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    ax=None,
    cmap: str | None = None,
    colorbar: bool = True,
    title: str | None = None,
    units: str | None = None,
    **kwargs,
):
    """Plot a 2D field as a latitude–longitude map, substellar point at the centre.

    Args:
        outputs_or_field: a climate dict (then `field` is required) or a `(lat, lon)` array.
        field: name of the field to plot from a climate dict.
        lat, lon: axes in degrees; default to the CLERO grid.
        ax: existing matplotlib axes to draw on; otherwise a new figure is made.
        cmap: matplotlib colormap name; defaults to a per-variable choice.
        colorbar: whether to add a colorbar (default True).
        title: plot title.
        units: colorbar label; defaults to a per-variable label.
        **kwargs: forwarded to `pcolormesh` (e.g. `vmin`, `vmax`, `norm`), useful for
            sharing a colour scale across panels.

    Returns:
        The `(fig, ax)` pair.
    """
    import matplotlib.pyplot as plt

    data = surface_map(outputs_or_field, field, lat=lat, lon=lon)
    # Use cell edges on the default grid so cells tile [-90, 90] x [-180, 180] (no polar gap);
    # fall back to the given centres for any custom axes.
    y = latitude_edges(data["lat"].size) if np.allclose(data["lat"], latitude_centers(data["lat"].size)) else data["lat"]
    x = longitude_edges(data["lon"].size) if np.allclose(data["lon"], longitude_centers(data["lon"].size)) else data["lon"]
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    mesh = ax.pcolormesh(x, y, data["values"], shading="auto", cmap=cmap or _default_cmap(field), **kwargs)
    ax.set(xlim=(-180.0, 180.0), ylim=(-90.0, 90.0), xlabel="longitude", ylabel="latitude")
    ax.set_xticks([-180, -90, 0, 90, 180])
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(mesh, ax=ax, label=units or _CBAR_LABEL.get(_base_field(field), field or "value"))
    return fig, ax


def ice_fraction_map(
    outputs_or_field: dict[str, Any] | np.ndarray,
    *,
    freeze_K: float = 273.15,
    cmap: str = "Blues_r",
    title: str | None = None,
    **kwargs,
):
    """Map of surface ice, where cells colder than `freeze_K` count as ice.

    Blue is open water and white is ice. Takes a climate dict (uses `surface_temperature`)
    or a surface temperature array. Other arguments are passed to `field_map`.
    """
    ts = outputs_or_field["surface_temperature"] if isinstance(outputs_or_field, dict) else outputs_or_field
    ice = (_as_surface(ts) < freeze_K).astype(float)
    return field_map(ice, cmap=cmap, units="ice fraction", title=title, vmin=0.0, vmax=1.0, **kwargs)


def net_radiation_map(
    outputs_or_field: dict[str, Any] | np.ndarray,
    *,
    cmap: str = "RdBu_r",
    title: str | None = None,
    **kwargs,
):
    """Map of net top-of-atmosphere radiation, `ASR - OLR`.

    Positive is net heating (dayside), negative is net cooling (nightside), on a
    diverging colour scale centred at zero. Takes a climate dict (uses `asr` and `olr`)
    or a net-flux array. Other arguments are passed to `field_map`.
    """
    if isinstance(outputs_or_field, dict):
        net = _as_surface(outputs_or_field["asr"]) - _as_surface(outputs_or_field["olr"])
    else:
        net = _as_surface(outputs_or_field)
    vmax = float(np.nanmax(np.abs(net)))
    kwargs.setdefault("vmin", -vmax)
    kwargs.setdefault("vmax", vmax)
    return field_map(net, cmap=cmap, units="ASR $-$ OLR / W m$^{-2}$", title=title, **kwargs)


@_styled
def wind_map(
    outputs: dict[str, Any],
    *,
    level: int = 0,
    scalar: str = "surface_temperature",
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    ax=None,
    step: int = 2,
    cmap: str | None = None,
    colorbar: bool = True,
    title: str | None = None,
):
    """Wind arrows at one model level drawn over a background field.

    Args:
        outputs: a climate dict (uses `u_{level}`, `v_{level}` and the `scalar` field).
        level: model level for the wind, 0 being the lowest (default 0).
        scalar: name of the background field (default "surface_temperature").
        step: draw an arrow every `step` grid cells (default 2).
        lat, lon, ax, cmap, colorbar, title: as in `field_map`.
    """
    import matplotlib.pyplot as plt

    u = np.asarray(outputs[f"u_{level}"], dtype=float)
    v = np.asarray(outputs[f"v_{level}"], dtype=float)
    bg = _as_surface(outputs[scalar])
    latc = latitude_centers(bg.shape[0]) if lat is None else np.asarray(lat, dtype=float)
    lonc = longitude_centers(bg.shape[1]) if lon is None else np.asarray(lon, dtype=float)
    y = latitude_edges(bg.shape[0]) if np.allclose(latc, latitude_centers(bg.shape[0])) else latc
    x = longitude_edges(bg.shape[1]) if np.allclose(lonc, longitude_centers(bg.shape[1])) else lonc
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    mesh = ax.pcolormesh(x, y, bg, shading="auto", cmap=cmap or _default_cmap(scalar))
    q = ax.quiver(lonc[::step], latc[::step], u[::step, ::step], v[::step, ::step])
    ref = int(np.nanpercentile(np.hypot(u, v), 95)) or 1
    ax.quiverkey(q, 0.85, 1.06, ref, f"{ref} m s$^{{-1}}$", labelpos="E",
                 fontproperties={"size": plt.rcParams["xtick.labelsize"]})
    ax.set(xlim=(-180.0, 180.0), ylim=(-90.0, 90.0), xlabel="longitude", ylabel="latitude")
    ax.set_xticks([-180, -90, 0, 90, 180])
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(mesh, ax=ax, label=_CBAR_LABEL.get(_base_field(scalar), scalar))
    return fig, ax


@_styled
def wind_streamlines(
    outputs: dict[str, Any],
    *,
    level: int = 0,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    cmap: str = "plasma",
    density: float = 2.0,
    ax=None,
    colorbar: bool = True,
    title: str | None = None,
):
    """Wind streamlines at one model level, coloured by wind speed.

    The wind is interpolated to evenly spaced latitudes (which streamlines require) and
    wrapped across ±180° longitude. `level`, `ax`, `cmap`, `colorbar` and `title` are as
    in `wind_map`; `density` controls how many streamlines are drawn.
    """
    import matplotlib.pyplot as plt

    u = np.asarray(outputs[f"u_{level}"], dtype=float)
    v = np.asarray(outputs[f"v_{level}"], dtype=float)
    latc = latitude_centers(u.shape[0]) if lat is None else np.asarray(lat, dtype=float)
    lonc = longitude_centers(u.shape[1]) if lon is None else np.asarray(lon, dtype=float)
    lat_u = np.linspace(latc[0], latc[-1], latc.size)
    u, v = _regrid_lat(u, latc, lat_u), _regrid_lat(v, latc, lat_u)
    lonc = np.append(lonc, lonc[0] + 360.0)
    u = np.concatenate([u, u[:, :1]], axis=1)
    v = np.concatenate([v, v[:, :1]], axis=1)
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    strm = ax.streamplot(lonc, lat_u, u, v, color=np.hypot(u, v), cmap=cmap,
                         density=density, linewidth=1.0, arrowsize=0.8)
    ax.set(xlim=(-180.0, 180.0), ylim=(-90.0, 90.0), xlabel="longitude", ylabel="latitude")
    ax.set_xticks([-180, -90, 0, 90, 180])
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(strm.lines, ax=ax, label="wind speed / m s$^{-1}$")
    return fig, ax


@_styled
def zonal_cross_section(
    outputs: dict[str, Any],
    variable: str = "temperature",
    *,
    lat: np.ndarray | None = None,
    P0: float | None = None,
    include_surface: bool = True,
    ax=None,
    cmap: str | None = None,
    units: str | None = None,
    title: str | None = None,
    contour_levels: int = 20,
):
    """Latitude–pressure cross-section of the zonal mean of a multi-level variable.

    Pass `P0` (bar) for a pressure axis; otherwise the vertical axis is the model level
    index. For temperature with `P0` given, the section extends down to the surface using
    `surface_temperature`, unless `include_surface=False`.
    """
    import matplotlib.pyplot as plt

    add_surface = include_surface and variable == "temperature" and P0 is not None and "surface_temperature" in outputs
    stack = stack_levels(outputs, variable, surface_key="surface_temperature" if add_surface else None)
    lat = latitude_centers(stack.shape[1]) if lat is None else np.asarray(lat, dtype=float)
    pressure_axis = P0 is not None
    y = pressure_levels(P0, surface=add_surface) / 100.0 if pressure_axis else np.arange(stack.shape[0], dtype=float)
    fig, ax = (plt.subplots(figsize=(5, 3), constrained_layout=True) if ax is None else (ax.figure, ax))
    cf = ax.contourf(lat, y, stack.mean(axis=2), levels=contour_levels, cmap=cmap or _default_cmap(variable))
    ax.invert_yaxis()
    if pressure_axis:
        ax.set_yscale("log")
    ax.set(xlabel="latitude", ylabel="pressure / hPa" if pressure_axis else "model level")
    fig.colorbar(cf, ax=ax, label=units or _CBAR_LABEL.get(_base_field(variable), variable))
    if title is not None:
        ax.set_title(title)
    return fig, ax


def _regrid_lat(field: np.ndarray, lat_src: np.ndarray, lat_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate a (lat, lon) field from one latitude axis to another."""
    return np.stack([np.interp(lat_dst, lat_src, col) for col in field.T], axis=1)


def _surface_array(outputs_or_field: dict[str, Any] | np.ndarray, field: str | None) -> np.ndarray:
    if isinstance(outputs_or_field, dict):
        if field is None:
            raise ValueError("field is required when passing a prediction dictionary")
        return _as_surface(outputs_or_field[field])
    return _as_surface(outputs_or_field)


def _as_surface(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D surface field, got shape {values.shape}")
    return values


def _base_field(field: str | None) -> str | None:
    if field is None:
        return None
    base, _, suffix = field.rpartition("_")
    return base if suffix.isdigit() else field


def _default_cmap(field: str | None) -> str:
    return _FIELD_CMAP.get(_base_field(field), "viridis")


def _axis(values: np.ndarray | None, size: int, default) -> np.ndarray:
    axis = default(size) if values is None else np.asarray(values, dtype=float)
    if axis.shape != (size,):
        raise ValueError(f"expected axis shape {(size,)}, got {axis.shape}")
    return axis
