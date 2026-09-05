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


def spectral_synthesis(field: np.ndarray, *, n_lat: int = 256, n_lon: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a CLERO field on a fine grid by exact spherical-harmonic synthesis.

    CLERO fields are band-limited (T21 on the 32×64 Gaussian grid): the grid values are
    samples of a smooth function, and this recovers that function rather than drawing
    the samples as cells. Longitude is handled by FFT and, for each zonal wavenumber,
    associated-Legendre coefficients are fitted on the Gaussian latitudes (exactly
    determined for a band-limited field) and evaluated on `n_lat` equal-area latitudes
    and `n_lon` longitudes tiling the sphere. No information is added or lost.

    Returns `(values, lat, lon)` with `values` of shape `(n_lat, n_lon)` and cell-centre
    axes in degrees, substellar point at longitude 0.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.special import lpmv

    values = _as_surface(field)
    src_lat, src_lon = values.shape
    lmax = min((2 * src_lat - 1) // 3, (src_lon - 1) // 3)  # T21 for the 32×64 CLERO grid
    mu_src, _ = leggauss(src_lat)
    mu_dst = _equal_area_mu(n_lat)
    spec = np.fft.rfft(values, axis=1) / src_lon
    shift = np.deg2rad(longitude_centers(n_lon)[0] - longitude_centers(src_lon)[0])  # re-anchor column 0 to the fine grid
    out = np.zeros((n_lat, n_lon // 2 + 1), dtype=complex)
    for m in range(lmax + 1):
        degrees = np.arange(m, lmax + 1)
        coeffs = np.linalg.lstsq(np.stack([lpmv(m, l, mu_src) for l in degrees], axis=1), spec[:, m], rcond=None)[0]
        out[:, m] = np.stack([lpmv(m, l, mu_dst) for l in degrees], axis=1) @ coeffs * np.exp(1j * m * shift)
    fine = np.fft.irfft(out, n=n_lon, axis=1) * n_lon
    return fine, np.degrees(np.arcsin(mu_dst)), longitude_centers(n_lon)


def zonal_spectral_synthesis(zonal: np.ndarray, *, n_lat: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate zonal-mean profiles `(..., lat)` on `n_lat` equal-area latitudes by exact Legendre synthesis.

    The zonal mean of a band-limited field is a Legendre series on the Gaussian
    latitudes; see `spectral_synthesis`. Returns `(values, lat)`.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.special import lpmv

    values = np.asarray(zonal, dtype=float)
    src_lat = values.shape[-1]
    degrees = np.arange((2 * src_lat - 1) // 3 + 1)
    mu_src, _ = leggauss(src_lat)
    mu_dst = _equal_area_mu(n_lat)
    coeffs = np.linalg.lstsq(np.stack([lpmv(0, l, mu_src) for l in degrees], axis=1), values.reshape(-1, src_lat).T, rcond=None)[0]
    fine = (np.stack([lpmv(0, l, mu_dst) for l in degrees], axis=1) @ coeffs).T.reshape(*values.shape[:-1], n_lat)
    return fine, np.degrees(np.arcsin(mu_dst))


def _equal_area_mu(n_lat: int) -> np.ndarray:
    """Cell-centre sin(latitude) of `n_lat` equal-area bands."""
    edges = np.linspace(-1.0, 1.0, n_lat + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _mesh_edges(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell edges for pcolormesh: exact for the Gaussian grid and for equal-area synthesis grids, else the centres."""
    n_lat, n_lon = lat.size, lon.size
    if np.allclose(lat, latitude_centers(n_lat)):
        y = latitude_edges(n_lat)
    elif np.allclose(np.sin(np.deg2rad(lat)), _equal_area_mu(n_lat)):
        y = np.degrees(np.arcsin(np.linspace(-1.0, 1.0, n_lat + 1)))
    else:
        y = lat
    x = longitude_edges(n_lon) if np.allclose(lon, longitude_centers(n_lon)) else lon
    return x, y


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
    spectral: bool = True,
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
        spectral: draw the smooth band-limited field via `spectral_synthesis` (default);
            False draws the grid values as cells. Only applies on the CLERO grid.
        **kwargs: forwarded to `pcolormesh` (e.g. `vmin`, `vmax`, `norm`), useful for
            sharing a colour scale across panels.

    Returns:
        The `(fig, ax)` pair.
    """
    import matplotlib.pyplot as plt

    data = surface_map(outputs_or_field, field, lat=lat, lon=lon)
    values, latc, lonc = _render_field(data["values"], data["lat"], data["lon"], spectral)
    x, y = _mesh_edges(latc, lonc)
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    kwargs.setdefault("rasterized", spectral)  # keep vector output small on the fine grid
    mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap or _default_cmap(field), **kwargs)
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
    spectral: bool = True,
    **kwargs,
):
    """Map of surface ice, where the surface colder than `freeze_K` counts as ice.

    Blue is open water and white is ice. Takes a climate dict (uses `surface_temperature`)
    or a surface temperature array. With `spectral=True` (default) the ice edge is the
    `freeze_K` isoline of the smooth field from `spectral_synthesis`; otherwise whole grid
    cells are classified. Other arguments are passed to `field_map`.
    """
    ts = outputs_or_field["surface_temperature"] if isinstance(outputs_or_field, dict) else outputs_or_field
    ts, lat, lon = _render_field(_as_surface(ts), kwargs.pop("lat", None), kwargs.pop("lon", None), spectral)
    ice = (ts < freeze_K).astype(float)
    return field_map(ice, lat=lat, lon=lon, spectral=False, cmap=cmap, units="ice fraction", title=title, vmin=0.0, vmax=1.0, rasterized=spectral, **kwargs)


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
    spectral: bool = True,
):
    """Wind arrows at one model level drawn over a background field.

    Args:
        outputs: a climate dict (uses `u_{level}`, `v_{level}` and the `scalar` field).
        level: model level for the wind, 0 being the lowest (default 0).
        scalar: name of the background field (default "surface_temperature").
        step: draw an arrow every `step` grid cells (default 2).
        lat, lon, ax, cmap, colorbar, title, spectral: as in `field_map` (`spectral`
            applies to the background; arrows sit on the grid).
    """
    import matplotlib.pyplot as plt

    u = np.asarray(outputs[f"u_{level}"], dtype=float)
    v = np.asarray(outputs[f"v_{level}"], dtype=float)
    bg = _as_surface(outputs[scalar])
    latc = latitude_centers(bg.shape[0]) if lat is None else np.asarray(lat, dtype=float)
    lonc = longitude_centers(bg.shape[1]) if lon is None else np.asarray(lon, dtype=float)
    values, lat_bg, lon_bg = _render_field(bg, latc, lonc, spectral)
    x, y = _mesh_edges(lat_bg, lon_bg)
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap or _default_cmap(scalar), rasterized=spectral)
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
    spectral: bool = True,
):
    """Latitude–pressure cross-section of the zonal mean of a multi-level variable.

    Pass `P0` (bar) for a pressure axis; otherwise the vertical axis is the model level
    index. For temperature with `P0` given, the section extends down to the surface using
    `surface_temperature`, unless `include_surface=False`. With `spectral=True` (default)
    the zonal means are drawn on a fine latitude axis via `zonal_spectral_synthesis`.
    """
    import matplotlib.pyplot as plt

    add_surface = include_surface and variable == "temperature" and P0 is not None and "surface_temperature" in outputs
    stack = stack_levels(outputs, variable, surface_key="surface_temperature" if add_surface else None)
    lat = latitude_centers(stack.shape[1]) if lat is None else np.asarray(lat, dtype=float)
    zonal = stack.mean(axis=2)
    if spectral and np.allclose(lat, latitude_centers(lat.size)):
        zonal, lat = zonal_spectral_synthesis(zonal)
    pressure_axis = P0 is not None
    y = pressure_levels(P0, surface=add_surface) / 100.0 if pressure_axis else np.arange(stack.shape[0], dtype=float)
    fig, ax = (plt.subplots(figsize=(5, 3), constrained_layout=True) if ax is None else (ax.figure, ax))
    cf = ax.contourf(lat, y, zonal, levels=contour_levels, cmap=cmap or _default_cmap(variable))
    ax.invert_yaxis()
    if pressure_axis:
        ax.set_yscale("log")
    ax.set(xlabel="latitude", ylabel="pressure / hPa" if pressure_axis else "model level")
    fig.colorbar(cf, ax=ax, label=units or _CBAR_LABEL.get(_base_field(variable), variable))
    if title is not None:
        ax.set_title(title)
    return fig, ax


def _render_field(values: np.ndarray, lat: np.ndarray | None, lon: np.ndarray | None, spectral: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The field and axes to draw: the exact spectral synthesis on the CLERO grid when `spectral`, else the grid values."""
    lat = latitude_centers(values.shape[0]) if lat is None else np.asarray(lat, dtype=float)
    lon = longitude_centers(values.shape[1]) if lon is None else np.asarray(lon, dtype=float)
    if spectral and np.allclose(lat, latitude_centers(lat.size)) and np.allclose(lon, longitude_centers(lon.size)):
        return spectral_synthesis(values)
    return values, lat, lon


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
