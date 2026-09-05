from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from clero.climate_analysis import (
    bond_albedo,
    dayside_mean,
    field_map,
    ice_fraction,
    ice_fraction_map,
    latitude_centers,
    latitude_weights,
    map_records,
    meridional_mean,
    net_toa_radiation,
    nightside_mean,
    plot_profile,
    pressure_levels,
    profile_table,
    stack_levels,
    summary_table,
    surface_map,
    vertical_profile,
    water_vapor_path,
    write_csv,
    zonal_mean,
)


def outputs() -> dict[str, np.ndarray]:
    return {
        "surface_temperature": np.arange(8.0).reshape(2, 4),
        "temperature": np.arange(24.0).reshape(3, 2, 4),
    }


def test_summary_profile_and_map_tables_are_prediction_only(tmp_path: Path) -> None:
    pred = outputs()
    lat = latitude_centers(2)
    summary = summary_table(pred, lat)
    profiles = profile_table(pred, variables=["temperature"], lat=lat, levels=np.array([1000.0, 500.0, 100.0]))
    records = map_records(pred, "surface_temperature", lat=lat, value_name="temperature")

    assert {"name": "surface_temperature", "metric": "global_mean", "value": 3.5} in summary
    assert len(profiles) == 9
    assert profiles[0] == {"variable": "temperature", "profile": "global_mean", "level": 1000.0, "value": 3.5}
    assert_allclose(records[0]["lat"], -35.26438968)
    assert records[0]["lon"] == -135.0
    assert records[0]["temperature"] == 0.0
    assert write_csv(summary, tmp_path / "summary.csv").read_text().startswith("metric,name,value\n")


def test_surface_map_and_axis_summaries() -> None:
    field = outputs()["surface_temperature"]
    data = surface_map(field)

    assert_allclose(data["lat"], [-35.26438968, 35.26438968])
    assert_allclose(data["lon"], [-135.0, -45.0, 45.0, 135.0])
    assert_allclose(data["values"], field)
    assert_allclose(latitude_centers(4), [-59.44440829, -19.87571915, 19.87571915, 59.44440829])
    assert_allclose(latitude_weights(latitude_centers(4)), [0.17392742, 0.32607258, 0.32607258, 0.17392742])
    assert_allclose(zonal_mean(field), [1.5, 5.5])
    assert_allclose(zonal_mean(outputs()["temperature"]), [[1.5, 5.5], [9.5, 13.5], [17.5, 21.5]])
    assert_allclose(meridional_mean(field, data["lat"]), [2.0, 3.0, 4.0, 5.0])


def test_dayside_and_nightside_means_return_scalars() -> None:
    field = np.array([[0.0, 100.0, 100.0, 0.0], [0.0, 10.0, 10.0, 0.0]])
    lat = latitude_centers(2)

    assert dayside_mean(field, lat) == 55.0
    assert nightside_mean(field, lat) == 0.0


def test_field_map_has_no_default_title_and_substellar_longitude_axis() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    before = dict(plt.rcParams)
    fig, ax = field_map(outputs()["surface_temperature"], colorbar=False)

    assert dict(plt.rcParams) == before  # styling is scoped to the call, not the session
    assert ax.get_title() == ""
    assert_allclose(ax.get_xlim(), [-180.0, 180.0])
    assert 0 in ax.get_xticks()
    plt.close(fig)


def test_ice_fraction_map_is_blue_white_zero_one_field() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    pred = {"surface_temperature": np.array([[200.0, 300.0], [260.0, 280.0]])}
    fig, ax = ice_fraction_map(pred, colorbar=False)

    vals = np.asarray(ax.collections[0].get_array()).ravel()
    assert set(np.unique(vals)).issubset({0.0, 1.0})  # frozen below 273.15 K -> 1
    plt.close(fig)


def test_water_vapor_path_matches_hydrostatic_integral() -> None:
    P0, g = 1.0, 9.8                               # P0 in bar (as in the emulator inputs)
    plev = pressure_levels(P0)
    assert plev.shape == (10,)
    assert plev[0] > plev[-1]                      # level 0 (surface) is highest pressure
    assert_allclose(plev[-1], 1000.0)              # TOA at P_top
    assert_allclose(plev[0], 0.925 * P0 * 1.0e5)   # surface level squeezed to f_bottom * P0

    q = np.full((10, 2, 4), 1.0e-3)                # uniform 1 g/kg
    wvp = water_vapor_path({f"specific_humidity_{i}": q[i] for i in range(10)}, P0=P0, gravity=g)
    expected = abs(np.trapezoid(q[:, 0, 0], x=plev)) / g
    assert wvp.shape == (2, 4)
    assert (wvp > 0).all()
    assert_allclose(wvp, expected)                 # uniform field -> uniform column
    assert_allclose(water_vapor_path(q, P0=P0, gravity=g), wvp)  # array input form


def test_net_toa_radiation_is_asr_minus_olr() -> None:
    asr = np.array([[10.0, 20.0], [30.0, 40.0]])
    olr = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert_allclose(net_toa_radiation({"asr": asr, "olr": olr}), [[9.0, 18.0], [27.0, 36.0]])
    assert_allclose(net_toa_radiation(asr, olr), [[9.0, 18.0], [27.0, 36.0]])


def test_ice_fraction_and_bond_albedo_match_metric_mosaic_definitions() -> None:
    lat = latitude_centers(4)
    surface = np.array([
        [260.0, 260.0, 280.0, 280.0],
        [260.0, 280.0, 280.0, 280.0],
        [260.0, 260.0, 260.0, 280.0],
        [280.0, 280.0, 280.0, 280.0],
    ])
    asr = np.full((4, 4), 200.0)

    expected_ice = np.average((surface < 273.15).mean(axis=1), weights=latitude_weights(lat))
    assert_allclose(ice_fraction({"surface_temperature": surface}, lat), expected_ice)
    assert_allclose(ice_fraction(surface, lat), expected_ice)
    assert_allclose(bond_albedo({"asr": asr}, 1000.0, lat), 0.2)
    assert_allclose(bond_albedo(asr, 1000.0, lat), 0.2)
    assert bond_albedo(asr, 100.0, lat) == 0.0


def test_profile_selection_uses_named_profiles() -> None:
    pred = outputs()
    lat = latitude_centers(2)

    assert_allclose(vertical_profile(pred, "temperature", lat=lat, profile="global_mean"), [3.5, 11.5, 19.5])
    assert_allclose(vertical_profile(pred["temperature"], lat=lat, profile="antistellar"), [5.5, 13.5, 21.5])


def test_plot_profile_shows_global_mean_pressure_profile() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    pred = outputs()
    levels = np.array([100000.0, 50000.0, 10000.0])
    fig, ax = plot_profile(pred, "temperature", levels=levels, units="K")

    assert ax.get_title() == ""
    assert ax.get_xlabel() == "K"
    assert ax.get_ylabel() == "pressure / hPa"
    assert ax.yaxis_inverted()
    assert ax.get_yscale() == "log"
    assert ax.lines[0].get_color() == "black"
    assert ax.lines[0].get_linewidth() == 2.0
    assert_allclose(ax.lines[0].get_xdata(), [3.5, 11.5, 19.5])
    assert_allclose(ax.lines[0].get_ydata(), levels / 100.0)
    plt.close(fig)


def test_level_stacking_matches_real_bundle_field_naming() -> None:
    pred = {
        "temperature_1": np.full((2, 2), 1.0),
        "temperature_0": np.zeros((2, 2)),
        "surface_temperature": np.ones((2, 2)),
    }

    assert_allclose(stack_levels(pred, "temperature"), np.stack([np.zeros((2, 2)), np.ones((2, 2))]))
    assert_allclose(vertical_profile(pred, "temperature", lat=latitude_centers(2), profile="global_mean"), [0.0, 1.0])


def test_temperature_profile_extends_to_surface_at_p0() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    pred = {f"temperature_{k}": np.full((2, 4), 250.0 + k) for k in range(10)}
    pred["surface_temperature"] = np.full((2, 4), 288.0)
    P0 = 1.0  # bar

    # helpers prepend exactly one surface element, paired at P0
    assert stack_levels(pred, "temperature").shape[0] == 10
    surf_stack = stack_levels(pred, "temperature", surface_key="surface_temperature")
    assert surf_stack.shape[0] == 11
    assert_allclose(surf_stack[0], 288.0)
    surf_plev = pressure_levels(P0, surface=True)
    assert surf_plev.shape == (11,)
    assert_allclose(surf_plev[0], P0 * 1.0e5)
    assert_allclose(surf_plev[1:], pressure_levels(P0))

    # surface on by default: bottom point sits at P0 (1000 hPa) = surface global-mean
    fig, ax = plot_profile(pred, "temperature", P0=P0)
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert len(x) == 11
    assert_allclose(y.max(), 1000.0)
    assert_allclose(x[0], 288.0)
    plt.close(fig)

    # opt-out drops back to model-level-only; no pressure axis also skips surface
    assert len(plot_profile(pred, "temperature", P0=P0, include_surface=False)[1].lines[0].get_xdata()) == 10
    assert len(plot_profile(pred, "temperature", include_surface=True)[1].lines[0].get_xdata()) == 10
    plt.close("all")
