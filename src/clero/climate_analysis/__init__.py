"""climate analysis helpers for CLERO prediction dictionaries."""

from .grid import grid_records, latitude_centers, latitude_edges, latitude_weights, longitude_centers, longitude_edges
from .maps import (
    field_map,
    ice_fraction_map,
    map_records,
    meridional_mean,
    net_radiation_map,
    surface_map,
    wind_map,
    wind_streamlines,
    zonal_cross_section,
    zonal_mean,
)
from .profiles import plot_profile, pressure_levels, profile_table, stack_levels, vertical_profile
from .diagnostics import bond_albedo, ice_fraction, net_toa_radiation, water_vapor_path
from .scalars import (
    dayside_mean,
    global_mean,
    nightside_mean,
    summarize_outputs,
    summary_table,
    write_csv,
)
from .summary_stats import profile_stats

__all__ = [
    "bond_albedo",
    "dayside_mean",
    "field_map",
    "global_mean",
    "grid_records",
    "ice_fraction",
    "ice_fraction_map",
    "latitude_centers",
    "latitude_edges",
    "latitude_weights",
    "longitude_centers",
    "longitude_edges",
    "map_records",
    "meridional_mean",
    "net_radiation_map",
    "net_toa_radiation",
    "nightside_mean",
    "plot_profile",
    "pressure_levels",
    "profile_table",
    "profile_stats",
    "summarize_outputs",
    "summary_table",
    "stack_levels",
    "surface_map",
    "vertical_profile",
    "water_vapor_path",
    "wind_map",
    "wind_streamlines",
    "write_csv",
    "zonal_cross_section",
    "zonal_mean",
]
