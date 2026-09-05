"""Summaries, profiles, maps and diagnostics for a predicted climate.

Every function here takes the output of `Emulator.predict` (a dict of `(32, 64)` fields,
called a climate dict below) or a single field array, and needs nothing beyond NumPy and
matplotlib. Functions come in groups:

- area means: `global_mean`, `dayside_mean`, `nightside_mean`, `summarize_outputs`, `summary_table`
- vertical structure: `stack_levels`, `pressure_levels`, `vertical_profile`, `profile_stats`, `profile_table`
- diagnostics: `water_vapor_path`, `net_toa_radiation`, `ice_fraction`, `bond_albedo`
- maps and plots: `field_map`, `ice_fraction_map`, `net_radiation_map`, `wind_map`, `wind_streamlines`,
  `zonal_cross_section`, `plot_profile`, `surface_map`, `zonal_mean`, `meridional_mean`
- grid: `latitude_centers`, `longitude_centers`, `latitude_edges`, `longitude_edges`, `latitude_weights`
- export: `grid_records`, `map_records`, `write_csv`

Multi-level variables (temperature, specific humidity, cloud fraction, u, v) are stored as
ten separate fields `name_0` to `name_9`; `stack_levels` assembles them into one
`(level, lat, lon)` array, which is what the profile and cross-section functions expect.
"""

from .grid import grid_records, latitude_centers, latitude_edges, latitude_weights, longitude_centers, longitude_edges
from .maps import (
    field_map,
    ice_fraction_map,
    map_records,
    meridional_mean,
    net_radiation_map,
    spectral_synthesis,
    surface_map,
    wind_map,
    wind_streamlines,
    zonal_cross_section,
    zonal_mean,
    zonal_spectral_synthesis,
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
    "spectral_synthesis",
    "surface_map",
    "vertical_profile",
    "water_vapor_path",
    "wind_map",
    "wind_streamlines",
    "write_csv",
    "zonal_cross_section",
    "zonal_mean",
    "zonal_spectral_synthesis",
]
