# CLERO (CLimate Emulator for ROcky exoplanets)

![Tests](https://github.com/edstevenson/clero/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Docs](https://img.shields.io/badge/docs-pdoc-brightgreen.svg)
![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)  

## What is it

CLERO is an exoGCM emulator that takes a hypothetical planet as input and outputs its 3D steady-state climate. It targets tidally locked ocean-covered rocky planets in or near the habitable zone; see [SCOPE.md](SCOPE.md) for detail on its scope of validity.

CLERO computes a probability distribution over climates: `predict` returns its mean (our best point estimate of the climate) and `sample` returns draws from the distribution (see [UNCERTAINTY.md](UNCERTAINTY.md)).

CLERO is based on Gaussian-process latent factor regression ([GPLFR](https://github.com/edstevenson/GPLFR)) and is trained on [ThousandWorlds](https://github.com/astroautomata/ThousandWorlds).

Worked examples in [demos/](demos/).

## Inputs


| key     | unit            | notes                                                                                                    |
| ------- | --------------- | -------------------------------------------------------------------------------------------------------- |
| radius  | Earth radii     | planet radius                                                                                            |
| gravity | m/s²            | surface gravity                                                                                          |
| P_rot   | days            | rotation period (assumed = orbital, tidally locked)                                                      |
| P0      | bar             | surface pressure                                                                                         |
| CO2     | volume fraction | atmospheric CO2                                                                                          |
| CH4     | volume fraction | atmospheric CH4                                                                                          |
| F_star  | W/m²            | stellar flux at the planet                                                                               |
| T_star  | K               | stellar effective temperature                                                                            |
| GCM     | str             | climate model the prediction targets (optional, defaults to `"um"`; case-insensitive; see options below) |


`CO2 + CH4 <= 1`; the rest of the atmosphere is N2. The validity ranges from [SCOPE.md](SCOPE.md) are available as `clero.CORE_DOMAIN` and `clero.EXTENDED_DOMAIN`.

#### GCM options

`um`, `exocam` (recommended — the two high-fidelity targets). `exoplasim` is a lower-fidelity exoGCM. `exocam-pre2022` and `lfric` should be avoided at inference time — `exocam-pre2022` is an older ExoCAM version, and  `lfric` has few training simulations.

For predictions where self-consistency is important (e.g., spatially resolved plots), use a single emulated GCM. Where self-consistency is not needed (e.g., global means), once can average the emulated UM and ExoCAM predictions; this reduces dependence on either GCM's structural biases.

## Outputs


| fields                   | unit  |
| ------------------------ | ----- |
| `surface_temperature`    | K     |
| `asr`, `olr`             | W/m²  |
| `temperature_0..9`       | K     |
| `specific_humidity_0..9` | kg/kg |
| `cloud_fraction_0..9`    | 0–1   |
| `u_0..9`, `v_0..9`       | m/s   |


The 3D variables (`temperature`, `specific_humidity`, `cloud_fraction`, `u`, `v`) are given on 10 pressure levels, level 0 closest to the surface and level 9 closest to the top of the atmosphere; `clero.climate_analysis.pressure_levels(P0)` returns the level pressures.

## Install

```bash
pip install clero
```

Optional extras:

```bash
pip install "clero[gpu]"    # torch, for CUDA batch prediction
pip install "clero[demos]"  # jupyter, to run the notebooks in demos/
```



## Quickstart

```python
from clero import Emulator

emu = Emulator()
inputs = {
    "T_star": 3000.0,     # K
    "F_star": 1000.0,     # W/m^2
    "radius": 1.0,        # Earth radii
    "gravity": 9.8,       # m/s^2
    "P_rot": 10.0,        # days
    "P0": 1.0,            # bar
    "CO2": 4.0e-4,        # volume fraction
    "CH4": 0.0,           # volume fraction
    "GCM": "um",
}

# predict climate
mean = emu.predict(inputs)                           # CLERO's best point estimate of the climate
samples = emu.sample(inputs, n_samples=100, seed=0)   # draws from climate distribution

print(mean["surface_temperature"].shape)     # (32, 64)
print(samples["surface_temperature"].shape)  # (100, 32, 64)
```

Don't want to spell out every parameter? Start from a bundled preset and override what you need:

```python
from clero import EARTH, M_EARTH, TRAPPIST1E
mean = emu.predict({**EARTH, "CO2": 1.0e-3})   # Earth-like planet, overriden to have 1000 ppm CO2
mean = emu.predict(M_EARTH)                    # Earth-like but around a 2600 K M dwarf (self-consistent 5 day rotation period)
mean = emu.predict({**TRAPPIST1E, "CO2": 0.0, "CH4": 0.0})    # TRAPPIST-1e with an N₂-only atmosphere 
```

There's also a walk-through notebook, [demos/quickstart.ipynb](demos/quickstart.ipynb).

## API

Full reference (generated from the docstrings): [https://edstevenson.github.io/clero/](https://edstevenson.github.io/clero/)

`Emulator` (prediction):


| call                          | returns                                                                                   | output type                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `predict(inputs)`             | predict the climate (mean of climate distribution)                                        | `dict[str, ndarray]`, each dict value is `(32, 64)`            |
| `sample(inputs, n_samples=…)` | draws from climate distribution                                                           | `dict[str, ndarray]`, each dict value is `(n_samples, 32, 64)` |
| `to_physical` / `to_model`    | move a field dict between physical and model space (see [UNCERTAINTY.md](UNCERTAINTY.md)) | `dict[str, ndarray]`, shapes preserved                         |
| `output_names`, `grid_shape`  | the 53 field names; `(32, 64)`                                                            | `list[str]`; `tuple[int, int]`                                 |


Top-level helpers:


| name                             | what                                                                                                                                               |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EARTH`, `M_EARTH`, `TRAPPIST1E` | preset input dicts (see [Quickstart](#quickstart))                                                                                                 |
| `CORE_DOMAIN`, `EXTENDED_DOMAIN` | `(low, high)` per input, from [SCOPE.md](SCOPE.md)                                                                                                 |
| `orbital_period(F_star, T_star)` | tidally locked rotation period in days from flux and stellar temperature, via empirical stellar relations; prefer a measured period when available |


`clero.climate_analysis` (helper functions for analyzing climates):


| group                | functions                                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| scalar summaries     | `summarize_outputs`, `summary_table`, `global_mean`, `dayside_mean`, `nightside_mean`                                       |
| vertical structure   | `vertical_profile`, `profile_table`, `profile_stats`, `stack_levels`, `pressure_levels`                                     |
| physical diagnostics | `water_vapor_path`, `net_toa_radiation`, `ice_fraction`, `bond_albedo`                                                      |
| maps & grids         | `surface_map`, `zonal_mean`, `meridional_mean`, `map_records`, `grid_records`, `spectral_synthesis`, `zonal_spectral_synthesis` |
| plots                | `field_map`, `ice_fraction_map`, `net_radiation_map`, `wind_map`, `wind_streamlines`, `zonal_cross_section`, `plot_profile` |
| axes & weights       | `latitude_centers`, `longitude_centers`, `latitude_edges`, `longitude_edges`, `latitude_weights`                            |
| io                   | `write_csv`                                                                                                                 |


e.g.,

```python
from clero.climate_analysis import stack_levels
climate_mean = emu.predict(inputs)
T = stack_levels(climate_mean, "temperature")  # per-level fields -> one (10, 32, 64) array
```

The plotting functions draw the smooth band-limited field rather than the 32×64 cells: CLERO's output is a T21 spherical-harmonic expansion, so `spectral_synthesis` evaluates it exactly on a fine equal-area grid (no information is added; the grid values are samples of this field). Pass `spectral=False` to see the cells instead. Statistics and diagnostics always use the native grid.



## Batches and GPU

`predict`/`sample` take a batch directly (list of dicts or a column dict). CPU stays pure NumPy:

```python
mean, variance = emu.predict(
    inputs_list,                       # list of input dicts
    space="model",
    return_variance=True,
    fields=["surface_temperature"]     # subset of outputs
)
```

GPU needs torch with CUDA. Build the emulator with a device and the torch state is cached on first use:

```python
emu = Emulator(device="cuda")
samples = emu.sample(inputs_list, n_samples=64)
```



## Citation

If you use CLERO, please cite the paper:

```bibtex
// TODO: add correct CLERO paper citation here when available
```

