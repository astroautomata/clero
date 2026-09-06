# clero demos

Executable notebooks showing how to use CLERO


| notebook                    | what it shows                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `quickstart.ipynb`          | Predict a planet, plot a map and a profile, get the per-cell variance, draw samples.                              |
| `trappist1e.ipynb`          | A single-planet report: surface map, dayside, nightside and global means, vertical profiles, CSV export.          |
| `uncertainty.ipynb`         | Uncertainties from samples and the analytic variance: derived quantities, the global mean, humidity in model space, and the core domain. |
| `parameter_sweep.ipynb`     | A 400-planet `F_star` × `CO2` batch, mapped as global-mean surface temperature.                                  |
| `gpu_metric_mosaic.ipynb`   | A 10,000-planet `P_rot` × `P0` batch on the GPU, summarized as a 3 × 3 mosaic of climate diagnostics.            |


Keep notebooks inference-only: loading the exported checkpoint, prediction/sampling,
and lightweight summary stats. Do not add training-time objects or internal
export/build workflows here.

Every notebook is executed end-to-end in CI (`tests/test_demos.py`), so they must
run top-to-bottom against the bundled model.

