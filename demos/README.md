# clero demos

Executable notebooks showing how to use CLERO


| notebook                    | what it shows                                                                                                                                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `quickstart.ipynb`          | Introduction to CLERO. Basic usage.                                                                                                                                                                            |
| `trappist1e.ipynb`          | A single-planet climate report for a real target — surface map, dayside/nightside/global summary, vertical profiles, and CSV export.                                                                           |
| `using_clero_samples.ipynb` | Working with predictive samples: the climate distribution of a derived scalar, the model-space-then-`to_physical` recipe for skewed fields, field subsetting, and batched/GPU sampling.                     |
| `gpu_metric_mosaic.ipynb`   | GPU-accelerated 10,000-case `P_rot` × `P0` batch prediction over the extended domain, summarised as a 3x3 mosaic of nine climate diagnostics.                                                            |
| `parameter_sweep.ipynb`     | A 400-planet `F_star` × `CO2` sweep mapping global-mean surface temperature — the kind of sensitivity scan an emulator makes instant.                                                                          |
| `uncertainty.ipynb`         | Calibrated uncertainty: the climate distribution of a derived scalar from samples, the model-space-then-`to_physical` recipe for skewed fields, and predictive uncertainty growing outside the core domain. |


Keep notebooks inference-only: loading the exported checkpoint, prediction/sampling,
and lightweight summary stats. Do not add training-time objects or internal
export/build workflows here.

Every notebook is executed end-to-end in CI (`tests/test_demos.py`), so they must
run top-to-bottom against the bundled model.

