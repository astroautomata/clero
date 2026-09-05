# Spaces and uncertainty

How CLERO represents its climate distribution and how to summarise it correctly. `emu` and `inputs` are as in the README quickstart; `import numpy as np` is assumed.

**Two spaces.** Each field has a *model space* in which CLERO's climate distribution is Gaussian: log for specific humidity, smoothed logit for cloud fraction, and the identity for temperature, winds and fluxes. `space="physical"` (default) returns natural units; `space="model"` returns the transformed values. Summary statistics of the skewed fields (humidity, cloud fraction) should be computed in model space and mapped back with `to_physical`; a physical-space `mean ± std` of specific humidity, for instance, implies negative values.

**Three ways to get uncertainty.**

- `predict(..., space="model", return_variance=True)` also returns, for every field, the per-cell variance of the climate distribution (a `(32, 64)` array matching the mean), computed exactly rather than from samples. It carries no covariance between cells, so it cannot give the uncertainty of an area mean. It raises a `ValueError` in physical space, for the reason above. `split_variance=True` separates the spatially coherent part from the spatially white residual.
- `sample(...)` returns draws from the climate distribution. By default they are spatially coherent, which is what you want for maps, masks and any statistic computed from a whole draw, including the uncertainty of global or regional means (push each draw through the statistic).
- `sample(..., sample_residual=True)` adds the spatially white residual term to each draw, so per-cell spreads match the variance from `predict` (which always includes it). Details in the `sample` docstring and the paper's Methods.

**Example: summarise humidity in dex**

Using samples:

```python
samples = emu.sample(inputs, n_samples=256, space="model", fields=["specific_humidity_3"], sample_residual=True)
logq = samples["specific_humidity_3"] / np.log(10.0)   # ln(q) -> log10(q)
logq_mean = logq.mean(axis=0)
logq_std = logq.std(axis=0, ddof=1)
q_center = emu.to_physical("specific_humidity_3", logq_mean * np.log(10.0), inputs)
```

Or analytically:

```python
mean, var = emu.predict(inputs, space="model", return_variance=True)
logq_mean = mean["specific_humidity_3"] / np.log(10.0)
logq_std = np.sqrt(var["specific_humidity_3"]) / np.log(10.0)
q_center = emu.to_physical("specific_humidity_3", mean["specific_humidity_3"], inputs)
```

**Convert between spaces** (the transform helpers need the inputs since ASR/OLR are F_star-normalised)

```python
physical = emu.to_physical(model_fields, inputs)   # model space -> physical units
model = emu.to_model(physical_fields, inputs)      # physical units -> model space

q = emu.to_physical("specific_humidity_3", model_q, inputs)   # one field -> one array
```
