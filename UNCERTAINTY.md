# Spaces and uncertainty

## Two spaces

- Each variable can be returned in "physical space" or "model space".
- Physical space just means where the variable has its natural units; this is usually ultimately what we want. It is set by `space="physical"` (default)
- But CLERO natively predicts variables in "model space" where some variables have been transformed such that their distributions over planets are more Gaussian-like (some variables are already Gaussian-like in physical space, such as temperature and winds; others are highly skewed, like specific humidity and cloud fraction, so these get transformed). It is set by `space="model"`
- If you just want a prediction, use physical space. 
- If you need summary statistics (means, standard deviations, etc.), you should computed these in *model space* first, and then map back to natural units with `to_physical`, or you will get skewed summaries.

**Example: summarise humidity in dex**

Analytically:

```python
mean, var = emu.predict(inputs, space="model", return_variance=True)
q_median = emu.to_physical("specific_humidity_3", mean["specific_humidity_3"], inputs)  # kg/kg
q_std_dex = np.sqrt(var["specific_humidity_3"]) / np.log(10.0)                         # ± in dex
```

Or using samples:

```python
samples = emu.sample(inputs, n_samples=256, space="model", fields=["specific_humidity_3"])
lnq = samples["specific_humidity_3"]
q_median = emu.to_physical("specific_humidity_3", lnq.mean(axis=0), inputs)  # kg/kg
q_std_dex = lnq.std(axis=0, ddof=1) / np.log(10.0)                          # ± in dex
```

**Convert between spaces** (the transform helpers need the inputs since ASR/OLR are F_star-normalised)

```python
physical = emu.to_physical(model_fields, inputs)   # model space -> physical units
model = emu.to_model(physical_fields, inputs)      # physical units -> model space

q = emu.to_physical("specific_humidity_3", model_q, inputs)   # one field -> one array
```

## Three ways to get uncertainty

- `predict(..., space="model", return_variance=True)` also returns, for every field, the per-cell variance of the climate distribution (a `(32, 64)` array matching the mean), computed exactly rather than from samples. It carries no covariance between cells, so it cannot give the uncertainty of an area mean. It raises a `ValueError` in physical space, for the reason in 'Two spaces'.
- `sample(...)` returns draws from the climate distribution.
- Both of the above exclude a spatially white residual term (per-cell scatter with no spatial structure) by default. `predict(..., include_residual=True)` and `sample(..., sample_residual=True)` add it, and the two remain consistent with each other. This is mainly for quantitative cell-by-cell comparison against other climate predictions like GCM output, and is probably rarely useful otherwise. `predict(..., split_variance=True)` returns the coherent and residual variances separately. Details in the paper's Methods.


