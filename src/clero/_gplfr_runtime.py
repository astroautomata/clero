"""NumPy inference path for exported GPLFR map bundles."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from ._checkpoint import _CheckpointBundle
from .domain import EXTENDED_DOMAIN


ASR_OLR_FIELDS = {"asr", "olr"}
BatchInputs = list[dict[str, float | str]] | tuple[dict[str, float | str], ...] | dict[str, Any]

# Public inputs use astro-friendly units; the model consumes SI. Convert at the input boundary.
_R_EARTH = 6.371e6  # m per Earth radius (radius input)
_BAR = 1.0e5  # Pa per bar (P0 input)
_INPUT_UNIT_SCALE = {"radius": _R_EARTH, "P0": _BAR}
_DEFAULT_GCM = "um"  # GCM target used when inputs omit "GCM"
_POSITIVE_INPUTS = ("T_star", "F_star", "radius", "gravity", "P_rot", "P0")


_VARIANCE_SPACE_MSG = (
    "Predictive variance is not meaningful in physical space for nonlinearly-transformed "
    "fields (e.g. humidity, cloud fraction). Request space='model' for the Gaussian "
    "predictive variance, and inverse-transform with Emulator.to_physical if you need "
    "physical-space quantities."
)


def predict_gplfr_batch(
    bundle: _CheckpointBundle,
    inputs: BatchInputs,
    *,
    space: str = "physical",
    return_variance: bool = False,
    split_variance: bool = False,
    batch_size: int | None = 256,
    fields: list[str] | tuple[str, ...] | None = None,
    unpack: bool = True,
):
    _check_space(space)
    return_variance = return_variance or split_variance
    if return_variance and space != "model":
        raise ValueError(_VARIANCE_SPACE_MSG)
    x_raw, s = _coerce_inputs_batch(bundle, inputs)
    if not return_variance:
        return _concat_batch([
            _pack_batch(bundle, _predict_grid_arrays(bundle, x_raw[start:stop], s[start:stop], space), fields, unpack)
            for start, stop in _chunks(x_raw.shape[0], batch_size)
        ])
    parts = []
    for start, stop in _chunks(x_raw.shape[0], batch_size):
        arrays = _predict_transformed_grid_mean_and_variance_arrays(bundle, x_raw[start:stop], s[start:stop], split_variance)
        parts.append(tuple(_pack_batch(bundle, arr, fields, unpack) for arr in arrays))
    return tuple(_concat_batch(list(group)) for group in zip(*parts))


def predict_gplfr_samples_batch(
    bundle: _CheckpointBundle,
    inputs: BatchInputs,
    *,
    space: str = "physical",
    n_samples: int = 64,
    seed: int | None = None,
    sample_residual: bool = False,
    batch_size: int | None = 256,
    fields: list[str] | tuple[str, ...] | None = None,
    unpack: bool = True,
) -> dict[str, np.ndarray] | np.ndarray:
    _check_space(space)
    x_raw, s = _coerce_inputs_batch(bundle, inputs)
    rng = np.random.default_rng(seed)
    return _concat_sample_batch([
        _pack_sample_batch(bundle, _predict_grid_samples(bundle, x_raw[start:stop], s[start:stop], n_samples, rng, space, sample_residual), fields, unpack)
        for start, stop in _chunks(x_raw.shape[0], batch_size)
    ])


def outputs_to_physical(bundle: _CheckpointBundle, fields: dict[str, np.ndarray], inputs: BatchInputs) -> dict[str, np.ndarray]:
    """Map model-space (Gaussian) output fields to physical units."""
    return _transform_outputs(bundle, fields, inputs, inverse=True)


def outputs_to_model(bundle: _CheckpointBundle, fields: dict[str, np.ndarray], inputs: BatchInputs) -> dict[str, np.ndarray]:
    """Map physical output fields to model space (where the predictive is Gaussian)."""
    return _transform_outputs(bundle, fields, inputs, inverse=False)


def _transform_outputs(bundle: _CheckpointBundle, fields: dict[str, np.ndarray], inputs: BatchInputs, *, inverse: bool) -> dict[str, np.ndarray]:
    transforms = bundle.manifest["output_transforms"]
    raw_by_output = dict(zip(bundle.output_names, bundle.manifest["raw_output_names"]))
    normalize = bundle.manifest.get("asr_olr_normalize_by_f_star", False)
    f_star = _f_star_factor(bundle, inputs)
    out = {}
    for name, arr in fields.items():
        raw = raw_by_output[name]
        spec = transforms[raw]
        a = np.asarray(arr, dtype=np.float64)
        scale = normalize and _base_var(raw) in ASR_OLR_FIELDS
        if inverse:
            a = _inverse_preprocess(a, spec["strategy"], spec["kwargs"])
            if _base_var(raw) == "specific_humidity":
                a = np.clip(a, 0.0, 1.0)
            a = a * f_star if scale else a
        else:
            a = a / f_star if scale else a
            a = _preprocess(a, spec["strategy"], spec["kwargs"])
        out[name] = a.astype(np.float32, copy=False)
    return out


def _f_star_factor(bundle: _CheckpointBundle, inputs: BatchInputs) -> float | np.ndarray:
    idx = bundle.input_names.index("F_star")
    if _is_batch(inputs):
        x, _ = _coerce_inputs_batch(bundle, inputs)
        return x[:, idx].reshape(-1, 1, 1)
    return float(inputs["F_star"])


def _is_batch(inputs: BatchInputs) -> bool:
    if isinstance(inputs, (list, tuple)):
        return True
    if isinstance(inputs, dict):
        seq = {key: np.ndim(value) > 0 for key, value in inputs.items()}
        if any(seq.values()) and not all(seq.values()):
            scalar_keys = [key for key, is_seq in seq.items() if not is_seq]
            raise ValueError(f"input dict mixes scalar and sequence values (scalar: {scalar_keys}); pass a single planet (all scalars) or a column batch (all sequences)")
        return any(seq.values())
    raise TypeError("inputs must be a planet dict, a list of planet dicts, or a column dict")


def _check_space(space: str) -> None:
    if space not in ("physical", "model"):
        raise ValueError(f"space must be 'physical' or 'model', got {space!r}")


def _predict_grid_arrays(bundle: _CheckpointBundle, x_raw: np.ndarray, s: np.ndarray, space: str = "physical") -> np.ndarray:
    state = _state(bundle)
    x = _transform_inputs(bundle, x_raw)
    z_mean = _latent_mean(state, x, s)
    y = _decoder_mean(bundle, state, z_mean)
    if "linear_Gamma" in state:
        y = y + np.einsum("np,paf->naf", _design_matrix(bundle, x, s), state["linear_Gamma"])
    y = y * state.get("sh_mask", np.ones(y.shape[1:], dtype=bool))[None]
    grid = _spectral_grid_mean(bundle, state, y)
    return grid if space == "model" else _inverse_preprocess_outputs(bundle, grid, x_raw)


def _predict_transformed_grid_mean_and_variance_arrays(bundle: _CheckpointBundle, x_raw: np.ndarray, s: np.ndarray, split: bool = False) -> tuple[np.ndarray, ...]:
    state = _state(bundle)
    _check_variance_enabled(state)
    x = _transform_inputs(bundle, x_raw)
    z_mean, z_var = _latent_stats(state, x, s)
    y_mean, *y_vars = _decoder_mean_and_variance(bundle, state, z_mean, z_var, split)
    if "linear_Gamma" in state:
        y_mean = y_mean + np.einsum("np,paf->naf", _design_matrix(bundle, x, s), state["linear_Gamma"])
    mask = state.get("sh_mask", np.ones(y_mean.shape[1:], dtype=bool))[None]
    mean = _spectral_grid_mean(bundle, state, y_mean * mask).astype(np.float32, copy=False)
    return (mean, *(_spectral_grid_variance(bundle, state, y_var * mask).reshape(mean.shape) for y_var in y_vars))


def _predict_grid_samples(bundle: _CheckpointBundle, x_raw: np.ndarray, s: np.ndarray, n_samples: int, rng: np.random.Generator, space: str = "physical", sample_residual: bool = False) -> np.ndarray:
    state = _state(bundle)
    _check_variance_enabled(state)
    x = _transform_inputs(bundle, x_raw)
    z_mean, z_var = _latent_stats(state, x, s)
    z = z_mean[None] + np.sqrt(z_var)[None] * rng.standard_normal((int(n_samples), *z_mean.shape))
    y = _decoder_sample(bundle, state, z, rng, sample_residual)
    if "linear_Gamma" in state:
        y = y + np.einsum("np,paf->naf", _design_matrix(bundle, x, s), state["linear_Gamma"])[None]
    y = y * state.get("sh_mask", np.ones(y.shape[2:], dtype=bool))[None, None]
    flat = _spectral_grid_mean(bundle, state, y.reshape(-1, y.shape[2], y.shape[3]))
    grid = flat if space == "model" else _inverse_preprocess_outputs(bundle, flat, np.tile(x_raw, (int(n_samples), 1)))
    return grid.reshape(int(n_samples), x_raw.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]).astype(np.float32, copy=False)


def _pack_batch(
    bundle: _CheckpointBundle,
    values: np.ndarray,
    fields: list[str] | tuple[str, ...] | None,
    unpack: bool,
) -> dict[str, np.ndarray] | np.ndarray:
    names = list(bundle.output_names)
    idx = np.arange(len(names)) if fields is None else np.asarray([names.index(name) for name in fields])
    selected = values[:, idx]
    if not unpack:
        return selected.reshape(selected.shape[0], -1)
    return {names[i]: selected[:, j] for j, i in enumerate(idx)}


def _pack_sample_batch(
    bundle: _CheckpointBundle,
    values: np.ndarray,
    fields: list[str] | tuple[str, ...] | None,
    unpack: bool,
) -> dict[str, np.ndarray] | np.ndarray:
    names = list(bundle.output_names)
    idx = np.arange(len(names)) if fields is None else np.asarray([names.index(name) for name in fields])
    selected = values[:, :, idx]
    if not unpack:
        return selected.reshape(selected.shape[0], selected.shape[1], -1)
    return {names[i]: selected[:, :, j] for j, i in enumerate(idx)}


def _concat_batch(parts):
    if isinstance(parts[0], dict):
        return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    return np.concatenate(parts, axis=0)


def _concat_sample_batch(parts):
    if isinstance(parts[0], dict):
        return {key: np.concatenate([part[key] for part in parts], axis=1) for key in parts[0]}
    return np.concatenate(parts, axis=1)


def _chunks(n: int, batch_size: int | None):
    step = n if batch_size is None else int(batch_size)
    for start in range(0, n, step):
        yield start, min(start + step, n)


def _state(bundle: _CheckpointBundle) -> dict[str, np.ndarray]:
    if bundle.gplfr_state is None:
        raise ValueError("checkpoint is not a GPLFR bundle")
    return bundle.gplfr_state


def _check_variance_enabled(state: dict[str, np.ndarray]) -> None:
    missing = [name for name in ("sigma", "nugget_noise_by_sim") if name not in state]
    missing.extend(
        key
        for i in range(int(state["n_decoder_states"].item()))
        for key in (f"A_chol_{i}", f"U_{i}", f"K_chol_{i}")
        if key not in state
    )
    if missing:
        raise NotImplementedError(f"GPLFR bundle does not contain predictive-variance state: missing {missing}")


def _coerce_inputs_batch(bundle: _CheckpointBundle, inputs: BatchInputs) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted({name for row in ([inputs] if isinstance(inputs, dict) else inputs) for name in bundle.input_names if name not in row})
    if missing:
        raise ValueError(f"planet inputs are missing required keys {missing}; expected {bundle.input_names}")
    if isinstance(inputs, dict):
        n = len(np.atleast_1d(inputs[bundle.input_names[0]]))
        x = np.column_stack([np.broadcast_to(np.asarray(inputs[name], dtype=np.float64), n) for name in bundle.input_names])
        labels = np.broadcast_to(np.asarray(inputs.get("GCM", _DEFAULT_GCM)), n)
    else:
        x = np.asarray([[row[name] for name in bundle.input_names] for row in inputs], dtype=np.float64)
        labels = np.asarray([row.get("GCM", _DEFAULT_GCM) for row in inputs])
    _validate_inputs(bundle, x)
    _warn_extended_domain(bundle, x)
    for name, scale in _INPUT_UNIT_SCALE.items():
        if name in bundle.input_names:
            x[:, bundle.input_names.index(name)] *= scale
    return x, np.asarray([_sim_index(bundle, str(label)) for label in labels], dtype=np.int64)


def _validate_inputs(bundle: _CheckpointBundle, x: np.ndarray) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError("all planet inputs must be finite")
    positive = x[:, [bundle.input_names.index(name) for name in _POSITIVE_INPUTS if name in bundle.input_names]]
    if np.any(positive <= 0.0):
        raise ValueError(f"{_POSITIVE_INPUTS} must be > 0")
    if not {"CO2", "CH4"} <= set(bundle.input_names):
        return
    co2 = x[:, bundle.input_names.index("CO2")]
    ch4 = x[:, bundle.input_names.index("CH4")]
    if np.any(co2 < 0.0) or np.any(ch4 < 0.0):
        raise ValueError("CO2 and CH4 volume fractions must be non-negative")
    if np.any(co2 + ch4 > 1.0):
        raise ValueError("CO2 + CH4 must be <= 1")


def _warn_extended_domain(bundle: _CheckpointBundle, x: np.ndarray) -> None:
    for name, (lo, hi) in EXTENDED_DOMAIN.items():
        if name not in bundle.input_names:
            continue
        values = x[:, bundle.input_names.index(name)]
        if np.any((values < lo) | (values > hi)):
            warnings.warn(
                f"{name}={_value_range(values)} is outside CLERO's extended domain "
                f"[{lo:g}, {hi:g}] from SCOPE.md; predictions may be unreliable.",
                UserWarning,
                stacklevel=3,
            )


def _value_range(values: np.ndarray) -> str:
    lo, hi = float(np.min(values)), float(np.max(values))
    return f"{lo:g}" if lo == hi else f"batch range {lo:g}..{hi:g}"


def _sim_index(bundle: _CheckpointBundle, name: str) -> int:
    labels = list(bundle.manifest["gcm_labels"])
    label_map = {label.lower(): i for i, label in enumerate(labels)}
    key = name.lower()
    if key not in label_map:
        raise ValueError(f"unknown GCM {name!r}; expected one of {labels}")
    return label_map[key]


def _base_var(name: str) -> str:
    head, _, tail = name.rpartition("_")
    return head if tail.isdigit() else name


def _preprocess(x: np.ndarray, strategy: str, kwargs: dict[str, Any]) -> np.ndarray:
    if strategy == "Z-scaling":
        return x
    if strategy == "log_Z-scaling":
        return np.log(np.maximum(x, float(kwargs.get("epsilon", 1.0e-30))))
    if strategy == "arcsinh_Z-scaling":
        return np.arcsinh(x / float(kwargs["s"]))
    if strategy == "smoothed_logit_Z-scaling":
        eps = float(kwargs["epsilon"])
        y = (x + eps) / (1.0 + 2.0 * eps)
        return np.log(y / (1.0 - y))
    raise ValueError(f"unsupported preprocessing strategy {strategy!r}")


def _inverse_preprocess(x: np.ndarray, strategy: str, kwargs: dict[str, Any]) -> np.ndarray:
    if strategy == "Z-scaling":
        return x
    if strategy == "log_Z-scaling":
        return np.exp(x)
    if strategy == "arcsinh_Z-scaling":
        return np.sinh(x) * float(kwargs["s"])
    if strategy == "smoothed_logit_Z-scaling":
        eps = float(kwargs["epsilon"])
        y = 1.0 / (1.0 + np.exp(-x))
        return np.clip(y * (1.0 + 2.0 * eps) - eps, 0.0, 1.0)
    raise ValueError(f"unsupported preprocessing strategy {strategy!r}")


def _transform_inputs(bundle: _CheckpointBundle, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None]
    cols = []
    transforms = bundle.manifest["input_transforms"]
    for i, name in enumerate(bundle.input_names):
        spec = transforms[name]
        y = _preprocess(x[:, i], spec["strategy"], spec["kwargs"])
        cols.append((y - float(spec["mean"])) / float(spec["std"]))
    return np.column_stack(cols).astype(np.float64, copy=False)


def _kernel(kernel: str, X1: np.ndarray, ell: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
    X2 = X1 if X2 is None else X2
    diff = X1[:, None, :] - X2[None, :, :]
    r2 = np.sum(diff**2 / ell[None, None, :] ** 2, axis=-1)
    if kernel == "rbf":
        return np.exp(-0.5 * r2)
    r = np.sqrt(r2 + np.finfo(r2.dtype).eps)
    if kernel == "matern32":
        z = math.sqrt(3.0) * r
        return (1.0 + z) * np.exp(-z)
    if kernel == "matern52":
        z = math.sqrt(5.0) * r
        return (1.0 + z + (5.0 / 3.0) * r2) * np.exp(-z)
    raise ValueError(f"unknown GPLFR kernel {kernel!r}")


def _cross_kernel(state: dict[str, np.ndarray], x: np.ndarray, s: np.ndarray) -> np.ndarray:
    K_new_train = _kernel(str(state["kernel"].item()), x, state["ell"], state["X_train"])
    return K_new_train * state["Ks"][s[:, None], state["s_train"][None]]


def _latent_mean(state: dict[str, np.ndarray], x: np.ndarray, s: np.ndarray) -> np.ndarray:
    return _cross_kernel(state, x, s) @ state["K_inv_Z"]


def _latent_stats(state: dict[str, np.ndarray], x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K_new_train = _cross_kernel(state, x, s)
    alpha = solve_triangular(state["L_K"], K_new_train.T, lower=True)  # k^T K^-1 k = ||L^-1 k||^2
    k_new_new = state["Ks"][s, s] + state["nugget_noise_by_sim"][s]
    base_var = np.maximum(k_new_new - np.sum(np.square(alpha), axis=0), 0.0)
    return K_new_train @ state["K_inv_Z"], base_var[:, None] * np.square(state["sigma_f"])[None]


def _decoder_mean(bundle: _CheckpointBundle, state: dict[str, np.ndarray], z: np.ndarray) -> np.ndarray:
    n, n_coeffs, n_fields = int(z.shape[0]), int(state["n_coeffs"].item()), int(state["n_fields"].item())
    out = np.zeros((n, n_coeffs, n_fields), dtype=np.float64)
    v = z * state["tau"][None]
    for i in range(int(state["n_decoder_states"].item())):
        a = state[f"a_idxs_{i}"].astype(np.int64)
        f = state[f"f_idxs_{i}"].astype(np.int64)
        mu_w = state[f"mu_w_{i}"]
        out[:, a[:, None], f[None, :]] = np.einsum("nq,fqa->naf", v, mu_w)
    return out / state.get("alpha_sqrt", np.ones((n_coeffs, n_fields), dtype=np.float64))[None]


def _decoder_sample(bundle: _CheckpointBundle, state: dict[str, np.ndarray], z: np.ndarray, rng: np.random.Generator, sample_residual: bool = False) -> np.ndarray:
    n_samples, n, n_coeffs, n_fields = int(z.shape[0]), int(z.shape[1]), int(state["n_coeffs"].item()), int(state["n_fields"].item())
    out = np.zeros((n_samples, n, n_coeffs, n_fields), dtype=np.float64)
    v = z * state["tau"][None, None]
    sigma_sq = float(state["sigma"].item()) ** 2 + float(state.get("jitter", np.asarray(1.0e-6)).item())
    for i in range(int(state["n_decoder_states"].item())):
        a = state[f"a_idxs_{i}"].astype(np.int64)
        f = state[f"f_idxs_{i}"].astype(np.int64)
        mu = np.einsum("sbq,fqa->sbaf", v, state[f"mu_w_{i}"])
        if sample_residual:
            if state[f"U_{i}"].shape[1]:
                raise NotImplementedError("residual sampling currently supports only decoder_field_coreg_rank=0 bundles")
            if f"A_inv_{i}" not in state:
                A_chol = state[f"A_chol_{i}"]
                eye = np.eye(A_chol.shape[-1], dtype=A_chol.dtype)
                state[f"A_inv_{i}"] = np.stack([np.linalg.solve(L.T, np.linalg.solve(L, eye)) for L in A_chol])
            var = np.einsum("fqk,sbq,sbk->sbf", state[f"A_inv_{i}"], v, v) + sigma_sq
            mu = mu + np.sqrt(np.maximum(var, 0.0))[:, :, None] * rng.standard_normal(mu.shape)
        out[:, :, a[:, None], f[None, :]] = mu
    return out / state.get("alpha_sqrt", np.ones((n_coeffs, n_fields), dtype=np.float64))[None, None]


def _decoder_mean_and_variance(bundle: _CheckpointBundle, state: dict[str, np.ndarray], z: np.ndarray, z_var: np.ndarray, split: bool = False) -> tuple[np.ndarray, ...]:
    n, n_coeffs, n_fields = int(z.shape[0]), int(state["n_coeffs"].item()), int(state["n_fields"].item())
    mean = np.zeros((n, n_coeffs, n_fields), dtype=np.float64)
    var_coherent = np.zeros_like(mean)
    var_residual = np.zeros_like(mean)
    v = z * state["tau"][None]
    v_var = z_var * np.square(state["tau"])[None]
    sigma_sq = float(state["sigma"].item()) ** 2 + float(state.get("jitter", np.asarray(1.0e-6)).item())
    for i in range(int(state["n_decoder_states"].item())):
        a = state[f"a_idxs_{i}"].astype(np.int64)
        f = state[f"f_idxs_{i}"].astype(np.int64)
        mu_w = state[f"mu_w_{i}"]
        mean[:, a[:, None], f[None, :]] = np.einsum("nq,fqa->naf", v, mu_w)
        coherent, residual = _decoder_group_variance(state, i, v, v_var, mu_w, sigma_sq)
        var_coherent[:, a[:, None], f[None, :]] = coherent
        var_residual[:, a[:, None], f[None, :]] = residual
    alpha_sq = np.square(state.get("alpha_sqrt", np.ones((n_coeffs, n_fields), dtype=np.float64)))[None]
    if split:
        return mean / np.sqrt(alpha_sq), var_coherent / alpha_sq, var_residual / alpha_sq
    return mean / np.sqrt(alpha_sq), (var_coherent + var_residual) / alpha_sq


def _decoder_group_variance(
    state: dict[str, np.ndarray],
    i: int,
    v: np.ndarray,
    v_var: np.ndarray,
    mu_w: np.ndarray,
    sigma_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    U = state[f"U_{i}"]
    if U.shape[1]:
        raise NotImplementedError("analytic diagonal variance currently supports decoder_field_coreg_rank=0 bundles")
    latent_var = np.einsum("nq,fqa->naf", v_var, np.square(mu_w))
    if f"A_inv_{i}" not in state:
        A_chol = state[f"A_chol_{i}"]
        eye = np.eye(A_chol.shape[-1], dtype=A_chol.dtype)
        state[f"A_inv_{i}"] = np.stack([np.linalg.solve(L.T, np.linalg.solve(L, eye)) for L in A_chol])
    A_inv = state[f"A_inv_{i}"]
    vv = v[:, :, None] * v[:, None, :] + np.eye(v.shape[1])[None] * v_var[:, :, None]
    decoder_var = np.einsum("fqk,nqk->nf", A_inv, vv) + sigma_sq
    return latent_var, np.broadcast_to(decoder_var[:, None, :], latent_var.shape)


def _design_matrix(bundle: _CheckpointBundle, X: np.ndarray, s: np.ndarray) -> np.ndarray:
    cfg = bundle.manifest["linear_trend"]["design_cfg"]
    cols = []
    if cfg.get("intercept", True):
        cols.append(np.ones((X.shape[0], 1), dtype=X.dtype))
    if cfg.get("inputs", True):
        cols.append(X)
    if cfg.get("sim_onehot", False):
        oh = np.eye(len(bundle.manifest["gcm_labels"]), dtype=X.dtype)[s]
        if cfg.get("intercept", True):
            oh = oh[:, 1:]
        if oh.size:
            cols.append(oh)
    return np.concatenate(cols, axis=1)


def _unnormalise_spectral(bundle: _CheckpointBundle, state: dict[str, np.ndarray], coeffs_n: np.ndarray) -> np.ndarray:
    out = np.zeros_like(coeffs_n, dtype=np.float32)
    raw_names = bundle.manifest["raw_output_names"]
    for j, name in enumerate(raw_names):
        mask = state[f"spectral_mask_{j}"].astype(bool)
        out[:, j, mask] = coeffs_n[:, j, mask] * float(state["spectral_sigma"][j]) + state[f"spectral_mean_{j}"]
    return out


def _spectral_grid_mean(bundle: _CheckpointBundle, state: dict[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    coeffs_norm = np.transpose(y, (0, 2, 1)).astype(np.float32, copy=False)
    coeffs = _unnormalise_spectral(bundle, state, coeffs_norm)
    return (coeffs @ state["inverse_sht"].T).reshape(y.shape[0], coeffs.shape[1], *bundle.manifest["grid_shape"])


def _spectral_grid_variance(bundle: _CheckpointBundle, state: dict[str, np.ndarray], y_var: np.ndarray) -> np.ndarray:
    var_coeffs = np.zeros((y_var.shape[0], y_var.shape[2], y_var.shape[1]), dtype=np.float32)
    for j, _ in enumerate(bundle.manifest["raw_output_names"]):
        mask = state[f"spectral_mask_{j}"].astype(bool)
        var_coeffs[:, j, mask] = y_var[:, mask, j] * float(state["spectral_sigma"][j]) ** 2
    return (var_coeffs @ np.square(state["inverse_sht"].T)).astype(np.float32, copy=False)


def _inverse_preprocess_outputs(bundle: _CheckpointBundle, y: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
    out = y.astype(np.float32, copy=True)
    transforms = bundle.manifest["output_transforms"]
    f_star_idx = bundle.input_names.index("F_star")
    f_star = x_raw[:, f_star_idx][:, None, None]
    for j, name in enumerate(bundle.manifest["raw_output_names"]):
        spec = transforms[name]
        channel = _inverse_preprocess(out[:, j], spec["strategy"], spec["kwargs"])
        if _base_var(name) == "specific_humidity":
            channel = np.clip(channel, 0.0, 1.0)
        if bundle.manifest.get("asr_olr_normalize_by_f_star", False) and _base_var(name) in ASR_OLR_FIELDS:
            channel = channel * f_star
        out[:, j] = channel
    return out
