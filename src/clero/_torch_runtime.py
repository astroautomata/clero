"""Optional Torch runtime for large CLERO batches."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._checkpoint import _CheckpointBundle
from ._gplfr_runtime import ASR_OLR_FIELDS, BatchInputs, _base_var, _chunks, _coerce_inputs_batch, _concat_batch, _concat_sample_batch, _pack_batch, _pack_sample_batch


def make_torch_state(bundle: _CheckpointBundle, *, device: str = "cuda", dtype: str = "float32") -> dict[str, Any]:
    torch = _torch()
    tdtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    state: dict[str, Any] = {}
    for key, value in bundle.gplfr_state.items():
        if value.dtype.kind in "USO":
            state[key] = value.item() if value.shape == () else value
        elif value.dtype.kind == "b":
            state[key] = torch.as_tensor(value, device=device, dtype=torch.bool)
        elif value.dtype.kind in "iu":
            state[key] = torch.as_tensor(value, device=device, dtype=torch.long)
        else:
            state[key] = torch.as_tensor(value, device=device, dtype=tdtype)
    return state


def predict_gplfr_batch_torch(
    bundle: _CheckpointBundle,
    inputs: BatchInputs,
    *,
    state: dict[str, Any] | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    space: str = "physical",
    return_variance: bool = False,
    split_variance: bool = False,
    include_residual: bool = False,
    batch_size: int | None = 512,
    fields: list[str] | tuple[str, ...] | None = None,
    unpack: bool = True,
):
    torch = _torch()
    state = make_torch_state(bundle, device=device, dtype=dtype) if state is None else state
    x_raw, s = _coerce_inputs_batch(bundle, inputs)
    return_variance = return_variance or split_variance
    parts = []
    with torch.inference_mode():
        for start, stop in _chunks(x_raw.shape[0], batch_size):
            xb = _tensor(torch, x_raw[start:stop], device, dtype)
            sb = _tensor(torch, s[start:stop], device, "long")
            if return_variance:
                arrays = _predict_mean_and_variance(bundle, state, xb, sb, split_variance, include_residual)
                parts.append(tuple(_pack_batch(bundle, arr.cpu().numpy(), fields, unpack) for arr in arrays))
            else:
                values = _predict_grid(bundle, state, xb, sb, space)
                parts.append((_pack_batch(bundle, values.cpu().numpy(), fields, unpack),))
    concat = tuple(_concat_batch(list(group)) for group in zip(*parts))
    return concat if return_variance else concat[0]


def predict_gplfr_samples_batch_torch(
    bundle: _CheckpointBundle,
    inputs: BatchInputs,
    *,
    state: dict[str, Any] | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    space: str = "physical",
    n_samples: int = 64,
    seed: int | None = None,
    sample_residual: bool = False,
    batch_size: int | None = 512,
    fields: list[str] | tuple[str, ...] | None = None,
    unpack: bool = True,
) -> dict[str, np.ndarray] | np.ndarray:
    torch = _torch()
    state = make_torch_state(bundle, device=device, dtype=dtype) if state is None else state
    generator = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
    x_raw, s = _coerce_inputs_batch(bundle, inputs)
    parts = []
    with torch.inference_mode():
        for start, stop in _chunks(x_raw.shape[0], batch_size):
            values = _predict_samples(bundle, state, _tensor(torch, x_raw[start:stop], device, dtype), _tensor(torch, s[start:stop], device, "long"), n_samples, generator, space, sample_residual)
            parts.append(_pack_sample_batch(bundle, values.cpu().numpy(), fields, unpack))
    return _concat_sample_batch(parts)


def _torch():
    import torch

    return torch


def _tensor(torch, value: np.ndarray, device: str, dtype: str):
    return torch.as_tensor(value, device=device, dtype=torch.long if dtype == "long" else {"float32": torch.float32, "float64": torch.float64}[dtype])


def _predict_grid(bundle: _CheckpointBundle, state: dict[str, Any], x_raw, s, space: str = "physical"):
    x = _transform_inputs(bundle, x_raw)
    z_mean = _latent_mean(state, x, s)
    y = _decoder_mean(bundle, state, z_mean)
    if "linear_Gamma" in state:
        y = y + _torch().einsum("np,paf->naf", _design_matrix(bundle, x, s), state["linear_Gamma"])
    y = y * state.get("sh_mask")[None]
    grid = _spectral_grid_mean(bundle, state, y)
    return grid if space == "model" else _inverse_preprocess_outputs(bundle, grid, x_raw)


def _predict_mean_and_variance(bundle: _CheckpointBundle, state: dict[str, Any], x_raw, s, split: bool = False, include_residual: bool = False):
    x = _transform_inputs(bundle, x_raw)
    z_mean, z_var = _latent_stats(state, x, s)
    y_mean, coherent, residual = _decoder_mean_and_variance(bundle, state, z_mean, z_var)
    y_vars = (coherent, residual) if split else (coherent + residual if include_residual else coherent,)
    if "linear_Gamma" in state:
        y_mean = y_mean + _torch().einsum("np,paf->naf", _design_matrix(bundle, x, s), state["linear_Gamma"])
    mask = state["sh_mask"][None]
    mean = _spectral_grid_mean(bundle, state, y_mean * mask)
    return (mean, *(_spectral_grid_variance(bundle, state, y_var * mask).reshape(mean.shape) for y_var in y_vars))


def _predict_samples(bundle: _CheckpointBundle, state: dict[str, Any], x_raw, s, n_samples: int, generator, space: str = "physical", sample_residual: bool = False):
    torch = _torch()
    x = _transform_inputs(bundle, x_raw)
    z_mean, z_var = _latent_stats(state, x, s)
    z = z_mean[None] + torch.sqrt(z_var)[None] * torch.randn((int(n_samples), *z_mean.shape), device=z_mean.device, dtype=z_mean.dtype, generator=generator)
    y = _decoder_sample(bundle, state, z, generator, sample_residual)
    if "linear_Gamma" in state:
        y = y + torch.einsum("np,paf->naf", _design_matrix(bundle, x, s), state["linear_Gamma"])[None]
    y = y * state["sh_mask"][None, None]
    flat = _spectral_grid_mean(bundle, state, y.reshape(-1, y.shape[2], y.shape[3]))
    grid = flat if space == "model" else _inverse_preprocess_outputs(bundle, flat, x_raw.repeat((int(n_samples), 1)))
    return grid.reshape(int(n_samples), x_raw.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]).float()


def _transform_inputs(bundle: _CheckpointBundle, x):
    cols = []
    transforms = bundle.manifest["input_transforms"]
    for i, name in enumerate(bundle.input_names):
        spec = transforms[name]
        y = _preprocess(x[:, i], spec["strategy"], spec["kwargs"])
        cols.append((y - float(spec["mean"])) / float(spec["std"]))
    return _torch().stack(cols, dim=1)


def _preprocess(x, strategy: str, kwargs: dict[str, Any]):
    torch = _torch()
    if strategy == "Z-scaling":
        return x
    if strategy == "log_Z-scaling":
        return torch.log(torch.clamp_min(x, float(kwargs.get("epsilon", 1.0e-30))))
    if strategy == "arcsinh_Z-scaling":
        return torch.asinh(x / float(kwargs["s"]))
    if strategy == "smoothed_logit_Z-scaling":
        eps = float(kwargs["epsilon"])
        y = (x + eps) / (1.0 + 2.0 * eps)
        return torch.log(y / (1.0 - y))
    raise ValueError(f"unsupported preprocessing strategy {strategy!r}")


def _inverse_preprocess(x, strategy: str, kwargs: dict[str, Any]):
    torch = _torch()
    if strategy == "Z-scaling":
        return x
    if strategy == "log_Z-scaling":
        return torch.exp(x)
    if strategy == "arcsinh_Z-scaling":
        return torch.sinh(x) * float(kwargs["s"])
    if strategy == "smoothed_logit_Z-scaling":
        eps = float(kwargs["epsilon"])
        y = 1.0 / (1.0 + torch.exp(-x))
        return torch.clamp(y * (1.0 + 2.0 * eps) - eps, 0.0, 1.0)
    raise ValueError(f"unsupported preprocessing strategy {strategy!r}")


def _kernel(kernel: str, X1, ell, X2=None):
    torch = _torch()
    X2 = X1 if X2 is None else X2
    r2 = torch.sum((X1[:, None, :] - X2[None, :, :]) ** 2 / ell[None, None, :] ** 2, dim=-1)
    if kernel == "rbf":
        return torch.exp(-0.5 * r2)
    r = torch.sqrt(r2 + torch.finfo(r2.dtype).eps)
    if kernel == "matern32":
        z = 3.0**0.5 * r
        return (1.0 + z) * torch.exp(-z)
    if kernel == "matern52":
        z = 5.0**0.5 * r
        return (1.0 + z + (5.0 / 3.0) * r2) * torch.exp(-z)
    raise ValueError(f"unknown GPLFR kernel {kernel!r}")


def _latent_mean(state: dict[str, Any], x, s):
    K_new_train = _kernel(str(state["kernel"]), x, state["ell"], state["X_train"])
    K_new_train = K_new_train * state["Ks"][s[:, None], state["s_train"][None]]
    return K_new_train @ state["K_inv_Z"]


def _latent_stats(state: dict[str, Any], x, s):
    torch = _torch()
    K_new_train = _kernel(str(state["kernel"]), x, state["ell"], state["X_train"])
    K_new_train = K_new_train * state["Ks"][s[:, None], state["s_train"][None]]
    mean = K_new_train @ state["K_inv_Z"]
    alpha = torch.linalg.solve_triangular(state["L_K"], K_new_train.T, upper=False)
    K_inv_k = torch.linalg.solve_triangular(state["L_K"].T, alpha, upper=True).T
    base_var = torch.clamp_min(state["Ks"][s, s] + state["nugget_noise_by_sim"][s] - torch.sum(K_new_train * K_inv_k, dim=1), 0.0)
    return mean, base_var[:, None] * torch.square(state["sigma_f"])[None]


def _decoder_mean(bundle: _CheckpointBundle, state: dict[str, Any], z):
    torch = _torch()
    n, n_coeffs, n_fields = z.shape[0], int(state["n_coeffs"].item()), int(state["n_fields"].item())
    out = torch.zeros((n, n_coeffs, n_fields), device=z.device, dtype=z.dtype)
    v = z * state["tau"][None]
    for i in range(int(state["n_decoder_states"].item())):
        a, f, mu_w = state[f"a_idxs_{i}"], state[f"f_idxs_{i}"], state[f"mu_w_{i}"]
        out[:, a[:, None], f[None, :]] = torch.einsum("nq,fqa->naf", v, mu_w)
    return out / state["alpha_sqrt"][None]


def _decoder_sample(bundle: _CheckpointBundle, state: dict[str, Any], z, generator, sample_residual: bool = False):
    torch = _torch()
    n_samples, n, n_coeffs, n_fields = z.shape[0], z.shape[1], int(state["n_coeffs"].item()), int(state["n_fields"].item())
    out = torch.zeros((n_samples, n, n_coeffs, n_fields), device=z.device, dtype=z.dtype)
    v = z * state["tau"][None, None]
    sigma_sq = float(state["sigma"].item()) ** 2 + float(state["jitter"].item())
    for i in range(int(state["n_decoder_states"].item())):
        a, f, mu_w = state[f"a_idxs_{i}"], state[f"f_idxs_{i}"], state[f"mu_w_{i}"]
        mu = torch.einsum("sbq,fqa->sbaf", v, mu_w)
        if sample_residual:
            if state[f"U_{i}"].shape[1]:
                raise NotImplementedError("residual sampling currently supports only decoder_field_coreg_rank=0 bundles")
            key = f"A_inv_{i}"
            if key not in state:
                state[key] = torch.cholesky_inverse(state[f"A_chol_{i}"])
            var = torch.einsum("fqk,sbq,sbk->sbf", state[key], v, v) + sigma_sq
            mu = mu + torch.sqrt(torch.clamp_min(var, 0.0))[:, :, None] * torch.randn(mu.shape, device=z.device, dtype=z.dtype, generator=generator)
        out[:, :, a[:, None], f[None, :]] = mu
    return out / state["alpha_sqrt"][None, None]


def _decoder_mean_and_variance(bundle: _CheckpointBundle, state: dict[str, Any], z, z_var):
    torch = _torch()
    n, n_coeffs, n_fields = z.shape[0], int(state["n_coeffs"].item()), int(state["n_fields"].item())
    mean = torch.zeros((n, n_coeffs, n_fields), device=z.device, dtype=z.dtype)
    var_coherent = torch.zeros_like(mean)
    var_residual = torch.zeros_like(mean)
    v = z * state["tau"][None]
    v_var = z_var * torch.square(state["tau"])[None]
    sigma_sq = float(state["sigma"].item()) ** 2 + float(state["jitter"].item())
    for i in range(int(state["n_decoder_states"].item())):
        a, f, mu_w = state[f"a_idxs_{i}"], state[f"f_idxs_{i}"], state[f"mu_w_{i}"]
        mean[:, a[:, None], f[None, :]] = torch.einsum("nq,fqa->naf", v, mu_w)
        coherent, residual = _decoder_group_variance(state, i, v, v_var, mu_w, sigma_sq)
        var_coherent[:, a[:, None], f[None, :]] = coherent
        var_residual[:, a[:, None], f[None, :]] = residual
    alpha_sq = torch.square(state["alpha_sqrt"])[None]
    return mean / torch.sqrt(alpha_sq), var_coherent / alpha_sq, var_residual / alpha_sq


def _decoder_group_variance(state: dict[str, Any], i: int, v, v_var, mu_w, sigma_sq: float):
    torch = _torch()
    if state[f"U_{i}"].shape[1]:
        raise NotImplementedError("analytic diagonal variance currently supports decoder_field_coreg_rank=0 bundles")
    latent_var = torch.einsum("nq,fqa->naf", v_var, torch.square(mu_w))
    key = f"A_inv_{i}"
    if key not in state:
        state[key] = torch.cholesky_inverse(state[f"A_chol_{i}"])
    vv = v[:, :, None] * v[:, None, :] + torch.eye(v.shape[1], device=v.device, dtype=v.dtype)[None] * v_var[:, :, None]
    decoder_var = torch.einsum("fqk,nqk->nf", state[key], vv) + sigma_sq
    return latent_var, decoder_var[:, None, :].expand_as(latent_var)


def _design_matrix(bundle: _CheckpointBundle, X, s):
    torch = _torch()
    cfg = bundle.manifest["linear_trend"]["design_cfg"]
    cols = []
    if cfg.get("intercept", True):
        cols.append(torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype))
    if cfg.get("inputs", True):
        cols.append(X)
    if cfg.get("sim_onehot", False):
        oh = torch.eye(len(bundle.manifest["gcm_labels"]), device=X.device, dtype=X.dtype)[s]
        if cfg.get("intercept", True):
            oh = oh[:, 1:]
        if oh.numel():
            cols.append(oh)
    return torch.cat(cols, dim=1)


def _unnormalise_spectral(bundle: _CheckpointBundle, state: dict[str, Any], coeffs_n):
    torch = _torch()
    out = torch.zeros_like(coeffs_n, dtype=torch.float32)
    for j, _ in enumerate(bundle.manifest["raw_output_names"]):
        mask = state[f"spectral_mask_{j}"].bool()
        out[:, j, mask] = coeffs_n[:, j, mask].float() * float(state["spectral_sigma"][j].item()) + state[f"spectral_mean_{j}"].float()
    return out


def _spectral_grid_mean(bundle: _CheckpointBundle, state: dict[str, Any], y):
    coeffs = _unnormalise_spectral(bundle, state, y.transpose(1, 2))
    return (coeffs @ state["inverse_sht"].float().T).reshape(y.shape[0], coeffs.shape[1], *bundle.manifest["grid_shape"])


def _spectral_grid_variance(bundle: _CheckpointBundle, state: dict[str, Any], y_var):
    torch = _torch()
    var_coeffs = torch.zeros((y_var.shape[0], y_var.shape[2], y_var.shape[1]), device=y_var.device, dtype=torch.float32)
    for j, _ in enumerate(bundle.manifest["raw_output_names"]):
        mask = state[f"spectral_mask_{j}"].bool()
        var_coeffs[:, j, mask] = y_var[:, mask, j].float() * float(state["spectral_sigma"][j].item()) ** 2
    return (var_coeffs @ torch.square(state["inverse_sht"].float().T)).float()


def _inverse_preprocess_outputs(bundle: _CheckpointBundle, y, x_raw):
    torch = _torch()
    out = y.float().clone()
    transforms = bundle.manifest["output_transforms"]
    f_star = x_raw[:, bundle.input_names.index("F_star")][:, None, None]
    for j, name in enumerate(bundle.manifest["raw_output_names"]):
        channel = _inverse_preprocess(out[:, j], transforms[name]["strategy"], transforms[name]["kwargs"])
        if _base_var(name) == "specific_humidity":
            channel = torch.clamp(channel, 0.0, 1.0)
        if bundle.manifest.get("asr_olr_normalize_by_f_star", False) and _base_var(name) in ASR_OLR_FIELDS:
            channel = channel * f_star
        out[:, j] = channel
    return out
