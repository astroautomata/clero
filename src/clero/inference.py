"""The Emulator class."""

from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path

import numpy as np

from ._checkpoint import _load_checkpoint
from ._gplfr_runtime import (
    BatchInputs,
    _is_batch,
    outputs_to_model,
    outputs_to_physical,
    predict_gplfr_batch,
    predict_gplfr_samples_batch,
)


class Emulator:
    """The CLERO emulator. Loads the shipped model (the exact weights used in the paper).

    `predict` gives the best point estimate of a planet's climate and `sample` draws from
    the climate distribution. Both accept one planet or a batch, and return fields in
    physical units by default or in model space with `space="model"` (see UNCERTAINTY.md).

    Args:
        bundle: path to an alternative model directory (containing `manifest.json` and
            `gplfr_state.npz`). None (default) uses the shipped model.
        device: "cpu" (default) runs in NumPy; a torch device such as "cuda" runs on the GPU.
        dtype: "float32" (default) or "float64"; used only on GPU devices.
    """

    def __init__(self, *, bundle: str | Path | None = None, device: str = "cpu", dtype: str = "float32"):
        self._bundle = _load_checkpoint(bundle)
        self.device = device
        self.dtype = dtype
        self._torch_states = {}

    @property
    def output_names(self) -> list[str]:
        """The 53 output field names, e.g. "surface_temperature", "temperature_0"."""
        return self._bundle.output_names

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """(32, 64): the number of latitude and longitude cells in each output field."""
        return tuple(int(x) for x in self._bundle.manifest["grid_shape"])

    @cached_property
    def bundle_sha256(self) -> str:
        """sha256 checksum of the loaded weights file, so results can be tied to an exact model."""
        return hashlib.sha256((self._bundle.root / "gplfr_state.npz").read_bytes()).hexdigest()

    def predict(
        self,
        inputs: dict[str, float] | BatchInputs,
        *,
        space: str = "physical",
        return_variance: bool = False,
        split_variance: bool = False,
        include_residual: bool = False,
        device: str | None = None,
        batch_size: int | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        unpack: bool = True,
    ):
        """Best point estimate of the climate for one planet or a batch.

        Args:
            inputs: a planet dict with keys radius, gravity, P_rot, P0, CO2, CH4, F_star, T_star
                and optionally GCM (see the README for units). A batch is a list of such dicts,
                or a dict of equal-length sequences such as `{"F_star": [...], "GCM": [...]}`.
            space: "physical" (default) returns natural units; "model" returns the transformed
                space in which the climate distribution is Gaussian (see UNCERTAINTY.md).
            return_variance: also return the variance of each field at every grid cell. Only
                available with `space="model"`. Like `sample`'s default draws, this excludes the
                spatially white residual term.
            include_residual: add the spatially white residual term to the variance; the
                analogue of `sample_residual=True` in `sample` (see UNCERTAINTY.md).
            split_variance: return the coherent and residual variances separately instead
                (implies `return_variance`); their sum is the total.
            device: override the device chosen at construction for this call.
            batch_size: planets per chunk (default 256 on CPU, 512 on GPU).
            fields: subset of output field names to compute and return. None means all 53.
            unpack: if True (default) return dicts keyed by field name; if False return flat arrays.

        Returns:
            A dict of `(32, 64)` fields, or `(n_planets, 32, 64)` for a batch. With
            `return_variance` a `(mean, variance)` pair; with `split_variance` a
            `(mean, coherent_variance, residual_variance)` triple.
        """
        batch = _is_batch(inputs)
        items = inputs if batch else [inputs]
        device = self.device if device is None else device
        if device == "cpu":
            result = predict_gplfr_batch(self._bundle, items, space=space, return_variance=return_variance, split_variance=split_variance, include_residual=include_residual, batch_size=batch_size or 256, fields=fields, unpack=unpack)
        else:
            from ._torch_runtime import predict_gplfr_batch_torch

            result = predict_gplfr_batch_torch(self._bundle, items, state=self._torch_state(device), device=device, dtype=self.dtype, space=space, return_variance=return_variance, split_variance=split_variance, include_residual=include_residual, batch_size=batch_size or 512, fields=fields, unpack=unpack)
        return result if batch else _drop_batch_axis(result)

    def sample(
        self,
        inputs: dict[str, float] | BatchInputs,
        *,
        n_samples: int = 64,
        seed: int | None = None,
        sample_residual: bool = False,
        space: str = "physical",
        device: str | None = None,
        batch_size: int | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        unpack: bool = True,
    ):
        """Draw climates from CLERO's predictive distribution for one planet or a batch.

        Each draw is a complete, spatially coherent set of climate fields, so any
        quantity you compute from a draw (a global mean, an ice fraction, a map) inherits
        a calibrated uncertainty when you repeat it across draws. See UNCERTAINTY.md for
        the spatially white residual (`sample_residual`) and for summarising humidity
        and cloud fraction in `space="model"`.

        Args:
            inputs: a planet dict or a batch, as in `predict`.
            n_samples: number of draws (default 64).
            seed: random seed for reproducible draws; None uses fresh randomness.
            sample_residual: add the spatially white residual scatter to each draw (default False).
            space: "physical" (default) or "model", as in `predict`.
            device, fields, unpack: as in `predict`.
            batch_size: planets per chunk, as in `predict`. Peak memory scales with
                `batch_size * n_samples`, so lower `batch_size` when raising `n_samples`.

        Returns:
            A dict of `(n_samples, 32, 64)` fields, or `(n_samples, n_planets, 32, 64)` for a batch.
        """
        batch = _is_batch(inputs)
        items = inputs if batch else [inputs]
        device = self.device if device is None else device
        if device == "cpu":
            result = predict_gplfr_samples_batch(self._bundle, items, space=space, n_samples=n_samples, seed=seed, sample_residual=sample_residual, batch_size=batch_size or 256, fields=fields, unpack=unpack)
        else:
            from ._torch_runtime import predict_gplfr_samples_batch_torch

            result = predict_gplfr_samples_batch_torch(self._bundle, items, state=self._torch_state(device), device=device, dtype=self.dtype, space=space, n_samples=n_samples, seed=seed, sample_residual=sample_residual, batch_size=batch_size or 512, fields=fields, unpack=unpack)
        return result if batch else _drop_sample_batch_axis(result)

    def to_physical(
        self,
        field_or_fields: str | dict[str, np.ndarray],
        values_or_inputs: np.ndarray | dict[str, float] | BatchInputs,
        inputs: dict[str, float] | BatchInputs | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Convert fields from model space to physical units (the inverse of `to_model`).

        Use `to_physical(fields, inputs)` for a dict of fields, or
        `to_physical("field_name", values, inputs)` for one array. `inputs` is the planet
        the fields belong to; it is needed because ASR and OLR are stored relative to F_star.
        """
        fields, inputs, name = _transform_args(field_or_fields, values_or_inputs, inputs)
        out = outputs_to_physical(self._bundle, fields, inputs)
        return out if name is None else out[name]

    def to_model(
        self,
        field_or_fields: str | dict[str, np.ndarray],
        values_or_inputs: np.ndarray | dict[str, float] | BatchInputs,
        inputs: dict[str, float] | BatchInputs | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Convert fields from physical units to model space (see UNCERTAINTY.md).

        Use `to_model(fields, inputs)` for a dict of fields, or
        `to_model("field_name", values, inputs)` for one array.
        """
        fields, inputs, name = _transform_args(field_or_fields, values_or_inputs, inputs)
        out = outputs_to_model(self._bundle, fields, inputs)
        return out if name is None else out[name]

    def _torch_state(self, device: str):
        from ._torch_runtime import make_torch_state

        key = (device, self.dtype)
        if key not in self._torch_states:
            self._torch_states[key] = make_torch_state(self._bundle, device=device, dtype=self.dtype)
        return self._torch_states[key]


def _drop_batch_axis(result):
    if isinstance(result, tuple):
        return tuple(_drop_batch_axis(part) for part in result)
    if isinstance(result, dict):
        return {key: value[0] for key, value in result.items()}
    return result[0]


def _drop_sample_batch_axis(result):
    if isinstance(result, dict):
        return {key: value[:, 0] for key, value in result.items()}
    return result[:, 0]


def _transform_args(field_or_fields, values_or_inputs, inputs):
    if isinstance(field_or_fields, str):
        if inputs is None:
            raise TypeError("single-field transforms require (name, values, inputs)")
        return {field_or_fields: values_or_inputs}, inputs, field_or_fields
    if inputs is not None:
        raise TypeError("dict transforms use (fields, inputs), without a third argument")
    if not isinstance(field_or_fields, dict):
        raise TypeError("expected a field name or a dict of fields")
    return field_or_fields, values_or_inputs, None
