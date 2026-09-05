"""prediction API"""

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
    """CLERO climate emulator. Loads the bundled model (the exact weights used in the paper) on init.

    Predictions can be returned in two spaces:

    - ``"physical"`` (default) — fields in their natural units.
    - ``"model"`` — the space the emulator is Gaussian in (log humidity, logit cloud
      fraction, ...). Predictive variance is only meaningful here.

    Args:
        bundle: path to an alternative bundle directory (``manifest.json`` + ``gplfr_state.npz``);
            None (default) uses the shipped ``_model_bundle``.
        device: "cpu" for the numpy path, or a torch device like "cuda" / "cuda:0".
        dtype: torch dtype name, used only on non-cpu devices.
    """

    def __init__(self, *, bundle: str | Path | None = None, device: str = "cpu", dtype: str = "float32"):
        self._bundle = _load_checkpoint(bundle)
        self.device = device
        self.dtype = dtype
        self._torch_states = {}

    @property
    def output_names(self) -> list[str]:
        """Field names produced by the emulator."""
        return self._bundle.output_names

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """(n_lat, n_lon) of the predicted grids."""
        return tuple(int(x) for x in self._bundle.manifest["grid_shape"])

    @cached_property
    def bundle_sha256(self) -> str:
        """sha256 of the loaded ``gplfr_state.npz`` (weights provenance; compare with README)."""
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
        """Predict the climate for one planet or a batch.

        Args:
            inputs: a planet dict (T_star, F_star, radius, gravity, P_rot, P0, CO2, CH4, GCM),
                or a batch as a list of such dicts or a column dict (e.g. {"F_star": [...], "GCM": [...]}).
            space: "physical" (default) or "model".
            return_variance: also return the per-cell predictive variance (spatially coherent part
                only, matching ``sample``'s default draws). Requires space="model" (variance is not
                meaningful in physical space for nonlinearly-transformed fields).
            include_residual: add the spatially white residual term to the variance, the analogue of
                ``sample_residual=True`` (see UNCERTAINTY.md).
            split_variance: return the coherent and residual variances separately (implies
                ``return_variance``); total = coherent + residual.
            device / batch_size: override device and chunk size for this call.
            fields: subset of output fields. None means all.
            unpack: if True, dict(s) keyed by field name; else flat arrays.

        Returns:
            A field dict (or flat array); a (mean, variance) pair if return_variance;
            a (mean, coherent_variance, residual_variance) triple if split_variance.
            Single-planet inputs drop the batch axis.
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
        the spatially white residual (``sample_residual``) and for summarising humidity
        and cloud fraction in ``space="model"``.

        Args:
            inputs: a planet dict or a batch (see ``predict``).
            n_samples: number of draws.
            seed: random seed; None uses fresh randomness.
            sample_residual: add the spatially white residual scatter to each draw (default False).
            space: "physical" (default) or "model".
            device / fields / unpack: as in ``predict``.
            batch_size: planets per chunk, as in ``predict``. Peak memory scales as
                batch_size * n_samples, so lower batch_size when raising n_samples.

        Returns:
            A field dict (or flat array) with the sample axis first. Single-planet inputs
            drop the batch axis, giving (n_samples, 32, 64) per field.
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
        """Map model-space fields to physical units (inverse of to_model).

        Use ``to_physical(fields, inputs)`` for a dict, or
        ``to_physical("field_name", values, inputs)`` for one field. ``inputs`` supplies
        F_star, needed because ASR/OLR are F_star-normalised.
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
        """Map physical fields to model space, where the predictive is Gaussian.

        Use ``to_model(fields, inputs)`` for a dict, or
        ``to_model("field_name", values, inputs)`` for one field.
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
