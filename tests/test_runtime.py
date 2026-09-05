from __future__ import annotations

import hashlib
import importlib.util
import json
import pkgutil
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import clero
import clero.inference
from clero._checkpoint import _CheckpointBundle
from clero._gplfr_runtime import outputs_to_physical, predict_gplfr_batch
from clero import CORE_DOMAIN, EARTH, EXTENDED_DOMAIN, M_EARTH, TRAPPIST1E, Emulator
from clero._checkpoint import _load_checkpoint
from clero.climate_analysis import global_mean, grid_records, latitude_centers, longitude_centers, profile_stats, summarize_outputs


def build_gplfr_bundle(root: Path) -> Path:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "model_type": "gplfr_map",
                "input_names": ["F_star"],
                "output_names": ["surface_temperature"],
                "raw_output_names": ["surface_temperature"],
                "grid_shape": [2, 2],
                "gcm_labels": ["um"],
                "input_transforms": {"F_star": {"strategy": "Z-scaling", "kwargs": {"stat_key_pattern": "F_star"}, "mean": 0.0, "std": 1.0}},
                "output_transforms": {"surface_temperature": {"strategy": "Z-scaling", "kwargs": {"stat_key_pattern": "surface_temperature"}}},
                "asr_olr_normalize_by_f_star": False,
                "linear_trend": {"design_cfg": {}},
            }
        )
    )
    np.savez(
        root / "gplfr_state.npz",
        kernel=np.asarray("matern52"),
        X_train=np.full((1, 1), 1000.0),
        s_train=np.zeros(1, dtype=np.int64),
        Z_train=np.full((1, 1), 2.0),
        L_K=np.ones((1, 1)),
        K_inv_Z=np.full((1, 1), 2.0),
        Ks=np.ones((1, 1)),
        ell=np.ones(1),
        sigma_f=np.ones(1),
        sigma=np.asarray(0.5),
        jitter=np.asarray(0.0),
        nugget_noise_by_sim=np.asarray([0.1]),
        tau=np.ones(1),
        sh_mask=np.ones((1, 1), dtype=bool),
        alpha_sqrt=np.ones((1, 1)),
        inverse_sht=np.ones((4, 1), dtype=np.float32),
        n_coeffs=np.asarray(1),
        n_fields=np.asarray(1),
        n_decoder_states=np.asarray(1),
        a_idxs_0=np.asarray([0]),
        f_idxs_0=np.asarray([0]),
        mu_w_0=np.asarray([[[3.0]]]),
        A_chol_0=np.asarray([[[1.0]]]),
        U_0=np.zeros((1, 0)),
        K_chol_0=np.zeros((0, 0)),
        spectral_mask_0=np.asarray([True]),
        spectral_mean_0=np.asarray([0.0], dtype=np.float32),
        spectral_sigma=np.asarray([1.0], dtype=np.float32),
    )
    return root


def test_private_load_checkpoint_resolves_gplfr_bundle(tmp_path: Path) -> None:
    bundle = _load_checkpoint(build_gplfr_bundle(tmp_path) / "manifest.json")

    assert bundle.root == tmp_path
    assert bundle.input_names == ["F_star"]
    assert bundle.output_names == ["surface_temperature"]
    assert "Z_train" in bundle.gplfr_state


def test_private_load_checkpoint_rejects_legacy_linear_bundle(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": "0.1.0", "input_names": []}))

    try:
        _load_checkpoint(tmp_path)
    except ValueError as exc:
        assert "gplfr_map" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("legacy non-GPLFR bundles should not load")


def test_top_level_implementation_modules_are_private() -> None:
    public_modules = {module.name for module in pkgutil.iter_modules(clero.__path__) if not module.name.startswith("_")}

    assert public_modules == {"climate_analysis", "domain", "inference", "orbit", "presets"}
    assert not hasattr(clero, "load_checkpoint")
    assert importlib.util.find_spec("clero.checkpoint") is None
    assert importlib.util.find_spec("clero.gplfr_runtime") is None


def test_bundled_radiative_fields_use_public_names() -> None:
    bundle = _load_checkpoint()
    manifest_text = json.dumps(bundle.manifest)
    legacy_names = [f"{name}_cloudy" for name in ("asr", "olr")]

    assert "asr" in bundle.output_names
    assert "olr" in bundle.output_names
    assert all(name not in manifest_text for name in legacy_names)


def test_private_gplfr_bundle_predicts_mean_and_variance(tmp_path: Path) -> None:
    bundle_dir = build_gplfr_bundle(tmp_path)
    bundle = _load_checkpoint(bundle_dir)
    items = [{"F_star": 1000.0, "GCM": "UM"}]
    mean, variance = predict_gplfr_batch(bundle, items, space="model", return_variance=True)
    _, total = predict_gplfr_batch(bundle, items, space="model", return_variance=True, include_residual=True)
    mean_vec, variance_vec = predict_gplfr_batch(bundle, items, space="model", return_variance=True, unpack=False)

    assert_allclose(mean["surface_temperature"], np.full((1, 2, 2), 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(variance["surface_temperature"], np.full((1, 2, 2), 0.9), rtol=1.0e-6, atol=1.0e-6)  # coherent only by default
    assert_allclose(total["surface_temperature"], np.full((1, 2, 2), 5.25), rtol=1.0e-6, atol=1.0e-6)  # + white residual
    assert_allclose(mean_vec, np.full((1, 4), 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(variance_vec, np.full((1, 4), 0.9), rtol=1.0e-6, atol=1.0e-6)


def test_variance_requires_model_space(tmp_path: Path) -> None:
    bundle = _load_checkpoint(build_gplfr_bundle(tmp_path))
    try:
        predict_gplfr_batch(bundle, [{"F_star": 1000.0, "GCM": "UM"}], return_variance=True)
    except ValueError as exc:
        assert "model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("variance in physical space should raise")


def test_public_emulator_uses_internal_default_bundle(monkeypatch, tmp_path: Path) -> None:
    bundle = _load_checkpoint(build_gplfr_bundle(tmp_path))
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: bundle)
    inputs = {"F_star": 1000.0, "GCM": "UM"}
    emulator = Emulator()
    pred = emulator.predict(inputs)
    mean, variance = emulator.predict(inputs, space="model", return_variance=True, include_residual=True)
    samples = emulator.sample(inputs, n_samples=8, seed=0)
    _, variance_vec = emulator.predict(inputs, space="model", return_variance=True, include_residual=True, unpack=False)
    mean_vec = emulator.predict(inputs, unpack=False)
    batch_mean = emulator.predict([inputs, inputs], unpack=False)
    batch_mean_from_columns = emulator.predict({"F_star": [1000.0, 1000.0], "GCM": ["UM", "um"]}, unpack=False)
    batch_mean_dict, batch_variance_dict = emulator.predict([inputs, inputs], space="model", return_variance=True, include_residual=True, fields=["surface_temperature"])

    assert_allclose(pred["surface_temperature"], np.full((2, 2), 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(mean["surface_temperature"], np.full((2, 2), 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(variance["surface_temperature"], np.full((2, 2), 5.25), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(mean_vec, np.full(4, 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(variance_vec, np.full(4, 5.25), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(batch_mean, np.full((2, 4), 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(batch_mean_from_columns, np.full((2, 4), 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(batch_mean_dict["surface_temperature"], np.full((2, 2, 2), 6.0), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(batch_variance_dict["surface_temperature"], np.full((2, 2, 2), 5.25), rtol=1.0e-6, atol=1.0e-6)
    assert samples["surface_temperature"].shape == (8, 2, 2)
    assert samples["surface_temperature"].dtype == np.float32


def test_sample_seed_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    inputs = {"F_star": 1000.0, "GCM": "UM"}
    emulator = Emulator()

    first = emulator.sample(inputs, n_samples=16, seed=0)["surface_temperature"]
    second = emulator.sample(inputs, n_samples=16, seed=0)["surface_temperature"]
    different = emulator.sample(inputs, n_samples=16, seed=1)["surface_temperature"]

    assert_allclose(first, second)
    assert not np.allclose(first, different)


def test_sample_mean_matches_predict_mean(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    inputs = {"F_star": 1000.0, "GCM": "UM"}
    emulator = Emulator()

    mean = emulator.predict(inputs)["surface_temperature"]
    samples = emulator.sample(inputs, n_samples=4096, seed=0)["surface_temperature"]

    assert_allclose(samples.mean(axis=0), mean, rtol=0.0, atol=0.18)


def test_sample_marginal_variance_matches_predict_variance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    inputs = {"F_star": 1000.0, "GCM": "UM"}
    emulator = Emulator()

    _, coherent = emulator.predict(inputs, space="model", return_variance=True)
    _, total = emulator.predict(inputs, space="model", return_variance=True, include_residual=True)
    samples = emulator.sample(inputs, n_samples=4096, seed=1)["surface_temperature"]
    full = emulator.sample(inputs, n_samples=4096, seed=1, sample_residual=True)["surface_temperature"]

    assert_allclose(samples.var(axis=0, ddof=1), coherent["surface_temperature"], rtol=0.08, atol=0.0)  # defaults mirror
    assert_allclose(full.var(axis=0, ddof=1), total["surface_temperature"], rtol=0.08, atol=0.0)  # residual flags mirror


def test_predict_split_variance_components(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    inputs = {"F_star": 1000.0, "GCM": "UM"}
    emulator = Emulator()

    mean, coherent, residual = emulator.predict(inputs, space="model", split_variance=True)
    mean_only = emulator.predict(inputs, space="model")
    _, default = emulator.predict(inputs, space="model", return_variance=True)
    _, total = emulator.predict(inputs, space="model", return_variance=True, include_residual=True)

    assert_allclose(mean["surface_temperature"], mean_only["surface_temperature"], rtol=0.0, atol=0.0)
    assert_allclose(default["surface_temperature"], coherent["surface_temperature"], rtol=0.0, atol=0.0)
    assert_allclose(coherent["surface_temperature"], np.full((2, 2), 0.9), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(residual["surface_temperature"], np.full((2, 2), 4.35), rtol=1.0e-6, atol=1.0e-6)
    assert_allclose(coherent["surface_temperature"] + residual["surface_temperature"], total["surface_temperature"], rtol=1.0e-6, atol=1.0e-6)
    with pytest.raises(ValueError):
        emulator.predict(inputs, split_variance=True)


KEPLER_1229B_Q90 = {
    "T_star": 3777.11181640625, "F_star": 724.1682548522949, "radius": 1.408750295639038,
    "gravity": 16.864251467227934, "P_rot": 86.827392578125, "P0": 1.0,
    "CO2": 0.01, "CH4": 0.001, "GCM": "exocam",
}


def _speckle_index(mask: np.ndarray) -> float:
    n = sum(np.roll(mask, d, ax) for d, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)))
    return float((((mask == 1) & (n <= 1)) | ((mask == 0) & (n >= 3))).mean())


def _checkerboard_energy(field: np.ndarray) -> float:
    lap = 4 * field - sum(np.roll(field, d, ax) for d, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)))
    return float(np.mean(lap**2) / np.var(field))


def test_bundled_model_coherent_samples_are_spatially_clean() -> None:
    emulator = Emulator()
    mean = emulator.predict(KEPLER_1229B_Q90, fields=["surface_temperature"])["surface_temperature"]
    coherent = emulator.sample(KEPLER_1229B_Q90, n_samples=8, seed=0, fields=["surface_temperature"])["surface_temperature"]
    residual = emulator.sample(KEPLER_1229B_Q90, n_samples=8, seed=0, sample_residual=True, fields=["surface_temperature"])["surface_temperature"]

    speckle = np.mean([_speckle_index((m <= 273.15).astype(int)) for m in coherent])
    rough_coherent = np.mean([_checkerboard_energy(m - mean) for m in coherent])
    rough_residual = np.mean([_checkerboard_energy(m - mean) for m in residual])

    assert speckle < 0.005
    assert rough_coherent < 0.3 * rough_residual
    assert_allclose(coherent.mean(axis=0), residual.mean(axis=0), rtol=0.0, atol=25.0)


def test_sample_residual_flag_controls_white_decoder_variance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    inputs = {"F_star": 1000.0, "GCM": "UM"}
    emulator = Emulator()

    coherent = emulator.sample(inputs, n_samples=4096, seed=2)["surface_temperature"]
    full = emulator.sample(inputs, n_samples=4096, seed=2, sample_residual=True)["surface_temperature"]

    assert_allclose(coherent.mean(axis=0), np.full((2, 2), 6.0), rtol=0.0, atol=0.1)
    assert_allclose(coherent.var(axis=0, ddof=1), np.full((2, 2), 0.9), rtol=0.08, atol=0.0)
    assert_allclose(full.var(axis=0, ddof=1), np.full((2, 2), 5.25), rtol=0.08, atol=0.0)


def test_sample_batch_shapes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    emulator = Emulator()
    inputs = {"F_star": [1000.0, 1000.0], "GCM": ["UM", "um"]}

    samples = emulator.sample(inputs, n_samples=8, seed=0)
    samples_vec = emulator.sample(inputs, n_samples=8, seed=0, unpack=False)
    field_samples = emulator.sample(inputs, n_samples=8, seed=0, fields=["surface_temperature"])
    single_vec = emulator.sample({"F_star": 1000.0, "GCM": "UM"}, n_samples=8, seed=0, unpack=False)

    assert samples["surface_temperature"].shape == (8, 2, 2, 2)
    assert samples_vec.shape == (8, 2, 4)
    assert field_samples["surface_temperature"].shape == (8, 2, 2, 2)
    assert single_vec.shape == (8, 4)


def test_predict_single_vs_batch_axis(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    emulator = Emulator()
    inputs = {"F_star": 1000.0, "GCM": "UM"}

    single = emulator.predict(inputs)["surface_temperature"]
    batch = emulator.predict([inputs, inputs])["surface_temperature"]

    assert single.shape == (2, 2)
    assert batch.shape == (2, 2, 2)


def test_to_physical_inverts_to_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    emulator = Emulator()
    inputs = {"F_star": 1000.0, "GCM": "UM"}

    physical = emulator.predict(inputs)
    model = emulator.predict(inputs, space="model")
    round_trip = emulator.to_physical(emulator.to_model(physical, inputs), inputs)
    single_model = emulator.to_model("surface_temperature", physical["surface_temperature"], inputs)
    single_physical = emulator.to_physical("surface_temperature", model["surface_temperature"], inputs)

    assert_allclose(emulator.to_physical(model, inputs)["surface_temperature"], physical["surface_temperature"], rtol=1.0e-5, atol=1.0e-5)
    assert_allclose(round_trip["surface_temperature"], physical["surface_temperature"], rtol=1.0e-5, atol=1.0e-5)
    assert_allclose(single_model, model["surface_temperature"], rtol=1.0e-5, atol=1.0e-5)
    assert_allclose(single_physical, physical["surface_temperature"], rtol=1.0e-5, atol=1.0e-5)


def test_to_physical_clips_specific_humidity(tmp_path: Path) -> None:
    bundle = _CheckpointBundle(
        root=tmp_path,
        gplfr_state={},
        manifest={
            "input_names": ["F_star"],
            "output_names": ["specific_humidity_0", "surface_temperature"],
            "raw_output_names": ["specific_humidity_0", "surface_temperature"],
            "output_transforms": {
                "specific_humidity_0": {"strategy": "log_Z-scaling", "kwargs": {"epsilon": 1.0e-30}},
                "surface_temperature": {"strategy": "Z-scaling", "kwargs": {}},
            },
            "asr_olr_normalize_by_f_star": False,
        },
    )

    physical = outputs_to_physical(
        bundle,
        {
            "specific_humidity_0": np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32),
            "surface_temperature": np.array([[250.0, 260.0, 270.0, 280.0]], dtype=np.float32),
        },
        {"F_star": 1000.0},
    )

    assert_allclose(physical["specific_humidity_0"], [[np.exp(-1.0), 1.0, 1.0, 1.0]], rtol=1.0e-6)
    assert_allclose(physical["surface_temperature"], [[250.0, 260.0, 270.0, 280.0]])


def test_torch_physical_prediction_clips_specific_humidity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from clero._torch_runtime import _inverse_preprocess_outputs

    bundle = _CheckpointBundle(
        root=tmp_path,
        gplfr_state={},
        manifest={
            "input_names": ["F_star"],
            "raw_output_names": ["specific_humidity_0"],
            "output_transforms": {"specific_humidity_0": {"strategy": "log_Z-scaling", "kwargs": {"epsilon": 1.0e-30}}},
            "asr_olr_normalize_by_f_star": False,
        },
    )

    y = torch.tensor([[[[0.0, 2.0], [-2.0, 1.0]]]], dtype=torch.float32)
    x_raw = torch.tensor([[1000.0]], dtype=torch.float32)

    out = _inverse_preprocess_outputs(bundle, y, x_raw).cpu().numpy()
    assert_allclose(out, [[[[1.0, 1.0], [np.exp(-2.0), 1.0]]]], rtol=1.0e-6)


def test_mixed_scalar_sequence_dict_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(clero.inference, "_load_checkpoint", lambda path=None: _load_checkpoint(build_gplfr_bundle(tmp_path)))
    emulator = Emulator()

    try:
        emulator.predict({"F_star": [1000.0, 1000.0], "GCM": "UM"})
    except ValueError as exc:
        assert "mixes scalar and sequence" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a dict mixing scalar and sequence values should raise")


def test_gplfr_predictions_default_gcm_to_um(tmp_path: Path) -> None:
    bundle = _load_checkpoint(build_gplfr_bundle(tmp_path))

    default = predict_gplfr_batch(bundle, [{"F_star": 1000.0}])
    explicit = predict_gplfr_batch(bundle, [{"F_star": 1000.0, "GCM": "um"}])
    for key in default:
        assert_allclose(default[key], explicit[key])


def test_inputs_use_astro_units_and_presets_are_complete() -> None:
    from clero import EARTH, TRAPPIST1E
    from clero._gplfr_runtime import _coerce_inputs_batch

    bundle = _load_checkpoint()  # real bundle has radius/P0 inputs to convert
    assert set(bundle.input_names) <= set(EARTH)                      # EARTH alone is a complete planet
    assert set(bundle.input_names) <= set(M_EARTH) and M_EARTH["T_star"] == 2600.0 and M_EARTH["P_rot"] == 5.03
    assert {"P0", "CO2", "CH4"}.isdisjoint(TRAPPIST1E)               # TRAPPIST1E omits the assumed atmosphere
    assert set(bundle.input_names) <= set({**EARTH, **TRAPPIST1E})   # ... but completes when overlaid on EARTH

    x, _ = _coerce_inputs_batch(bundle, [EARTH])
    cols = dict(zip(bundle.input_names, x[0]))
    assert_allclose(cols["radius"], EARTH["radius"] * 6.371e6)  # Earth radii -> m
    assert_allclose(cols["P0"], EARTH["P0"] * 1.0e5)            # bar -> Pa
    assert_allclose(cols["F_star"], EARTH["F_star"])            # SI inputs unchanged


def test_atmosphere_fractions_are_validated() -> None:
    from clero import EARTH
    from clero._gplfr_runtime import _coerce_inputs_batch

    bundle = _load_checkpoint()
    for bad in ({**EARTH, "CO2": -1.0e-3}, {**EARTH, "CH4": -1.0e-3}):
        with pytest.raises(ValueError, match="non-negative"):
            _coerce_inputs_batch(bundle, [bad])

    with pytest.raises(ValueError, match="CO2 \\+ CH4"):
        _coerce_inputs_batch(bundle, {**{key: [value] for key, value in EARTH.items()}, "CO2": [0.8], "CH4": [0.3]})


def test_extended_domain_warning() -> None:
    from clero import EARTH
    from clero._gplfr_runtime import _coerce_inputs_batch

    bundle = _load_checkpoint()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _coerce_inputs_batch(bundle, [{**EARTH, "P0": 20.0, "CH4": 0.06, "gravity": 20.1}])

    messages = [str(item.message) for item in caught]
    assert any("P0" in message and "extended domain" in message for message in messages)
    assert any("CH4" in message and "extended domain" in message for message in messages)
    assert any("gravity" in message and "extended domain" in message for message in messages)


def test_summary_and_grid_helpers_cover_surface_and_profile_outputs() -> None:
    outputs = {
        "surface_temperature": np.arange(8.0).reshape(2, 4),
        "temperature": np.arange(16.0).reshape(2, 2, 4),
    }
    lat = latitude_centers(2)
    lon = longitude_centers(4)

    summary = summarize_outputs(outputs, lat)
    profiles = profile_stats(outputs["temperature"], lat)
    surface_records = grid_records(outputs["surface_temperature"], lat=lat, lon=lon, value_name="surface_temperature")
    profile_records = grid_records(outputs["temperature"], lat=lat, lon=lon, levels=np.array([1000.0, 500.0]), value_name="temperature")

    assert_allclose(lat, [-35.26438968, 35.26438968])
    assert_allclose(lon, [-135.0, -45.0, 45.0, 135.0])
    assert summary["surface_temperature"]["global_mean"] == global_mean(outputs["surface_temperature"], lat)
    assert_allclose(summary["temperature"]["global_mean_profile"], profiles["global_mean_profile"])
    assert len(surface_records) == 8
    assert len(profile_records) == 16
    assert profile_records[-1]["level"] == 500.0


BUNDLE_SHA256 = "055c4f346d0c9f6941b48ebc0637f1cbb52c33185e34e90867d41d8bca975492"
TRAPPIST1E_1BAR = {**TRAPPIST1E, "P0": 1.0, "CO2": 4e-4, "CH4": 0.0}
# (global, dayside, nightside) area-weighted surface temperature means, pinned 2026-09-05 against BUNDLE_SHA256.
GOLDEN_TS = {"um": (234.694, 260.990, 208.399), "exocam": (240.603, 261.804, 219.401)}


def test_scope_table_matches_domain_dicts() -> None:
    rows = re.findall(r"^\| (\w+) / [^|]*\| ([\d.]+) – ([\d.]+) +\| ([\d.]+) – ([\d.]+) +\|$", (Path(__file__).resolve().parents[1] / "SCOPE.md").read_text(), re.M)
    core = {name: (float(lo), float(hi)) for name, lo, hi, _, _ in rows}
    extended = {name: (float(lo), float(hi)) for name, _, _, lo, hi in rows}
    assert core == CORE_DOMAIN
    assert extended == EXTENDED_DOMAIN


def test_missing_nonfinite_and_nonpositive_inputs_raise() -> None:
    emulator = Emulator()
    with pytest.raises(ValueError, match="missing required keys \\['P0'\\]"):
        emulator.predict({key: value for key, value in EARTH.items() if key != "P0"})
    with pytest.raises(ValueError, match="missing required keys"):
        emulator.predict([EARTH, TRAPPIST1E])
    for bad in ({**EARTH, "T_star": np.nan}, {**EARTH, "CO2": np.inf}):
        with pytest.raises(ValueError, match="finite"):
            emulator.predict(bad)
    for name in ("T_star", "F_star", "radius", "gravity", "P_rot", "P0"):
        with pytest.raises(ValueError, match="> 0"):
            emulator.predict({**EARTH, name: 0.0})
    with pytest.raises(ValueError, match="> 0"):
        emulator.predict({**{key: [value, value] for key, value in EARTH.items()}, "P_rot": [10.0, -1.0]})


def test_bundle_sha256_and_version(tmp_path: Path) -> None:
    emulator = Emulator()
    assert emulator.bundle_sha256 == BUNDLE_SHA256
    assert emulator.bundle_sha256 == hashlib.sha256((Path(clero.__file__).parent / "_model_bundle" / "gplfr_state.npz").read_bytes()).hexdigest()
    assert clero.__version__ == "0.1.0"
    assert Emulator(bundle=build_gplfr_bundle(tmp_path)).bundle_sha256 != BUNDLE_SHA256


@pytest.mark.parametrize("gcm", sorted(GOLDEN_TS))
def test_golden_trappist1e_surface_temperature_means(gcm: str) -> None:
    from clero.climate_analysis import dayside_mean, nightside_mean

    ts = Emulator().predict({**TRAPPIST1E_1BAR, "GCM": gcm}, fields=["surface_temperature"])["surface_temperature"]
    lat = latitude_centers(ts.shape[0])
    assert_allclose([global_mean(ts, lat), dayside_mean(ts, lat), nightside_mean(ts, lat)], GOLDEN_TS[gcm], rtol=1e-4)
