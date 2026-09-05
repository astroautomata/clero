"""Private bundle container helpers for CLERO inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_GPLFR_REQUIRED_FILES = ("manifest.json", "gplfr_state.npz")
_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "_model_bundle"


class _CheckpointBundle:
    def __init__(self, root: Path, manifest: dict[str, Any], gplfr_state: dict[str, np.ndarray]):
        self.root = root
        self.manifest = manifest
        self.gplfr_state = gplfr_state

    @property
    def input_names(self) -> list[str]:
        return list(self.manifest["input_names"])

    @property
    def output_names(self) -> list[str]:
        return list(self.manifest["output_names"])


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_checkpoint(path: str | Path | None = None) -> _CheckpointBundle:
    root = _DEFAULT_CHECKPOINT if path is None else Path(path)
    if root.is_file():
        root = root.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_type") != "gplfr_map":
        raise ValueError("CLERO currently supports only gplfr_map bundles")
    for name in _GPLFR_REQUIRED_FILES:
        if not (root / name).exists():
            raise FileNotFoundError(root / name)
    return _CheckpointBundle(root=root, manifest=manifest, gplfr_state=_load_npz(root / "gplfr_state.npz"))
