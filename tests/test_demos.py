from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

DEMOS = sorted((Path(__file__).resolve().parents[1] / "demos").glob("*.ipynb"))


@pytest.mark.parametrize("path", DEMOS, ids=[p.name for p in DEMOS])
def test_demo_notebook_runs(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()
