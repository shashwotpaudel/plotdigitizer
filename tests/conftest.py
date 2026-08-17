"""Shared fixtures: the ground-truth figure corpus.

The corpus is generated on demand, so a fresh checkout can run the tests without a
separate build step, and the PNGs never have to be committed as binary blobs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"
sys.path.insert(0, str(ROOT / "tools"))


class Figure:
    """One corpus figure: the image plus everything that is true about it."""

    def __init__(self, record: dict, data_dir: Path):
        self.record = record
        self.name = record["name"]
        self.path = data_dir / record["file"]
        self._image = None

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            from plotdigitizer.image_io import load_image
            self._image = load_image(self.path)
        return self._image

    @property
    def axes_box(self) -> dict:
        return self.record["axes_box"]

    @property
    def xticks(self) -> list[dict]:
        return self.record["xticks"]

    @property
    def yticks(self) -> list[dict]:
        return self.record["yticks"]

    @property
    def series(self) -> list[dict]:
        return self.record["series"]

    @property
    def x_range(self) -> float:
        lo, hi = self.record["xlim"]
        return abs(hi - lo)

    @property
    def y_range(self) -> float:
        lo, hi = self.record["ylim"]
        return abs(hi - lo)

    def __repr__(self) -> str:
        return f"<Figure {self.name}>"


def _ensure_corpus() -> dict:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        from make_test_plots import build_corpus
        build_corpus(DATA_DIR)
    return json.loads(manifest_path.read_text())


@pytest.fixture(scope="session")
def corpus() -> dict[str, Figure]:
    manifest = _ensure_corpus()
    return {rec["name"]: Figure(rec, DATA_DIR) for rec in manifest["figures"]}


@pytest.fixture(scope="session")
def figures(corpus) -> list[Figure]:
    return list(corpus.values())


def pytest_generate_tests(metafunc):
    """Parametrise any test asking for ``figure`` over the whole corpus."""
    if "figure" in metafunc.fixturenames:
        manifest = _ensure_corpus()
        records = manifest["figures"]
        metafunc.parametrize(
            "figure",
            [Figure(rec, DATA_DIR) for rec in records],
            ids=[rec["name"] for rec in records],
        )
