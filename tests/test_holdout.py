"""Generalization checks on figures the detector was never tuned against.

Every threshold in this project was chosen while looking at ``tests/data``. These
figures were written afterwards, and each one broke something real when first run:
ggplot's spineless grey panel defeated frame detection entirely, and two-tone markers
split into four series. They are kept as tests so those fixes cannot quietly rot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_DIR = ROOT / "tests" / "holdout"
sys.path.insert(0, str(ROOT / "tools"))

from plotdigitizer.pipeline import AutoDigitizer  # noqa: E402

from conftest import Figure  # noqa: E402

MEAN_ERROR_BUDGET = 0.6
MAX_ERROR_BUDGET = 1.5


def _ensure_holdout() -> dict:
    manifest = HOLDOUT_DIR / "manifest.json"
    if not manifest.exists():
        from make_holdout import build
        build(HOLDOUT_DIR)
    return json.loads(manifest.read_text())


def pytest_generate_tests(metafunc):  # pragma: no cover - collection hook
    if "holdout_figure" in metafunc.fixturenames:
        records = _ensure_holdout()["figures"]
        metafunc.parametrize(
            "holdout_figure",
            [Figure(record, HOLDOUT_DIR) for record in records],
            ids=[record["name"] for record in records],
        )


@pytest.fixture(scope="module")
def digitizer():
    return AutoDigitizer(device="cpu")


def _normalised(values, limits, scale):
    values = np.asarray(values, dtype=float)
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.log10(np.where(values > 0, values, np.nan))
            limits = np.log10(np.asarray(limits, dtype=float))
    lo, hi = np.asarray(limits, dtype=float)
    span = hi - lo
    return (values - lo) / (span if abs(span) > 1e-30 else 1.0)


def test_axes_are_calibrated(holdout_figure, digitizer):
    result = digitizer.run(holdout_figure.image)
    record = holdout_figure.record
    box = record["axes_box"]
    assert result.calibrated, f"{holdout_figure.name}: {result.warnings}"
    assert abs(result.calibration.x.to_pixel(record["xlim"][0]) - box["left"]) <= 2.0
    assert abs(result.calibration.x.to_pixel(record["xlim"][1]) - box["right"]) <= 2.0
    assert abs(result.calibration.y.to_pixel(record["ylim"][0]) - box["bottom"]) <= 2.0
    assert abs(result.calibration.y.to_pixel(record["ylim"][1]) - box["top"]) <= 2.0


def test_series_count(holdout_figure, digitizer):
    result = digitizer.run(holdout_figure.image)
    assert len(result.series) == len(holdout_figure.record["series"]), (
        f"{holdout_figure.name}: {[s.hex_color for s in result.series]}")


def test_points_are_accurate(holdout_figure, digitizer):
    result = digitizer.run(holdout_figure.image)
    record = holdout_figure.record
    truth = record["series"][0]
    detected = result.series[0]

    dx = _normalised(detected.data_points[:, 0], record["xlim"], record["xscale"])
    dy = _normalised(detected.data_points[:, 1], record["ylim"], record["yscale"])
    tx = _normalised(truth["x"], record["xlim"], record["xscale"])
    ty = _normalised(truth["y"], record["ylim"], record["yscale"])

    if truth["linestyle"] == "none":
        points = np.column_stack([dx, dy])
        distances = np.linalg.norm(
            np.column_stack([tx, ty])[:, None, :] - points[None, :, :], axis=2)
        error = distances.min(axis=1) * 100.0
    else:
        order = np.argsort(dx)
        sx, sy = dx[order], dy[order]
        inside = (tx >= sx.min()) & (tx <= sx.max())
        error = np.abs(np.interp(tx[inside], sx, sy) - ty[inside]) * 100.0

    assert np.nanmean(error) < MEAN_ERROR_BUDGET, (
        f"{holdout_figure.name}: mean {np.nanmean(error):.3f}%")
    assert np.nanmax(error) < MAX_ERROR_BUDGET, (
        f"{holdout_figure.name}: worst {np.nanmax(error):.3f}%")


def test_spineless_panel_is_found(digitizer):
    """ggplot draws no spines at all - the grey panel is the only boundary there is."""
    _ensure_holdout()
    from plotdigitizer.detect.frame import analyse_ink, detect_panel
    from plotdigitizer.image_io import load_image

    image = load_image(HOLDOUT_DIR / "holdout_ggplot.png")
    panel = detect_panel(image, analyse_ink(image))
    assert panel is not None, "grey panel not detected"
    frame, _ = panel
    assert frame.width > 0.5 * image.shape[1]


def test_ordinary_figures_have_no_panel(corpus):
    """A single-background figure must not be given a phantom panel."""
    from plotdigitizer.detect.frame import analyse_ink, detect_panel
    for name in ("linear_scatter", "dark_style", "grid_box"):
        image = corpus[name].image
        assert detect_panel(image, analyse_ink(image)) is None, name


def test_outlined_markers_are_one_series(digitizer):
    """markeredgecolor != markerfacecolor is one series drawn in two colours."""
    _ensure_holdout()
    from plotdigitizer.image_io import load_image
    result = digitizer.run(load_image(HOLDOUT_DIR / "holdout_edged.png"))
    assert len(result.series) == 1
    assert result.series[0].count == 20, "one point per marker"
