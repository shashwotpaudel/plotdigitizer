"""Series discovery and point extraction."""

from __future__ import annotations

import numpy as np
import pytest

from plotdigitizer.backend import NumpyBackend
from plotdigitizer.detect.extract import (
    ExtractionMode,
    ExtractionSettings,
    choose_mode,
    extract_points,
)
from plotdigitizer.detect.frame import analyse_ink, detect_frame
from plotdigitizer.detect.series import discover_series
from plotdigitizer.detect.ticks import detect_ticks
from plotdigitizer.pipeline import _inward_reach

#: figure name -> number of series actually plotted
EXPECTED_SERIES = {
    "linear_scatter": 1, "linear_line": 1, "multi_scatter_legend": 3, "log_y_line": 1,
    "loglog_scatter": 1, "grid_box": 1, "negative_range": 1, "offset_text": 1,
    "sci_offset": 1, "dashed_multi": 2, "highdpi_small": 1, "dark_style": 1,
    "dense_scatter": 1, "inward_ticks": 1,
}

#: figures whose data is drawn as isolated markers rather than a stroke
SCATTER_FIGURES = {"linear_scatter", "multi_scatter_legend", "loglog_scatter", "dense_scatter"}


def _content(figure):
    ink = analyse_ink(figure.image)
    frame = detect_frame(figure.image, ink)
    x_ticks = detect_ticks(ink, frame, "x")
    y_ticks = detect_ticks(ink, frame, "y")
    return discover_series(
        figure.image, ink, frame,
        x_ticks=x_ticks.positions, y_ticks=y_ticks.positions,
        backend=NumpyBackend(),
        inward_reach_x=_inward_reach(x_ticks), inward_reach_y=_inward_reach(y_ticks),
    )


def test_series_count_matches_the_figure(figure):
    """No phantom series from anti-aliasing, gridlines, legends or tick marks."""
    content = _content(figure)
    assert len(content.series) == EXPECTED_SERIES[figure.name], (
        f"{figure.name}: found {[s.hex_color for s in content.series]}"
    )


def test_series_colours_match_what_was_plotted(figure):
    content = _content(figure)
    plotted = [tuple(int(s["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
               for s in figure.series]
    for wanted in plotted:
        distances = [np.linalg.norm(np.array(s.color_rgb, float) - np.array(wanted, float))
                     for s in content.series]
        assert min(distances) < 40, (
            f"{figure.name}: no detected series near {wanted}; "
            f"got {[s.color_rgb for s in content.series]}"
        )


def test_extraction_mode_is_chosen_correctly(figure):
    """Markers must be counted individually; strokes must be traced."""
    expected = ExtractionMode.SCATTER if figure.name in SCATTER_FIGURES else ExtractionMode.CURVE
    for series in _content(figure).series:
        assert choose_mode(series.mask) is expected, f"{figure.name}"


def test_scatter_recovers_one_point_per_marker(corpus):
    """The count is the claim here: 60 markers must give 60 points, not a traced line."""
    for name in ("linear_scatter", "dense_scatter", "loglog_scatter"):
        figure = corpus[name]
        expected = len(figure.series[0]["x"])
        content = _content(figure)
        points = extract_points(content.series[0].mask,
                                ExtractionSettings(mode=ExtractionMode.SCATTER),
                                NumpyBackend())
        assert points.shape[0] == expected, f"{name}: {points.shape[0]} != {expected}"


def test_legend_box_is_found_and_excluded(corpus):
    content = _content(corpus["multi_scatter_legend"])
    assert content.legend_rect is not None, "framed legend should be detected"
    x0, y0, x1, y1 = content.legend_rect
    for series in content.series:
        inside = series.mask[y0:y1, x0:x1].sum()
        assert inside == 0, "legend swatches must not be extracted as data"


def test_gridlines_are_rejected(corpus):
    """The gridded figure has one data series; the grid must not become a second."""
    content = _content(corpus["grid_box"])
    assert len(content.series) == 1
    assert any(s.structural_score > 0.5 for s in content.rejected) or not content.rejected


class TestExtractionPrimitives:
    def test_empty_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        assert extract_points(mask, ExtractionSettings()).shape == (0, 2)

    def test_scatter_centroids(self):
        mask = np.zeros((40, 90), dtype=bool)
        for cx in (10, 40, 70):
            mask[18:23, cx - 2:cx + 3] = True
        points = extract_points(mask, ExtractionSettings(mode=ExtractionMode.SCATTER))
        assert points.shape[0] == 3
        assert np.allclose(points[:, 0], [10, 40, 70], atol=0.5)
        assert np.allclose(points[:, 1], 20, atol=0.5)

    def test_merged_markers_are_split(self):
        """Two touching markers must give two points, not one in between."""
        mask = np.zeros((40, 60), dtype=bool)
        mask[18:23, 18:23] = True
        mask[18:23, 23:28] = True          # touching neighbour
        mask[18:23, 45:50] = True          # a lone marker sets the reference size
        points = extract_points(mask, ExtractionSettings(mode=ExtractionMode.SCATTER))
        assert points.shape[0] == 3, f"got {points}"

    def test_curve_follows_the_stroke(self):
        mask = np.zeros((60, 100), dtype=bool)
        columns = np.arange(5, 95)
        rows = (10 + 0.4 * columns).astype(int)
        for c, r in zip(columns, rows):
            mask[r:r + 2, c] = True
        points = extract_points(mask, ExtractionSettings(mode=ExtractionMode.CURVE, x_step=1))
        assert points.shape[0] > 80
        predicted = 10 + 0.4 * points[:, 0]
        assert np.abs(points[:, 1] - predicted).mean() < 1.5

    def test_curve_bridges_dash_gaps_but_not_real_breaks(self):
        mask = np.zeros((40, 200), dtype=bool)
        for start in range(5, 80, 10):      # dashes with 4 px gaps
            mask[20:22, start:start + 6] = True
        mask[20:22, 150:190] = True         # a separate stretch after a wide gap
        settings = ExtractionSettings(mode=ExtractionMode.CURVE, x_step=1, max_gap=20)
        points = extract_points(mask, settings)
        xs = np.sort(points[:, 0])
        assert np.max(np.diff(xs)) > 20, "a genuine break must stay a break"

    def test_max_points_thins_the_output(self):
        mask = np.zeros((40, 300), dtype=bool)
        mask[20:22, 5:295] = True
        settings = ExtractionSettings(mode=ExtractionMode.CURVE, x_step=1, max_points=25)
        assert extract_points(mask, settings).shape[0] == 25


@pytest.mark.parametrize("name", ["linear_scatter", "dashed_multi"])
def test_backend_choice_does_not_change_results(corpus, name):
    """CPU and GPU must extract the same points, or the GPU path is not trustworthy."""
    from plotdigitizer.backend import select_backend
    gpu = select_backend("cuda")
    if gpu.kind != "cuda":
        pytest.skip("no CUDA backend available")

    content = _content(corpus[name])
    for series in content.series:
        settings = ExtractionSettings(mode=choose_mode(series.mask))
        on_cpu = extract_points(series.mask, settings, NumpyBackend())
        on_gpu = extract_points(series.mask, settings, gpu)
        assert on_cpu.shape == on_gpu.shape
        assert np.allclose(on_cpu, on_gpu, atol=0.5)
