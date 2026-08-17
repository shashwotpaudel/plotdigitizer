"""Snapping and stroke tracing - the two aids that make hand-tracing viable."""

from __future__ import annotations

import numpy as np
import pytest

from plotdigitizer.detect.frame import analyse_ink
from plotdigitizer.detect.snap import snap_to_ink
from plotdigitizer.detect.trace import (
    TraceSettings,
    runs_in_column,
    sweep_along,
    trace_stroke,
)


def _canvas(height=200, width=300) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _draw(image: np.ndarray, xs, ys, thickness=2, colour=(0, 0, 0)) -> None:
    for x, y in zip(np.asarray(xs, int), np.asarray(ys, int)):
        if 0 <= x < image.shape[1]:
            lo = max(0, y - thickness // 2)
            hi = min(image.shape[0], lo + thickness)
            image[lo:hi, x] = colour


class TestSnap:
    def test_snaps_to_the_centre_of_a_stroke(self):
        image = _canvas()
        image[100:107, :] = 0                      # a 7 px thick horizontal band
        ink = analyse_ink(image)
        result = snap_to_ink(ink, 150.0, 101.0)
        assert result.snapped
        assert result.row == pytest.approx(103.0)  # centre of rows 100..106

    def test_snapping_is_unbiased_by_approach_side(self):
        """Coming from above or below must land on the same row, not on the near edge."""
        image = _canvas()
        image[100:107, :] = 0
        ink = analyse_ink(image)
        from_above = snap_to_ink(ink, 150.0, 100.0).row
        from_below = snap_to_ink(ink, 150.0, 106.0).row
        assert from_above == pytest.approx(from_below)

    def test_finds_ink_in_a_neighbouring_column(self):
        image = _canvas()
        image[:, 150:152] = 0                      # a vertical stroke
        ink = analyse_ink(image)
        result = snap_to_ink(ink, 147.0, 80.0, radius=6.0)
        assert result.snapped
        assert result.column == pytest.approx(150.0, abs=1.0)

    def test_empty_space_is_left_alone(self):
        ink = analyse_ink(_canvas())
        result = snap_to_ink(ink, 40.0, 40.0)
        assert not result.snapped
        assert (result.column, result.row) == (40.0, 40.0)

    def test_thick_region_does_not_pull_to_its_middle(self):
        """A filled block has no meaningful centre line; stay on its edge."""
        image = _canvas()
        image[50:150, 100:200] = 0
        ink = analyse_ink(image)
        result = snap_to_ink(ink, 150.0, 52.0, max_run=40.0)
        assert result.row < 60.0, "should not jump to the middle of a filled block"

    def test_out_of_bounds_is_safe(self):
        ink = analyse_ink(_canvas())
        assert not snap_to_ink(ink, -5.0, 10.0).snapped
        assert not snap_to_ink(ink, 10.0, 9999.0).snapped


class TestRuns:
    def test_finds_separate_spans(self):
        mask = np.zeros((20, 3), dtype=bool)
        mask[2:5, 1] = True
        mask[10:14, 1] = True
        assert runs_in_column(mask, 1) == [(2, 5), (10, 14)]

    def test_run_touching_both_edges(self):
        mask = np.ones((6, 1), dtype=bool)
        assert runs_in_column(mask, 0) == [(0, 6)]

    def test_empty_column(self):
        assert runs_in_column(np.zeros((5, 2), dtype=bool), 0) == []


class TestTraceStroke:
    def test_follows_a_straight_line(self):
        image = _canvas()
        xs = np.arange(20, 280)
        ys = (30 + 0.4 * xs).astype(int)
        _draw(image, xs, ys)
        points = trace_stroke(analyse_ink(image).mask, 150, 30 + 0.4 * 150)
        assert points.shape[0] > 200
        predicted = 30 + 0.4 * points[:, 0]
        assert np.abs(points[:, 1] - predicted).max() < 2.0

    def test_follows_a_curve(self):
        image = _canvas()
        xs = np.arange(20, 280)
        ys = (100 + 50 * np.sin(xs / 40.0)).astype(int)
        _draw(image, xs, ys)
        points = trace_stroke(analyse_ink(image).mask, 150, 100 + 50 * np.sin(150 / 40.0))
        assert points.shape[0] > 200
        predicted = 100 + 50 * np.sin(points[:, 0] / 40.0)
        assert np.abs(points[:, 1] - predicted).mean() < 2.0

    def test_stays_on_its_own_curve_when_two_cross(self):
        """The whole point: connectivity would merge these, continuity does not."""
        image = _canvas()
        xs = np.arange(20, 280)
        rising = (40 + 0.5 * xs).astype(int)
        falling = (200 - 0.5 * xs).astype(int)
        _draw(image, xs, falling)
        _draw(image, xs, rising)

        points = trace_stroke(analyse_ink(image).mask, 40, 40 + 0.5 * 40)
        assert points.shape[0] > 200
        predicted = 40 + 0.5 * points[:, 0]
        # If it had jumped onto the falling line at the crossing, the error would be huge.
        assert np.abs(points[:, 1] - predicted).max() < 4.0

    def test_bridges_the_gaps_of_a_dashed_stroke(self):
        image = _canvas()
        for start in range(20, 280, 12):
            xs = np.arange(start, min(start + 7, 280))
            _draw(image, xs, (60 + 0.3 * xs).astype(int))
        points = trace_stroke(analyse_ink(image).mask, 22, 60 + 0.3 * 22)
        assert points.shape[0] > 100, "dashes should be joined into one stroke"
        assert points[:, 0].max() > 250

    def test_stops_at_a_real_break(self):
        image = _canvas()
        _draw(image, np.arange(20, 120), np.full(100, 80))
        _draw(image, np.arange(240, 290), np.full(50, 80))
        settings = TraceSettings(max_gap=20)
        points = trace_stroke(analyse_ink(image).mask, 60, 80, settings)
        assert points[:, 0].max() < 200, "a wide gap must end the trace"

    def test_seed_off_the_stroke_still_works(self):
        image = _canvas()
        _draw(image, np.arange(20, 280), np.full(260, 90), thickness=3)
        points = trace_stroke(analyse_ink(image).mask, 150, 93)
        assert points.shape[0] > 200

    def test_seed_on_empty_space_returns_nothing(self):
        image = _canvas()
        _draw(image, np.arange(20, 280), np.full(260, 90))
        assert trace_stroke(analyse_ink(image).mask, 150, 20).shape == (0, 2)

    def test_blank_mask_returns_nothing(self):
        assert trace_stroke(np.zeros((50, 50), dtype=bool), 10, 10).shape == (0, 2)

    def test_ignores_a_vertical_rule(self):
        """A tall run is a frame line, not a stroke to follow."""
        image = _canvas()
        image[:, 150:152] = 0
        settings = TraceSettings(max_run=40.0)
        assert trace_stroke(analyse_ink(image).mask, 150, 100, settings).shape == (0, 2)


class TestTraceOnRealFigure:
    def _trace(self, figure, series_index=0, fraction=0.5):
        from plotdigitizer.detect.frame import detect_frame
        from plotdigitizer.detect.trace import stroke_mask

        ink = analyse_ink(figure.image)
        frame = detect_frame(figure.image, ink)
        truth = figure.series[series_index]
        seed = int(len(truth["px"]) * fraction)
        seed_col, seed_row = truth["px"][seed], truth["py"][seed]
        mask = stroke_mask(figure.image, ink, frame, seed_col, seed_row)
        return trace_stroke(mask, seed_col, seed_row), truth

    def test_traces_a_dashed_curve_end_to_end(self, corpus):
        """A seed anywhere on the dashed sine should recover the whole curve."""
        points, truth = self._trace(corpus["dashed_multi"], 0)
        assert points.shape[0] > 50

        self._assert_on_curve(points, truth, tolerance=2.0)

    def test_stays_on_one_curve_of_the_monochrome_figure(self):
        """The case colour cannot solve: same ink, separated only by continuity."""
        import json
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        holdout = root / "tests" / "holdout"
        if not (holdout / "manifest.json").exists():
            sys.path.insert(0, str(root / "tools"))
            from make_holdout import build
            build(holdout)

        from conftest import Figure
        manifest = json.loads((holdout / "manifest.json").read_text())
        record = next(r for r in manifest["figures"] if r["name"] == "holdout_monochrome")

        points, truth = self._trace(Figure(record, holdout), 0)
        assert points.shape[0] > 50
        self._assert_on_curve(points, truth, tolerance=3.0)

    @staticmethod
    def _assert_on_curve(points, truth, tolerance: float) -> None:
        order = np.argsort(truth["px"])
        xs = np.asarray(truth["px"])[order]
        ys = np.asarray(truth["py"])[order]
        inside = (points[:, 0] >= xs.min()) & (points[:, 0] <= xs.max())
        expected = np.interp(points[inside, 0], xs, ys)
        error = np.abs(points[inside, 1] - expected)
        assert error.mean() < tolerance, f"mean {error.mean():.2f}px off the seeded curve"


class TestSweepAlong:
    """Dragging along a curve: the path says which curve, the columns say how densely."""

    @staticmethod
    def _two_curves():
        image = _canvas(240, 320)
        xs = np.arange(20, 300)
        upper = (60 + 0.15 * xs).astype(int)
        lower = (150 - 0.10 * xs).astype(int)
        _draw(image, xs, upper)
        _draw(image, xs, lower)
        return image, xs, upper, lower

    def test_one_point_per_column_swept(self):
        image = _canvas()
        xs = np.arange(30, 200)
        _draw(image, xs, (100 + 0.2 * xs).astype(int))
        # A coarse path, as a fast drag would produce.
        path = [(30, 106), (110, 122), (199, 139)]
        points = sweep_along(analyse_ink(image).mask, path)
        assert points.shape[0] > 160, "should fill in every column, not follow the samples"
        assert np.array_equal(np.unique(points[:, 0]), points[:, 0]), "one point per column"

    def test_density_does_not_depend_on_how_fast_you_drag(self):
        """The output must record the figure, not the gesture."""
        image = _canvas()
        xs = np.arange(30, 200)
        _draw(image, xs, (100 + 0.2 * xs).astype(int))
        mask = analyse_ink(image).mask

        # Same start and end, wildly different numbers of mouse events between them.
        quick = sweep_along(mask, [(30, 106), (199, 139)])
        slow = sweep_along(mask, [(x, 100 + 0.2 * x) for x in [*range(30, 199, 3), 199]])
        assert quick.shape == slow.shape
        assert np.allclose(quick, slow, atol=1.0)

    def test_the_path_decides_which_curve(self):
        image, xs, upper, lower = self._two_curves()
        mask = analyse_ink(image).mask

        along_upper = sweep_along(mask, [(x, 60 + 0.15 * x) for x in (30, 150, 290)])
        along_lower = sweep_along(mask, [(x, 150 - 0.10 * x) for x in (30, 150, 290)])

        expected_upper = 60 + 0.15 * along_upper[:, 0]
        expected_lower = 150 - 0.10 * along_lower[:, 0]
        assert np.abs(along_upper[:, 1] - expected_upper).mean() < 2.0
        assert np.abs(along_lower[:, 1] - expected_lower).mean() < 2.0
        # And the two sweeps must not have collapsed onto the same curve.
        assert np.abs(along_upper[:, 1] - along_lower[:, 1]).mean() > 10.0

    def test_a_sloppy_path_still_finds_the_curve(self):
        image = _canvas()
        xs = np.arange(30, 250)
        _draw(image, xs, (120 + 0.1 * xs).astype(int))
        rng = np.random.default_rng(0)
        path = [(x, 120 + 0.1 * x + rng.normal(0, 4)) for x in range(30, 250, 10)]
        points = sweep_along(analyse_ink(image).mask, path, corridor=12.0)
        assert points.shape[0] > 180
        assert np.abs(points[:, 1] - (120 + 0.1 * points[:, 0])).mean() < 2.0

    def test_straying_off_the_curve_leaves_a_gap(self):
        """A sweep that wanders must not invent points where it saw nothing."""
        image = _canvas()
        xs = np.arange(30, 250)
        _draw(image, xs, np.full(xs.size, 120))
        # The middle of the path lifts far above the curve, outside the corridor.
        path = [(30, 120), (120, 120), (150, 40), (180, 120), (249, 120)]
        points = sweep_along(analyse_ink(image).mask, path, corridor=10.0)
        covered = set(points[:, 0].astype(int))
        assert 150 not in covered, "no ink within the corridor there"
        assert 40 in covered and 240 in covered

    def test_corridor_limits_how_far_it_reaches(self):
        image, xs, upper, lower = self._two_curves()
        mask = analyse_ink(image).mask
        # A path midway between the two curves with a tight corridor finds neither.
        midway = [(x, 0.5 * ((60 + 0.15 * x) + (150 - 0.10 * x))) for x in (40, 150, 280)]
        assert sweep_along(mask, midway, corridor=3.0).shape[0] == 0

    def test_degenerate_input_is_safe(self):
        mask = np.zeros((40, 40), dtype=bool)
        assert sweep_along(mask, [(1, 1), (2, 2)]).shape == (0, 2)
        assert sweep_along(np.ones((40, 40), dtype=bool), [(1, 1)]).shape == (0, 2)

    def test_a_backtracking_drag_is_tolerated(self):
        image = _canvas()
        xs = np.arange(30, 200)
        _draw(image, xs, np.full(xs.size, 90))
        path = [(30, 90), (150, 90), (100, 90), (199, 90)]     # doubles back
        points = sweep_along(analyse_ink(image).mask, path)
        assert points.shape[0] > 150
        assert np.abs(points[:, 1] - 90).max() < 2.0
