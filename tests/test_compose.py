"""Combining series into layers, splitting them back, and finding stray points."""

from __future__ import annotations

import numpy as np
import pytest

from plotdigitizer.calibration import AxisCalibration, Calibration
from plotdigitizer.compose import combine_series, describe_sources, split_series
from plotdigitizer.detect.extract import ExtractionMode, ExtractionSettings
from plotdigitizer.detect.outliers import residuals_from_trend, select_outliers
from plotdigitizer.pipeline import Series


def _calibration() -> Calibration:
    return Calibration(x=AxisCalibration(0.0, 0.0, 100.0, 10.0),
                       y=AxisCalibration(100.0, 0.0, 0.0, 50.0))


def _series(name, colour, xs, ys, mode=ExtractionMode.CURVE) -> Series:
    points = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    series = Series(name=name, color=colour, settings=ExtractionSettings(mode=mode),
                    pixel_points=points, data_points=np.empty((0, 2)))
    series.recompute_data(_calibration())
    return series


class TestCombine:
    def test_points_are_pooled_and_ordered_by_x(self):
        a = _series("A", (1, 2, 3), [10, 30], [5, 6])
        b = _series("B", (4, 5, 6), [20, 40], [7, 8])
        combined = combine_series([a, b], calibration=_calibration())
        assert combined.count == 4
        assert list(combined.pixel_points[:, 0]) == [10, 20, 30, 40]

    def test_default_name_lists_the_members(self):
        combined = combine_series([_series("DES", (1, 1, 1), [1], [1]),
                                   _series("LLR", (2, 2, 2), [2], [2])])
        assert combined.name == "DES + LLR"

    def test_name_and_colour_can_be_chosen(self):
        combined = combine_series([_series("A", (1, 1, 1), [1], [1])],
                                  name="merged", color=(9, 9, 9))
        assert combined.name == "merged"
        assert combined.color == (9, 9, 9)

    def test_data_points_follow_the_calibration(self):
        combined = combine_series([_series("A", (1, 1, 1), [0, 100], [100, 0])],
                                  calibration=_calibration())
        assert combined.data_points[0] == pytest.approx([0.0, 0.0])
        assert combined.data_points[-1] == pytest.approx([10.0, 50.0])

    def test_combined_series_carries_no_mask(self):
        """It no longer maps to one region of ink, so re-extraction would be nonsense."""
        a = _series("A", (1, 1, 1), [1], [1])
        a.mask = np.ones((4, 4), dtype=bool)
        assert combine_series([a]).mask is None

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError):
            combine_series([])


class TestSplit:
    def test_round_trip_restores_everything(self):
        a = _series("DES", (214, 39, 40), [10, 30], [5, 6], ExtractionMode.CURVE)
        b = _series("LLR", (31, 119, 180), [20, 40], [7, 8], ExtractionMode.SCATTER)
        b.visible = False

        restored = split_series(combine_series([a, b]), calibration=_calibration())

        assert [s.name for s in restored] == ["DES", "LLR"]
        assert [s.color for s in restored] == [(214, 39, 40), (31, 119, 180)]
        assert [s.settings.mode for s in restored] == [ExtractionMode.CURVE,
                                                       ExtractionMode.SCATTER]
        assert [s.visible for s in restored] == [True, False]
        assert np.array_equal(restored[0].pixel_points, a.pixel_points)
        assert np.array_equal(restored[1].pixel_points, b.pixel_points)

    def test_split_recomputes_data_values(self):
        a = _series("A", (1, 1, 1), [0, 100], [100, 0])
        restored = split_series(combine_series([a]), calibration=_calibration())
        assert restored[0].data_points[-1] == pytest.approx([10.0, 50.0])

    def test_permanent_combine_cannot_be_split(self):
        a = _series("A", (1, 1, 1), [1], [1])
        b = _series("B", (2, 2, 2), [2], [2])
        combined = combine_series([a, b], keep_sources=False)
        assert combined.sources is None
        assert not combined.is_combined
        assert split_series(combined) == []

    def test_plain_series_cannot_be_split(self):
        assert split_series(_series("A", (1, 1, 1), [1], [1])) == []

    def test_nested_combines_unwind_one_layer_at_a_time(self):
        a = _series("A", (1, 1, 1), [1], [1])
        b = _series("B", (2, 2, 2), [2], [2])
        c = _series("C", (3, 3, 3), [3], [3])

        inner = combine_series([a, b], name="AB")
        outer = combine_series([inner, c], name="ABC")

        first = split_series(outer)
        assert [s.name for s in first] == ["AB", "C"]
        assert first[0].is_combined, "the inner combine must survive the outer split"

        second = split_series(first[0])
        assert [s.name for s in second] == ["A", "B"]

    def test_editing_the_combined_series_does_not_disturb_its_sources(self):
        """Provenance is a snapshot; the layer underneath is not live-edited."""
        a = _series("A", (1, 1, 1), [10, 20], [1, 2])
        combined = combine_series([a])
        combined.pixel_points[0] = (999.0, 999.0)
        restored = split_series(combined)
        assert np.array_equal(restored[0].pixel_points, np.array([[10.0, 1.0], [20.0, 2.0]]))

    def test_describe_sources(self):
        members = [_series(f"S{i}", (1, 1, 1), [i], [i]) for i in range(5)]
        combined = combine_series(members)
        assert "and 2 more" in describe_sources(combined)
        assert describe_sources(_series("A", (1, 1, 1), [1], [1])) == ""


class TestOutliers:
    @staticmethod
    def _curve(n=120):
        xs = np.linspace(0, 400, n)
        ys = 100 + 30 * np.sin(xs / 60.0)
        return np.column_stack([xs, ys])

    def test_clean_curve_has_no_outliers(self):
        assert select_outliers(self._curve()).size == 0

    def test_injected_spikes_are_found(self):
        points = self._curve()
        spikes = [20, 55, 90]
        points[spikes, 1] += 40.0
        found = set(select_outliers(points).tolist())
        assert set(spikes) <= found, f"missed spikes; found {sorted(found)}"

    def test_a_displaced_run_is_found_whole(self):
        """A trace that hops onto another curve moves a whole stretch, not one point.

        The interior of such a run is invisible to trend-deviation - a rolling median
        follows the displaced points once enough of them fill its window - so this is
        the case the step-pair detector exists for.
        """
        points = self._curve()
        points[60:75, 1] += 35.0
        found = set(select_outliers(points).tolist())
        assert set(range(61, 75)) <= found, f"middle of the run missed; got {sorted(found)}"
        assert found <= set(range(57, 78)), "selection spilled outside the run"

    def test_a_steep_but_honest_curve_is_not_flagged(self):
        """Every step is large on a steep curve; none of them is anomalous."""
        xs = np.linspace(0, 100, 120)
        points = np.column_stack([xs, 5.0 * xs])
        assert select_outliers(points).size == 0

    def test_a_real_step_in_the_data_is_not_flagged_as_a_detour(self):
        """A curve that steps up and stays there has not left and come back."""
        points = self._curve()
        points[60:, 1] += 40.0
        found = set(select_outliers(points).tolist())
        assert len(found) < 10, f"a permanent step should not select a run: {sorted(found)}"

    def test_sensitivity_controls_how_much_is_caught(self):
        points = self._curve()
        points[30, 1] += 12.0
        points[70, 1] += 40.0
        loose = select_outliers(points, sensitivity=2.0).size
        strict = select_outliers(points, sensitivity=8.0).size
        assert loose >= strict

    def test_noisy_data_does_not_flag_everything(self):
        rng = np.random.default_rng(0)
        points = self._curve()
        points[:, 1] += rng.normal(0, 1.0, points.shape[0])
        assert select_outliers(points).size < 0.1 * points.shape[0]

    def test_unordered_points_are_handled(self):
        """Indices must refer to the input order, not the sorted order."""
        points = self._curve(60)
        points[25, 1] += 50.0
        shuffled = points[::-1].copy()
        found = select_outliers(shuffled)
        assert (len(points) - 1 - 25) in found.tolist()

    def test_tiny_series_returns_nothing(self):
        assert select_outliers(np.array([[0.0, 0.0], [1.0, 1.0]])).size == 0

    def test_residuals_are_zero_for_a_straight_line(self):
        xs = np.arange(50, dtype=float)
        points = np.column_stack([xs, 2 * xs + 1])
        assert np.abs(residuals_from_trend(points)).max() < 1e-6
