"""Calibration maths: round trips, scale auto-detection, and outlier rejection."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from plotdigitizer.calibration import (
    AxisCalibration,
    AxisScale,
    Calibration,
    fit_axis,
)


class TestAxisRoundTrip:
    @pytest.mark.parametrize("scale,v1,v2", [
        (AxisScale.LINEAR, 0.0, 50.0),
        (AxisScale.LINEAR, -6.0, 6.0),
        (AxisScale.LOG10, 1e-2, 1e2),
        (AxisScale.LOGE, 0.5, 500.0),
        (AxisScale.RECIPROCAL, 0.5, 8.0),
    ])
    def test_pixel_value_round_trip(self, scale, v1, v2):
        axis = AxisCalibration(p1=57.0, v1=v1, p2=576.0, v2=v2, scale=scale)
        pixels = np.linspace(57.0, 576.0, 11)
        values = axis.to_data(pixels)
        back = axis.to_pixel(values)
        assert np.allclose(back, pixels, atol=1e-6)

    def test_endpoints_are_exact(self):
        axis = AxisCalibration(p1=100.0, v1=3.0, p2=400.0, v2=17.0)
        assert axis.to_data(100.0) == pytest.approx(3.0)
        assert axis.to_data(400.0) == pytest.approx(17.0)

    def test_inverted_axis(self):
        """Image rows grow downwards, so a normal y axis has value falling with pixel."""
        axis = AxisCalibration(p1=427.0, v1=0.0, p2=57.0, v2=50.0)
        assert axis.to_data(427.0) == pytest.approx(0.0)
        assert axis.to_data(242.0) == pytest.approx(25.0, abs=1e-6)

    def test_log_midpoint_is_geometric(self):
        axis = AxisCalibration(p1=0.0, v1=1.0, p2=100.0, v2=100.0, scale=AxisScale.LOG10)
        assert axis.to_data(50.0) == pytest.approx(10.0)

    def test_date_axis(self):
        t0 = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        t1 = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
        axis = AxisCalibration(p1=0.0, v1=t0, p2=365.0, v2=t1, scale=AxisScale.DATE)
        mid = axis.to_data(182.5)
        expected = (t0 + (t1 - t0) / 2).timestamp()
        assert mid == pytest.approx(expected, abs=1.0)


class TestValidity:
    def test_degenerate_pixels_invalid(self):
        assert not AxisCalibration(10.0, 0.0, 10.0, 5.0).is_valid

    def test_degenerate_values_invalid(self):
        assert not AxisCalibration(10.0, 4.0, 200.0, 4.0).is_valid

    def test_nonpositive_on_log_invalid(self):
        assert not AxisCalibration(10.0, 0.0, 200.0, 100.0, AxisScale.LOG10).is_valid

    def test_good_axis_valid(self):
        assert AxisCalibration(10.0, 0.0, 200.0, 100.0).is_valid


class TestFitting:
    def test_recovers_linear_axis(self):
        # Truth: value = (pixel - 57) / 10.38
        pixels = np.array([57.0, 161.0, 265.0, 369.0, 473.0, 576.0])
        values = (pixels - 57.0) / 10.38
        axis = fit_axis(pixels, values)
        assert axis is not None
        assert axis.scale is AxisScale.LINEAR
        assert axis.fit.rms_pixel_error < 0.05
        assert np.allclose(axis.to_data(pixels), values, atol=1e-6)

    def test_chooses_log_for_log_spaced_ticks(self):
        pixels = np.linspace(57.0, 576.0, 5)
        values = np.logspace(-2, 2, 5)
        axis = fit_axis(pixels, values)
        assert axis is not None
        assert axis.scale is AxisScale.LOG10
        assert axis.to_data(pixels[2]) == pytest.approx(1.0, rel=1e-6)

    def test_chooses_linear_for_linear_ticks_even_when_all_positive(self):
        """All-positive linear ticks must not be mistaken for a log axis."""
        pixels = np.linspace(57.0, 576.0, 6)
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        axis = fit_axis(pixels, values)
        assert axis is not None
        assert axis.scale is AxisScale.LINEAR

    def test_rejects_a_misread_label(self):
        """One OCR error must not drag the scale with it."""
        pixels = np.linspace(57.0, 576.0, 6)
        values = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        corrupted = values.copy()
        corrupted[3] = 300.0  # '30' misread as '300'
        axis = fit_axis(pixels, corrupted)
        assert axis is not None
        assert axis.fit.n_inliers == 5
        assert len(axis.fit.outliers) == 1
        assert axis.fit.outliers[0][1] == pytest.approx(300.0)
        assert np.allclose(axis.to_data(pixels), values, atol=1e-6)

    def test_confidence_high_for_clean_fit(self):
        pixels = np.linspace(57.0, 576.0, 6)
        values = np.linspace(0.0, 50.0, 6)
        axis = fit_axis(pixels, values)
        assert axis.fit.confidence > 0.9

    def test_returns_none_with_one_tick(self):
        assert fit_axis([100.0], [1.0]) is None

    def test_handles_two_ticks(self):
        axis = fit_axis([57.0, 576.0], [0.0, 10.0])
        assert axis is not None
        assert axis.to_data(316.5) == pytest.approx(5.0, abs=1e-6)


class TestCalibration2D:
    def test_point_mapping(self):
        cal = Calibration(
            x=AxisCalibration(57.0, 0.0, 576.0, 10.0),
            y=AxisCalibration(427.0, 0.0, 57.0, 50.0),
        )
        pts = np.array([[57.0, 427.0], [576.0, 57.0], [316.5, 242.0]])
        data = cal.to_data(pts)
        assert np.allclose(data[0], [0.0, 0.0], atol=1e-6)
        assert np.allclose(data[1], [10.0, 50.0], atol=1e-6)
        assert np.allclose(data[2], [5.0, 25.0], atol=1e-6)
        assert np.allclose(cal.to_pixel(data), pts, atol=1e-6)

    def test_serialisation_round_trip(self):
        cal = Calibration(
            x=AxisCalibration(57.0, 1.0, 576.0, 1000.0, AxisScale.LOG10),
            y=AxisCalibration(427.0, 0.0, 57.0, 50.0),
        )
        restored = Calibration.from_dict(cal.to_dict())
        assert restored.x.scale is AxisScale.LOG10
        assert restored.to_data(np.array([[300.0, 200.0]])) == pytest.approx(
            cal.to_data(np.array([[300.0, 200.0]]))
        )
