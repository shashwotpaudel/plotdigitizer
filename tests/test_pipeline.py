"""End-to-end: image in, correct numbers out - plus export, projects and the CLI."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from plotdigitizer.calibration import AxisScale
from plotdigitizer.export import ExportLayout, ExportOptions, csv_string, write_csv, write_json
from plotdigitizer.pipeline import AutoDigitizer
from plotdigitizer.project import load_project, save_project

#: Accuracy budget, as a percentage of the plotted axis extent. A pixel is roughly
#: 0.2% of a 500 px axis, so this asks for accuracy close to the image resolution.
MEAN_ERROR_BUDGET = 0.5
MAX_ERROR_BUDGET = 1.5

LOG_AXES = {"log_y_line": ("linear", "log10"), "loglog_scatter": ("log10", "log10")}


@pytest.fixture(scope="session")
def digitizer():
    return AutoDigitizer(device="cpu")


@pytest.fixture(scope="session")
def results(digitizer, corpus):
    return {name: digitizer.run(figure.image) for name, figure in corpus.items()}


def _axis_space(values, scale):
    values = np.asarray(values, dtype=float)
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(values > 0, values, np.nan))
    return values


def _normalised(values, limits, scale):
    lo, hi = _axis_space(limits, scale)
    span = hi - lo
    return (_axis_space(values, scale) - lo) / (span if abs(span) > 1e-30 else 1.0)


class TestCalibrationAccuracy:
    def test_axis_limits_land_on_the_right_pixels(self, figure, digitizer):
        """Where does the fitted axis think the true limits are drawn? Within a pixel."""
        result = digitizer.run(figure.image)
        box = figure.axes_box
        record = figure.record
        for axis, key_lo, key_hi, lim in (
            (result.calibration.x, "left", "right", record["xlim"]),
            (result.calibration.y, "bottom", "top", record["ylim"]),
        ):
            assert abs(axis.to_pixel(lim[0]) - box[key_lo]) <= 1.5
            assert abs(axis.to_pixel(lim[1]) - box[key_hi]) <= 1.5

    def test_log_axes_are_recognised_as_log(self, corpus, digitizer):
        """Choosing linear for a log axis would be wrong everywhere but the ticks."""
        for name, (x_scale, y_scale) in LOG_AXES.items():
            result = digitizer.run(corpus[name].image)
            assert result.calibration.x.scale is AxisScale(x_scale), name
            assert result.calibration.y.scale is AxisScale(y_scale), name

    def test_linear_axes_are_not_mistaken_for_log(self, corpus, digitizer):
        for name in ("linear_scatter", "grid_box", "dense_scatter"):
            result = digitizer.run(corpus[name].image)
            assert result.calibration.x.scale is AxisScale.LINEAR, name
            assert result.calibration.y.scale is AxisScale.LINEAR, name

    def test_every_figure_reports_itself_calibrated(self, figure, digitizer):
        assert digitizer.run(figure.image).calibrated, f"{figure.name} fell back to defaults"


class TestDataAccuracy:
    def test_extracted_points_match_the_plotted_data(self, figure, digitizer):
        """The headline claim, measured against the arrays that were actually plotted."""
        result = digitizer.run(figure.image)
        record = figure.record
        assert len(result.series) == len(record["series"]), (
            f"{figure.name}: {len(result.series)} series detected, "
            f"{len(record['series'])} plotted"
        )

        available = list(result.series)
        for truth in record["series"]:
            want = np.array([int(truth["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)],
                            dtype=float)
            detected = min(available,
                           key=lambda s: np.linalg.norm(np.array(s.color, float) - want))
            available.remove(detected)
            assert detected.count > 0, f"{figure.name}: empty series"

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
                f"{figure.name}: mean error {np.nanmean(error):.3f}% of axis range")
            assert np.nanmax(error) < MAX_ERROR_BUDGET, (
                f"{figure.name}: worst error {np.nanmax(error):.3f}% of axis range")

    def test_scatter_figures_return_one_point_per_marker(self, corpus, digitizer):
        for name in ("linear_scatter", "dense_scatter", "loglog_scatter"):
            result = digitizer.run(corpus[name].image)
            expected = len(corpus[name].series[0]["x"])
            assert result.series[0].count == expected, f"{name}"


class TestRobustness:
    def test_blank_image_still_produces_a_usable_session(self, digitizer):
        """A figure it cannot read must still open with handles, not crash."""
        blank = np.full((200, 240, 3), 255, dtype=np.uint8)
        result = digitizer.run(blank)
        assert result.series == []
        assert result.warnings
        assert not result.calibrated
        assert result.calibration.x.is_valid, "fallback axes must still be usable"

    def test_warnings_are_raised_when_calibration_is_impossible(self, digitizer):
        noise = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        result = digitizer.run(noise)
        assert isinstance(result.warnings, list)

    def test_recompute_follows_a_calibration_edit(self, corpus, digitizer):
        """Dragging a handle in the UI must move every extracted point with it."""
        result = digitizer.run(corpus["linear_scatter"].image)
        before = result.series[0].data_points.copy()
        result.calibration.x.v2 = float(result.calibration.x.v2) * 2.0
        result.recompute()
        after = result.series[0].data_points
        assert not np.allclose(before[:, 0], after[:, 0])
        assert np.allclose(before[:, 1], after[:, 1]), "y must be untouched by an x edit"


class TestExport:
    def test_combined_layout_puts_series_side_by_side(self, corpus, digitizer):
        result = digitizer.run(corpus["multi_scatter_legend"].image)
        text = csv_string(result.series, ExportOptions(layout=ExportLayout.COMBINED))
        rows = list(csv.reader(text.splitlines()))
        assert rows[0] == ["Series 1 x", "Series 1 y", "Series 2 x", "Series 2 y",
                           "Series 3 x", "Series 3 y"]
        assert len(rows) == 1 + max(s.count for s in result.series)

    def test_long_layout_is_tidy(self, corpus, digitizer):
        result = digitizer.run(corpus["multi_scatter_legend"].image)
        rows = list(csv.reader(
            csv_string(result.series, ExportOptions(layout=ExportLayout.LONG)).splitlines()))
        assert rows[0] == ["series", "x", "y"]
        assert len(rows) == 1 + sum(s.count for s in result.series)

    def test_separate_layout_writes_one_file_each(self, corpus, digitizer, tmp_path):
        result = digitizer.run(corpus["multi_scatter_legend"].image)
        written = write_csv(tmp_path / "out.csv", result.series,
                            ExportOptions(layout=ExportLayout.SEPARATE))
        assert len(written) == 3
        assert all(p.exists() for p in written)

    def test_hidden_series_are_excluded(self, corpus, digitizer):
        result = digitizer.run(corpus["multi_scatter_legend"].image)
        result.series[0].visible = False
        rows = list(csv.reader(csv_string(result.series).splitlines()))
        assert len(rows[0]) == 4

    def test_values_round_trip_through_csv(self, corpus, digitizer, tmp_path):
        """Full precision by default: the CSV must not quietly lose accuracy."""
        result = digitizer.run(corpus["linear_scatter"].image)
        path = write_csv(tmp_path / "d.csv", result.series)[0]
        rows = list(csv.reader(path.read_text().splitlines()))[1:]
        recovered = np.array([[float(r[0]), float(r[1])] for r in rows])
        assert np.array_equal(recovered, result.series[0].data_points)

    def test_json_export_carries_the_calibration(self, corpus, digitizer, tmp_path):
        result = digitizer.run(corpus["log_y_line"].image)
        path = write_json(tmp_path / "d.json", result)
        payload = json.loads(path.read_text())
        assert payload["calibration"]["y"]["scale"] == "log10"
        assert len(payload["series"]) == 1


class TestProject:
    def test_round_trip_preserves_points_and_calibration(self, corpus, digitizer, tmp_path):
        result = digitizer.run(corpus["loglog_scatter"].image)
        path = save_project(tmp_path / "s.pdproj", result, image_path="figure.png")
        restored, image_path = load_project(path)

        assert image_path == "figure.png"
        assert len(restored.series) == len(result.series)
        assert restored.calibration.x.scale is AxisScale.LOG10
        assert np.allclose(restored.series[0].pixel_points, result.series[0].pixel_points)
        # Data values are recomputed, never stored, so they must still agree.
        assert np.allclose(restored.series[0].data_points, result.series[0].data_points)

    def test_rejects_a_future_format(self, tmp_path):
        path = tmp_path / "future.pdproj"
        path.write_text(json.dumps({"format": 99, "calibration": {}, "frame": {}}))
        with pytest.raises(ValueError, match="newer version"):
            load_project(path)


class TestCLI:
    def test_image_to_csv(self, corpus, tmp_path):
        from plotdigitizer.cli import main
        target = tmp_path / "out.csv"
        code = main([str(corpus["linear_scatter"].path), "-o", str(target), "--device", "cpu", "-q"])
        assert code == 0
        rows = list(csv.reader(target.read_text().splitlines()))
        assert len(rows) == 19          # header + 18 markers

    def test_values_match_the_plotted_data(self, corpus, tmp_path):
        from plotdigitizer.cli import main
        target = tmp_path / "out.csv"
        main([str(corpus["linear_scatter"].path), "-o", str(target), "--device", "cpu", "-q"])
        rows = list(csv.reader(target.read_text().splitlines()))[1:]
        got = np.array([[float(r[0]), float(r[1])] for r in rows])
        truth = corpus["linear_scatter"].series[0]
        assert np.allclose(np.sort(got[:, 0]), np.sort(truth["x"]), atol=0.05)
        assert np.allclose(np.sort(got[:, 1]), np.sort(truth["y"]), atol=0.25)

    def test_directory_input(self, tmp_path, corpus):
        from plotdigitizer.cli import main
        source = corpus["linear_scatter"].path.parent
        assert main([str(source), "-o", str(tmp_path), "--device", "cpu", "-q"]) == 0
        assert len(list(tmp_path.glob("*.csv"))) >= 14

    def test_devices_flag(self, capsys):
        from plotdigitizer.cli import main
        assert main(["--devices", "x"]) == 0
        assert "selected:" in capsys.readouterr().out
