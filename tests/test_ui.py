"""Desktop UI behaviour, driven headlessly.

The detection thread is bypassed: the pipeline is run directly and its result handed to
the window through the same slot the worker uses. That keeps these tests about the UI's
own logic - editing, undo, propagation between panels - rather than about thread timing.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Redirect QSettings away from the real user configuration, before Qt resolves any
# paths. The window persists geometry, recent files and export options when it closes,
# so without this the suite overwrites the developer's own settings - and reads them
# back next time, which shows up as tests that pass once and then fail forever after
# because a previous run left the export layout somewhere else.
_SETTINGS_HOME = tempfile.mkdtemp(prefix="plotdigitizer-test-config-")
os.environ["XDG_CONFIG_HOME"] = _SETTINGS_HOME
atexit.register(shutil.rmtree, _SETTINGS_HOME, True)

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from plotdigitizer.calibration import AxisScale  # noqa: E402
from plotdigitizer.compose import combine_series, split_series  # noqa: E402
from plotdigitizer.detect.extract import ExtractionMode, ExtractionSettings  # noqa: E402
from plotdigitizer.pipeline import AutoDigitizer  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """Answer every message box automatically.

    The window now asks before discarding unsaved work, and a modal dialog in a
    headless run blocks forever with nobody to click it - one careless test would hang
    the whole suite. Defaults here are the "proceed" answers; the tests that are
    specifically about the prompts patch these again with the answer they want, which
    takes precedence because monkeypatch applies per test.
    """
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))


@pytest.fixture(scope="session")
def _digitized(corpus):
    """Run the pipeline once; the window tests reuse the result."""
    digitizer = AutoDigitizer(device="cpu")
    return {name: digitizer.run(corpus[name].image)
            for name in ("multi_scatter_legend", "linear_line")}


@pytest.fixture
def window(qapp, corpus, _digitized):
    """A window with an image loaded and a fresh copy of the detection result."""
    import copy

    from plotdigitizer.ui.main_window import MainWindow

    win = MainWindow(device="cpu")
    figure = corpus["multi_scatter_legend"]
    win._adopt_image(figure.image, figure.path)
    win._on_digitized(copy.deepcopy(_digitized["multi_scatter_legend"]))
    yield win
    # Most tests leave unsaved edits behind, and closing now asks about them. Drop the
    # flag rather than let a modal dialog block teardown with nobody to answer it.
    win._set_dirty(False)
    win.close()


class TestStartup:
    def test_window_builds_without_an_image(self, qapp):
        from plotdigitizer.ui.main_window import MainWindow
        win = MainWindow(device="cpu")
        assert win.canvas is not None
        assert not win.action_export.isEnabled()
        win.close()

    def test_panels_populate_from_the_detection(self, window):
        assert window.series_panel._list.count() == 3
        assert window.data_panel._table.rowCount() == 14
        assert "Axes detected" in window.calibrate_panel._badge.text()
        assert set(window.canvas._handles) == {"x1", "x2", "y1", "y2"}

    def test_handles_sit_on_the_detected_axes(self, window):
        result = window._result
        assert window.canvas._handles["x1"].position() == pytest.approx(result.calibration.x.p1)
        assert window.canvas._handles["y2"].position() == pytest.approx(result.calibration.y.p2)

    def test_export_preview_shows_real_numbers(self, window):
        text = window.export_panel._preview.toPlainText()
        assert "Series 1 x" in text
        assert len(text.splitlines()) > 2


class TestPointEditing:
    def test_left_click_adds_a_point(self, window):
        before = window._result.series[0].count
        window._on_point_added(0, 300.0, 200.0)
        assert window._result.series[0].count == before + 1
        assert window.data_panel._table.rowCount() == before + 1

    def test_added_point_lands_at_the_clicked_pixel(self, window):
        window._on_point_added(0, 300.0, 200.0)
        points = window._result.series[0].pixel_points
        assert np.isclose(points, [300.0, 200.0]).all(axis=1).any()

    def test_right_click_deletes_a_point(self, window):
        before = window._result.series[0].count
        window._on_point_removed(0, 3)
        assert window._result.series[0].count == before - 1

    def test_dragging_a_point_moves_it(self, window):
        window._on_point_moved(0, 2, 111.0, 222.0)
        assert tuple(window._result.series[0].pixel_points[2]) == (111.0, 222.0)

    def test_editing_a_table_value_moves_the_marker(self, window):
        """A number typed in the table must move the point, not desynchronise from it."""
        series = window._result.series[0]
        target_x, target_y = 5.0, 30.0
        window._on_value_edited(0, 0, target_x, target_y)
        assert series.data_points[0] == pytest.approx([target_x, target_y], rel=1e-6)
        # And the pixel it now sits on must map back to the same value.
        recovered = window._result.calibration.to_data(series.pixel_points[:1])[0]
        assert recovered == pytest.approx([target_x, target_y], rel=1e-6)


class TestUndo:
    def test_undo_restores_a_deleted_point(self, window):
        before = window._result.series[0].pixel_points.copy()
        window._on_point_removed(0, 5)
        window.undo()
        assert np.allclose(window._result.series[0].pixel_points, before)

    def test_redo_reapplies_it(self, window):
        window._on_point_removed(0, 5)
        after_delete = window._result.series[0].pixel_points.copy()
        window.undo()
        window.redo()
        assert np.allclose(window._result.series[0].pixel_points, after_delete)

    def test_undo_restores_a_calibration_edit(self, window):
        before = window._result.calibration.x.v2
        window.calibrate_panel._fields["x2"].setText("999")
        window._on_calibration_edited()
        assert window._result.calibration.x.v2 == pytest.approx(999.0)
        window.undo()
        assert window._result.calibration.x.v2 == pytest.approx(before)

    def test_a_drag_collapses_into_one_undo_step(self, window):
        """Twenty mouse-move events during one drag must not need twenty undos."""
        start = window._result.series[0].pixel_points.copy()
        for step in range(20):
            window._on_point_moved(0, 1, 200.0 + step, 150.0)
        window.undo()
        assert np.allclose(window._result.series[0].pixel_points, start)

    def test_undo_stack_is_bounded(self, window):
        from plotdigitizer.ui.main_window import UNDO_DEPTH
        for index in range(UNDO_DEPTH + 25):
            window._on_point_added(0, 100.0 + index, 150.0)
        assert len(window._undo) <= UNDO_DEPTH


class TestCalibrationInteraction:
    def test_dragging_a_handle_rescales_the_data(self, window):
        before = window._result.series[0].data_points.copy()
        window._on_handle_dragged("x2", window._result.calibration.x.p2 - 100.0)
        after = window._result.series[0].data_points
        assert not np.allclose(before[:, 0], after[:, 0])
        assert np.allclose(before[:, 1], after[:, 1])

    def test_changing_the_axis_type_is_applied(self, window):
        combo = window.calibrate_panel._scales["y"]
        combo.setCurrentIndex(combo.findData(AxisScale.LOG10.value))
        window._on_calibration_edited()
        assert window._result.calibration.y.scale is AxisScale.LOG10

    def test_invalid_values_are_refused_not_applied(self, window):
        """A blank field must leave the previous calibration in place."""
        before = window._result.calibration.x.v1
        window.calibrate_panel._fields["x1"].setText("")
        window._on_calibration_edited()
        assert window._result.calibration.x.v1 == pytest.approx(before)

    def test_negative_value_on_a_log_axis_is_flagged(self, window):
        combo = window.calibrate_panel._scales["y"]
        combo.setCurrentIndex(combo.findData(AxisScale.LOG10.value))
        window.calibrate_panel._fields["y1"].setText("-5")
        assert window.calibrate_panel._validate() is False


class TestSeriesPanel:
    def test_hiding_a_series_removes_it_from_the_export(self, window):
        window._on_series_visibility(0, False)
        assert "Series 1 x" not in window.export_panel._preview.toPlainText()

    def test_renaming_a_series_reaches_the_csv_header(self, window):
        window._on_series_renamed(0, "control")
        assert "control x" in window.export_panel._preview.toPlainText()

    def test_switching_the_active_series_updates_the_table(self, window):
        window._on_active_series(1)
        assert window.data_panel._series_index == 1
        assert window.data_panel._table.rowCount() == window._result.series[1].count

    def test_adding_an_empty_series(self, window):
        window._on_add_series()
        assert len(window._result.series) == 4
        assert window._result.series[-1].count == 0
        # A hand-made series has no mask, and asking to re-extract must not crash.
        window._on_series_settings(3, ExtractionSettings(mode=ExtractionMode.CURVE))

    def test_deleting_a_series(self, window):
        window._on_delete_series(1)
        assert len(window._result.series) == 2

    def test_mode_flip_reextracts(self, window, corpus, _digitized):
        """Switching a curve to scatter must actually re-run extraction."""
        import copy

        window._on_digitized(copy.deepcopy(_digitized["linear_line"]))
        series = window._result.series[0]
        assert series.settings.mode is ExtractionMode.CURVE
        points = window._extractor  # exercised synchronously below
        del points

        from plotdigitizer.detect.extract import extract_points
        scatter = ExtractionSettings(mode=ExtractionMode.SCATTER)
        recomputed = extract_points(series.mask, scatter)
        window._on_reextracted(0, recomputed)
        assert window._result.series[0].count == recomputed.shape[0]
        assert window.data_panel._table.rowCount() == recomputed.shape[0]


class TestSessionRoundTrip:
    def test_save_and_reload_through_the_window(self, window, tmp_path):
        from plotdigitizer.project import load_project, save_project
        window._on_point_removed(0, 0)
        expected = window._result.series[0].pixel_points.copy()

        path = save_project(tmp_path / "s.pdproj", window._result, window._image_path)
        restored, _ = load_project(path)
        assert np.allclose(restored.series[0].pixel_points, expected)


class TestHandleTargeting:
    """The reported bug: a calibration handle stealing clicks meant for a marker."""

    @staticmethod
    def _put_handle_on(window, point_index=3):
        """Move the x1 handle exactly onto one of the active series' markers."""
        column, row = window._result.series[0].pixel_points[point_index]
        window._result.calibration.x.p1 = float(column)
        window._sync_handles()
        return QPointF(float(column), float(row))

    def test_marker_under_a_handle_is_still_pickable(self, window):
        scene_point = self._put_handle_on(window)
        assert window.canvas._hit(scene_point) == 3
        view_point = window.canvas.mapFromScene(scene_point)
        assert not window.canvas._over_handle(view_point), (
            "the handle's grab region must not cover the data area")

    def test_handle_is_still_grabbable_at_its_grip(self, window):
        """Restricting the grab must not make the handle impossible to move."""
        handle = window.canvas._handles["x1"]
        handle.set_view_scale(1.0)
        start, _ = handle._span
        grip_point = QPointF(handle.position(), start + 2.0)
        assert handle.shape().contains(grip_point)

    def test_handle_is_not_grabbable_across_the_plot(self, window):
        handle = window.canvas._handles["x1"]
        handle.set_view_scale(1.0)
        start, end = handle._span
        middle = QPointF(handle.position(), 0.5 * (start + end))
        assert not handle.shape().contains(middle)

    def test_grab_region_is_a_constant_screen_size(self, window):
        """Otherwise the handle is a sliver when zoomed out and a wall when zoomed in."""
        handle = window.canvas._handles["x1"]
        start, _ = handle._span

        handle.set_view_scale(1.0)
        near = QPointF(handle.position() + 5.0, start + 2.0)
        assert handle.shape().contains(near)

        handle.set_view_scale(4.0)          # zoomed in: same screen offset, less scene
        assert not handle.shape().contains(near)


class TestDataLossGuards:
    def test_redigitize_is_undoable(self, window, _digitized):
        import copy
        before = window._result.series[0].pixel_points.copy()
        window._on_point_removed(0, 0)
        window._on_digitized(copy.deepcopy(_digitized["linear_line"]))
        assert len(window._result.series) == 1        # the replacement took effect
        window.undo()
        assert len(window._result.series) == 3
        assert np.allclose(window._result.series[0].pixel_points, before[1:])

    def test_editing_marks_the_session_dirty(self, window):
        assert not window._dirty
        window._on_point_added(0, 300.0, 200.0)
        assert window._dirty

    def test_saving_clears_dirty(self, window, tmp_path, monkeypatch):
        from plotdigitizer.project import save_project
        window._on_point_added(0, 300.0, 200.0)
        monkeypatch.setattr(
            "plotdigitizer.ui.main_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(tmp_path / "s.pdproj"), ""))
        window.save_session()
        assert not window._dirty
        assert (tmp_path / "s.pdproj").exists()
        del save_project

    def test_clean_session_needs_no_confirmation(self, window):
        assert window._confirm_discard("Closing") is True

    def test_dirty_session_asks_before_discarding(self, window, monkeypatch):
        """The prompt must actually appear, and Cancel must abort the action."""
        from PySide6.QtWidgets import QMessageBox

        asked = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: (asked.append(a[1]), QMessageBox.StandardButton.Cancel)[1]))
        window._on_point_added(0, 300.0, 200.0)
        assert window._confirm_discard("Closing") is False
        assert asked == ["Unsaved changes"]

    def test_discard_lets_the_action_proceed(self, window, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard))
        window._on_point_added(0, 300.0, 200.0)
        assert window._confirm_discard("Closing") is True

    def test_close_is_blocked_while_a_prompt_is_cancelled(self, window, monkeypatch):
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))
        window._on_point_added(0, 300.0, 200.0)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted(), "cancelling the prompt must keep the window open"

    def test_redigitize_over_manual_work_asks_first(self, window, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))
        window._on_point_added(0, 300.0, 200.0)
        assert window._confirm_replace_manual_work() is False


class TestTracingAssists:
    def test_added_points_snap_to_the_stroke(self, corpus, qapp, _digitized):
        """A click a few pixels off a curve should land on the curve."""
        import copy

        from plotdigitizer.ui.main_window import MainWindow
        win = MainWindow(device="cpu")
        figure = corpus["linear_line"]
        win._adopt_image(figure.image, figure.path)
        win._on_digitized(copy.deepcopy(_digitized["linear_line"]))

        truth = figure.series[0]
        middle = len(truth["px"]) // 2
        column, row = truth["px"][middle], truth["py"][middle]

        win._on_add_series()
        win._on_point_added(win._active, column, row + 4.0)   # deliberately off
        placed = win._result.series[win._active].pixel_points[0]
        assert abs(placed[1] - row) < 2.0, "snapping should recover the stroke centre"
        win.close()

    def test_snapping_can_be_turned_off(self, window):
        window._set_snap_enabled(False)
        window._on_add_series()
        window._on_point_added(window._active, 300.0, 200.0)
        assert tuple(window._result.series[window._active].pixel_points[0]) == (300.0, 200.0)

    def test_nudge_moves_by_the_documented_step(self, window):
        before = window._result.series[0].pixel_points[2].copy()
        window._on_point_nudged(0, 2, 1.0, 0.0)
        after = window._result.series[0].pixel_points[2]
        assert after[0] == pytest.approx(before[0] + 1.0)
        assert after[1] == pytest.approx(before[1])

    def test_a_run_of_nudges_is_one_undo_step(self, window):
        before = window._result.series[0].pixel_points.copy()
        for _ in range(10):
            window._on_point_nudged(0, 2, 1.0, 0.0)
        window.undo()
        assert np.allclose(window._result.series[0].pixel_points, before)

    def test_batch_delete_removes_exactly_the_selection(self, window):
        before = window._result.series[0].count
        window._on_points_removed(0, [1, 3, 5])
        assert window._result.series[0].count == before - 3

    def test_batch_delete_is_one_undo_step(self, window):
        before = window._result.series[0].pixel_points.copy()
        window._on_points_removed(0, list(range(0, 10)))
        window.undo()
        assert np.allclose(window._result.series[0].pixel_points, before)

    def test_traced_drag_points_share_one_undo_step(self, window):
        before = window._result.series[0].count
        for step in range(6):
            window._on_point_traced(0, 200.0 + step * 7, 250.0)
        assert window._result.series[0].count > before
        window.undo()
        assert window._result.series[0].count == before


class TestStrokeTracer:
    def test_seeding_a_curve_creates_a_series(self, corpus, qapp, _digitized):
        import copy

        from plotdigitizer.ui.main_window import MainWindow
        win = MainWindow(device="cpu")
        figure = corpus["linear_line"]
        win._adopt_image(figure.image, figure.path)
        win._on_digitized(copy.deepcopy(_digitized["linear_line"]))

        before = len(win._result.series)
        truth = figure.series[0]
        middle = len(truth["px"]) // 2
        win._on_stroke_seeded(truth["px"][middle], truth["py"][middle])

        assert len(win._result.series) == before + 1
        traced = win._result.series[-1]
        assert traced.count > 50
        # It must follow the curve it was seeded on.
        order = np.argsort(truth["px"])
        xs = np.asarray(truth["px"])[order]
        ys = np.asarray(truth["py"])[order]
        inside = (traced.pixel_points[:, 0] >= xs.min()) & (traced.pixel_points[:, 0] <= xs.max())
        expected = np.interp(traced.pixel_points[inside, 0], xs, ys)
        assert np.abs(traced.pixel_points[inside, 1] - expected).mean() < 3.0
        win.close()

    def test_seeding_empty_space_does_nothing(self, window):
        before = len(window._result.series)
        window._on_stroke_seeded(5.0, 5.0)
        assert len(window._result.series) == before


class TestFileFlow:
    def test_paste_converts_a_clipboard_image(self, qapp, corpus):
        from plotdigitizer.ui.canvas import numpy_to_qimage, qimage_to_numpy
        original = corpus["linear_scatter"].image
        restored = qimage_to_numpy(numpy_to_qimage(original))
        assert restored.shape == original.shape
        assert np.array_equal(restored, original)

    def test_odd_width_image_survives_row_padding(self, qapp):
        """QImage pads rows to 4 bytes; an odd width is where a naive reshape breaks."""
        from plotdigitizer.ui.canvas import numpy_to_qimage, qimage_to_numpy
        rng = np.random.default_rng(0)
        original = rng.integers(0, 255, (7, 13, 3), dtype=np.uint8)
        assert np.array_equal(qimage_to_numpy(numpy_to_qimage(original)), original)

    def test_dropped_path_is_recognised(self, qapp, corpus):
        from PySide6.QtCore import QMimeData, QUrl
        from PySide6.QtGui import QDropEvent

        from plotdigitizer.ui.main_window import MainWindow
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(corpus["linear_scatter"].path))])
        event = QDropEvent(QPointF(0, 0), Qt.DropAction.CopyAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        assert MainWindow._dropped_path(event) == corpus["linear_scatter"].path

    def test_unrelated_drop_is_ignored(self, qapp, tmp_path):
        from PySide6.QtCore import QMimeData, QUrl
        from PySide6.QtGui import QDropEvent

        from plotdigitizer.ui.main_window import MainWindow
        junk = tmp_path / "notes.txt"
        junk.write_text("hello")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(junk))])
        event = QDropEvent(QPointF(0, 0), Qt.DropAction.CopyAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        assert MainWindow._dropped_path(event) is None


class TestLassoSelection:
    """Right-drag selects instead of deleting, and says what it caught."""

    @staticmethod
    def _rect_over(window, indices):
        """A scene-space path enclosing exactly the given points of the active series."""
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainterPath
        points = window._result.series[window._active].pixel_points[list(indices)]
        rect = QRectF(points[:, 0].min() - 3, points[:, 1].min() - 3,
                      np.ptp(points[:, 0]) + 6, np.ptp(points[:, 1]) + 6)
        path = QPainterPath()
        path.addRect(rect)
        return path

    def test_a_sweep_selects_and_deletes_nothing(self, window):
        before = window._result.series[0].count
        caught = window.canvas._points_within(self._rect_over(window, [2, 3, 4]))
        window._on_selection_changed(0, caught)
        assert window._result.series[0].count == before, "selecting must not remove points"
        assert window._selection >= {2, 3, 4}

    def test_shift_adds_and_ctrl_subtracts(self, window):
        window._on_selection_changed(0, [1, 2, 3])
        assert window.canvas._combine_selection([5, 6], "add") == [1, 2, 3, 5, 6]
        assert window.canvas._combine_selection([2], "subtract") == [1, 3]
        assert window.canvas._combine_selection([9], "replace") == [9]

    def test_modifier_mapping(self, window):
        canvas = window.canvas
        assert canvas._selection_mode(Qt.KeyboardModifier.NoModifier) == "replace"
        assert canvas._selection_mode(Qt.KeyboardModifier.ShiftModifier) == "add"
        assert canvas._selection_mode(Qt.KeyboardModifier.ControlModifier) == "subtract"

    def test_freeform_and_rectangle_both_enclose(self, window):
        from PySide6.QtCore import QPointF
        canvas = window.canvas
        points = window._result.series[0].pixel_points
        target = points[3]

        canvas.set_lasso_shape("rectangle")
        canvas._lasso_origin = QPointF(target[0] - 5, target[1] - 5)
        rect_path = canvas._lasso_path(QPointF(target[0] + 5, target[1] + 5))
        assert 3 in canvas._points_within(rect_path)

        canvas.set_lasso_shape("freeform")
        canvas._lasso_points = [
            QPointF(target[0] - 6, target[1] - 6), QPointF(target[0] + 6, target[1] - 6),
            QPointF(target[0] + 6, target[1] + 6), QPointF(target[0] - 6, target[1] + 6),
        ]
        assert 3 in canvas._points_within(canvas._lasso_path(canvas._lasso_points[-1]))

    def test_escape_clears(self, window):
        window._on_selection_changed(0, [1, 2])
        window._clear_selection()
        assert window._selection == set()
        assert not window.selection_bar.isVisibleTo(window)

    def test_the_bar_appears_only_with_a_selection(self, window):
        # isVisibleTo rather than isVisible: the window itself is never shown in these
        # tests, which would make every child report invisible whatever its own flag.
        assert not window.selection_bar.isVisibleTo(window)
        window._on_selection_changed(0, [1, 2, 3])
        assert window.selection_bar.isVisibleTo(window)
        assert "3 of" in window.selection_bar._label.text()

    def test_selection_is_confined_to_the_active_series(self, window):
        window._on_selection_changed(1, [0, 1])
        assert window._selection == set(), "a selection for another series is ignored"


class TestSelectionStaysValid:
    """A selection pointing at the wrong rows would delete the wrong data."""

    def test_adding_a_point_shifts_the_selection(self, window):
        series = window._result.series[0]
        window._on_selection_changed(0, [0, 1, 2])
        first_x = series.pixel_points[0, 0]
        window._on_point_added(0, first_x - 50, 200.0)   # sorts in ahead of them all
        assert window._selection == {1, 2, 3}

    def test_adding_after_the_selection_leaves_it_alone(self, window):
        series = window._result.series[0]
        window._on_selection_changed(0, [0, 1])
        window._on_point_added(0, series.pixel_points[:, 0].max() + 50, 200.0)
        assert window._selection == {0, 1}

    def test_deleting_renumbers_what_remains(self, window):
        window._on_selection_changed(0, [3, 4, 5])
        window._on_point_removed(0, 0)
        assert window._selection == {2, 3, 4}

    def test_deleting_a_selected_point_drops_it(self, window):
        window._on_selection_changed(0, [3, 4])
        window._on_point_removed(0, 3)
        assert window._selection == {3}

    def test_undo_clears_the_selection(self, window):
        window._on_selection_changed(0, [1, 2])
        window._on_point_removed(0, 0)
        window.undo()
        assert window._selection == set(), "indices were rewritten wholesale"

    def test_switching_series_clears_the_selection(self, window):
        window._on_selection_changed(0, [1, 2])
        window._on_active_series(1)
        assert window._selection == set()

    def test_selection_never_points_past_the_end(self, window):
        window._set_selection([0, 1, 9999])
        assert max(window._selection) < window._result.series[0].count


class TestSelectionActions:
    def test_delete_removes_exactly_the_selection(self, window):
        series = window._result.series[0]
        before = series.pixel_points.copy()
        window._on_selection_changed(0, [1, 3])
        window.delete_selection()
        remaining = window._result.series[0].pixel_points
        assert remaining.shape[0] == before.shape[0] - 2
        assert np.array_equal(remaining, np.delete(before, [1, 3], axis=0))

    def test_delete_is_one_undo_step(self, window):
        before = window._result.series[0].pixel_points.copy()
        window._on_selection_changed(0, [0, 1, 2, 3])
        window.delete_selection()
        window.undo()
        assert np.allclose(window._result.series[0].pixel_points, before)

    def test_keep_only_is_the_complement_of_delete(self, window):
        series = window._result.series[0]
        before = series.pixel_points.copy()
        window._on_selection_changed(0, [2, 5, 7])
        window.keep_only_selection()
        assert np.array_equal(window._result.series[0].pixel_points, before[[2, 5, 7]])

    def test_invert(self, window):
        total = window._result.series[0].count
        window._on_selection_changed(0, [0, 1])
        window.invert_selection()
        assert window._selection == set(range(total)) - {0, 1}

    def test_move_to_an_existing_series_loses_no_points(self, window):
        result = window._result
        total_before = sum(s.count for s in result.series)
        source_before = result.series[0].count
        target_before = result.series[1].count

        window._on_selection_changed(0, [1, 2, 3])
        window.move_selection_to(1)

        assert result.series[0].count == source_before - 3
        assert result.series[1].count == target_before + 3
        assert sum(s.count for s in result.series) == total_before

    def test_moved_points_arrive_intact(self, window):
        moving = window._result.series[0].pixel_points[[1, 2, 3]].copy()
        window._on_selection_changed(0, [1, 2, 3])
        window.move_selection_to(1)
        arrived = window._result.series[1].pixel_points
        for point in moving:
            assert np.isclose(arrived, point).all(axis=1).any(), f"{point} did not arrive"

    def test_move_to_a_new_series(self, window):
        count = len(window._result.series)
        window._on_selection_changed(0, [0, 1])
        window.move_selection_to(-1)
        assert len(window._result.series) == count + 1
        assert window._result.series[-1].count == 2

    def test_move_is_undoable(self, window):
        before = [s.pixel_points.copy() for s in window._result.series]
        window._on_selection_changed(0, [1, 2])
        window.move_selection_to(1)
        window.undo()
        for original, restored in zip(before, window._result.series):
            assert np.allclose(original, restored.pixel_points)

    def test_actions_do_nothing_without_a_selection(self, window):
        before = window._result.series[0].count
        window.delete_selection()
        window.keep_only_selection()
        window.move_selection_to(1)
        assert window._result.series[0].count == before


class TestSpikeSelection:
    @pytest.fixture
    def curve_window(self, qapp, corpus, _digitized):
        """A window holding a densely traced curve, which is what spikes apply to."""
        import copy

        from plotdigitizer.ui.main_window import MainWindow
        win = MainWindow(device="cpu")
        figure = corpus["linear_line"]
        win._adopt_image(figure.image, figure.path)
        win._on_digitized(copy.deepcopy(_digitized["linear_line"]))
        yield win
        win._set_dirty(False)
        win.close()

    def test_selects_without_changing_anything(self, curve_window):
        series = curve_window._result.series[0]
        assert series.count > 100, "needs a dense curve"
        series.pixel_points[40, 1] += 40.0
        before = series.pixel_points.copy()

        curve_window.select_spikes(3.0)

        assert np.array_equal(curve_window._result.series[0].pixel_points, before), \
            "selecting must never modify the data"
        assert 40 in curve_window._selection

    def test_a_clean_curve_selects_nothing(self, curve_window):
        curve_window.select_spikes(3.0)
        assert curve_window._selection == set()

    def test_a_sparse_series_is_declined_rather_than_guessed(self, window):
        """14 scattered points have no local trend; guessing would select the wrong ones."""
        series = window._result.series[0]
        series.pixel_points[4, 1] += 60.0
        window.select_spikes(3.0)
        assert window._selection == set()


class TestCombineAndSplit:
    def test_combine_then_split_restores_the_originals(self, window):
        result = window._result
        originals = [(s.name, s.color, s.pixel_points.copy()) for s in result.series[:2]]
        total = sum(s.count for s in result.series)

        combined = combine_series(result.series[:2], calibration=result.calibration)
        result.series[:2] = [combined]
        assert combined.is_combined

        restored = split_series(combined, calibration=result.calibration)
        assert [(s.name, s.color) for s in restored] == [(n, c) for n, c, _ in originals]
        for (_, _, points), series in zip(originals, restored):
            assert np.array_equal(points, series.pixel_points)
        assert sum(s.count for s in restored) + result.series[-1].count == total

    def test_split_through_the_window_is_undoable(self, window):
        result = window._result
        names_before = [s.name for s in result.series]

        combined = combine_series(result.series[:2], calibration=result.calibration)
        window._push_undo()
        result.series[:2] = [combined]
        window._sync_all()

        window.split_series_at(0)
        assert [s.name for s in result.series] == names_before

        window.undo()
        assert len(result.series) == len(names_before) - 1

    def test_splitting_a_plain_series_is_refused(self, window):
        count = len(window._result.series)
        window.split_series_at(0)
        assert len(window._result.series) == count

    def test_the_panel_marks_a_combined_series(self, window):
        result = window._result
        result.series[:2] = [combine_series(result.series[:2],
                                            calibration=result.calibration)]
        window._sync_all()
        assert "merged" in window.series_panel._list.item(0).text()
        window.series_panel._list.setCurrentRow(0)
        assert window.series_panel._split.isEnabled()

    def test_undo_restores_series_colour_and_mask(self, window):
        """The snapshot fix: a restored series must be the same series."""
        series = window._result.series[1]
        series.mask = np.ones((6, 6), dtype=bool)
        colour, mask = series.color, series.mask

        window._on_delete_series(1)
        window.undo()

        restored = window._result.series[1]
        assert restored.color == colour
        assert restored.mask is mask, "mask must survive an undo, or re-extraction breaks"

    def test_project_round_trip_keeps_the_ability_to_split(self, window, tmp_path):
        from plotdigitizer.project import load_project, save_project
        result = window._result
        result.series[:2] = [combine_series(result.series[:2],
                                            calibration=result.calibration)]

        path = save_project(tmp_path / "s.pdproj", result, window._image_path)
        restored, _ = load_project(path)

        assert restored.series[0].is_combined
        parts = split_series(restored.series[0], calibration=restored.calibration)
        assert len(parts) == 2
        assert np.allclose(parts[0].pixel_points,
                           result.series[0].sources[0]["pixel_points"])


class TestGuidedSweep:
    """Alt+drag along a curve: the path says which curve, the columns say how densely."""

    @pytest.fixture
    def curve_window(self, qapp, corpus, _digitized):
        import copy

        from plotdigitizer.ui.main_window import MainWindow
        win = MainWindow(device="cpu")
        figure = corpus["linear_line"]
        win._adopt_image(figure.image, figure.path)
        win._on_digitized(copy.deepcopy(_digitized["linear_line"]))
        yield win
        win._set_dirty(False)
        win.close()

    @staticmethod
    def _sweep(win, path_points, modifier=Qt.KeyboardModifier.AltModifier):
        from PySide6.QtGui import QMouseEvent
        canvas = win.canvas

        def event(kind, scene_point, button, mods, buttons=None):
            view_point = canvas.mapFromScene(scene_point)
            return QMouseEvent(kind, QPointF(view_point),
                               canvas.viewport().mapToGlobal(view_point), button,
                               button if buttons is None else buttons, mods)

        canvas.mousePressEvent(event(QMouseEvent.Type.MouseButtonPress, path_points[0],
                                     Qt.MouseButton.LeftButton, modifier))
        for point in path_points[1:]:
            canvas.mouseMoveEvent(event(QMouseEvent.Type.MouseMove, point,
                                        Qt.MouseButton.NoButton, modifier,
                                        Qt.MouseButton.LeftButton))
        canvas.mouseReleaseEvent(event(QMouseEvent.Type.MouseButtonRelease, path_points[-1],
                                       Qt.MouseButton.LeftButton, modifier))

    @staticmethod
    def _path_along(win, step=40):
        truth = win._result.series[0].pixel_points
        return [QPointF(float(x), float(y)) for x, y in truth[::step]]

    def test_a_sweep_makes_its_own_series(self, curve_window):
        before = len(curve_window._result.series)
        self._sweep(curve_window, self._path_along(curve_window))
        assert len(curve_window._result.series) == before + 1
        assert curve_window._result.series[-1].name == "Segment 1"

    def test_swept_points_land_on_the_curve(self, curve_window):
        truth = curve_window._result.series[0].pixel_points
        self._sweep(curve_window, self._path_along(curve_window))
        swept = curve_window._result.series[-1].pixel_points
        expected = np.interp(swept[:, 0], truth[:, 0], truth[:, 1])
        assert np.abs(swept[:, 1] - expected).mean() < 2.0

    def test_ctrl_works_too(self, curve_window):
        """Alt+drag is grabbed by some desktops to move windows; Ctrl is the way out."""
        before = len(curve_window._result.series)
        self._sweep(curve_window, self._path_along(curve_window),
                    modifier=Qt.KeyboardModifier.ControlModifier)
        assert len(curve_window._result.series) == before + 1

    def test_a_plain_drag_still_lays_the_old_trail(self, curve_window):
        """Without the modifier nothing changes: no new series, points join the active one.

        The path is offset off the curve, because a drag that *starts* on an existing
        point moves that point instead - which is also unchanged behaviour.
        """
        before = len(curve_window._result.series)
        count_before = curve_window._result.series[0].count
        offset = [QPointF(p.x(), p.y() + 60.0) for p in self._path_along(curve_window)]
        self._sweep(curve_window, offset, modifier=Qt.KeyboardModifier.NoModifier)
        assert len(curve_window._result.series) == before
        assert curve_window._result.series[0].count > count_before

    def test_a_plain_drag_starting_on_a_point_still_moves_it(self, curve_window):
        before = len(curve_window._result.series)
        self._sweep(curve_window, self._path_along(curve_window),
                    modifier=Qt.KeyboardModifier.NoModifier)
        assert len(curve_window._result.series) == before, "must not create a series"

    def test_several_sweeps_stack_up_then_combine(self, curve_window):
        truth = curve_window._result.series[0].pixel_points
        third = len(truth) // 3
        for start, end in ((0, third), (third, 2 * third), (2 * third, len(truth) - 1)):
            path = [QPointF(float(x), float(y)) for x, y in truth[start:end:20]]
            if len(path) >= 2:
                self._sweep(curve_window, path)

        segments = [s for s in curve_window._result.series if s.name.startswith("Segment")]
        assert len(segments) == 3
        assert [s.name for s in segments] == ["Segment 1", "Segment 2", "Segment 3"]

        combined = combine_series(segments, name="whole curve",
                                  calibration=curve_window._result.calibration)
        assert combined.count == sum(s.count for s in segments)
        assert len(split_series(combined)) == 3

    def test_a_sweep_is_one_undo_step(self, curve_window):
        before = len(curve_window._result.series)
        self._sweep(curve_window, self._path_along(curve_window))
        curve_window.undo()
        assert len(curve_window._result.series) == before

    def test_sweeping_empty_space_adds_nothing(self, curve_window):
        before = len(curve_window._result.series)
        blank = [QPointF(120.0, 70.0), QPointF(200.0, 70.0), QPointF(280.0, 70.0)]
        self._sweep(curve_window, blank)
        assert len(curve_window._result.series) == before

    def test_the_overlay_is_cleared_afterwards(self, curve_window):
        self._sweep(curve_window, self._path_along(curve_window))
        assert not curve_window.canvas._lasso_item.isVisible()
        assert curve_window.canvas._sweep_points == []


class TestCanvasHelpers:
    def test_cursor_readout_reports_data_coordinates(self, window):
        window._on_cursor_moved(300.0, 200.0)
        assert "data (" in window.data_panel.parent().window().status_data.text()

    def test_cursor_off_image_clears_the_readout(self, window):
        window._on_cursor_moved(float("nan"), float("nan"))
        assert window.status_position.text() == "-"

    def test_nearest_point_pick(self, window):
        item = window.canvas._series_items[0]
        points = item.points()
        target = points[4]
        assert item.nearest(target[0] + 1.0, target[1] + 1.0, 9.0) == 4
        assert item.nearest(target[0] + 500.0, target[1], 9.0) == -1
