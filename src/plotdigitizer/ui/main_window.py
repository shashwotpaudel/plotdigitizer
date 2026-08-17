"""The application window: toolbar, canvas, magnifier and the four step panels.

The flow mirrors plotdigitizer.com - calibrate, pick series, check the data, export -
with the difference that opening an image runs the detector first, so the user starts
at "is this right?" instead of "here is an empty image, start clicking".

Undo is snapshot based. The editable state is a handful of small arrays plus a
calibration, so copying it wholesale before each mutation costs nothing and removes an
entire category of bug that per-command undo/redo invites.
"""

from __future__ import annotations

import copy
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend import describe_devices, select_backend
from ..calibration import Calibration
from ..compose import combine_series, split_series
from ..detect.extract import ExtractionMode, ExtractionSettings
from ..detect.frame import analyse_ink
from ..detect.outliers import select_outliers
from ..detect.snap import snap_to_ink
from ..detect.trace import stroke_mask, sweep_along, trace_stroke
from ..export import csv_string, read_csv_series, write_csv
from ..image_io import SUPPORTED_SUFFIXES, file_dialog_filter, load_image
from ..pipeline import DigitizationResult, Series
from ..project import PROJECT_SUFFIX, load_project, save_project
from .canvas import PlotCanvas, qimage_to_numpy
from .magnifier import Magnifier
from .panels.calibrate import CalibratePanel
from .panels.data_table import DataPanel
from .panels.export import ExportPanel
from .panels.series import SeriesPanel
from .selection_bar import SelectionBar
from .worker import DigitizeWorker, ExtractionWorker

log = logging.getLogger(__name__)

__all__ = ["MainWindow"]

UNDO_DEPTH = 40
RECENT_LIMIT = 10
AUTOSAVE_SECONDS = 30

#: Overlay colours for series the user creates. Deliberately not sampled from the
#: figure: a hand-made or traced series has to stand out *against* the artwork, and on
#: a monochrome plot the ink colour it came from would make it invisible.
NEW_SERIES_PALETTE = [
    (214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189),
    (255, 127, 14), (23, 190, 207),
]


class MainWindow(QMainWindow):
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.setWindowTitle("Plot Digitizer")
        self.resize(1360, 880)

        self._device = device
        self._backend = select_backend(device)
        self._result: DigitizationResult | None = None
        self._image: np.ndarray | None = None
        self._image_path: Path | None = None
        self._active = 0
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._worker: DigitizeWorker | None = None
        self._ink = None                 # InkImage, for snapping and stroke tracing
        self._dirty = False
        self._snap_enabled = True
        self._selection: set[int] = set()
        self._extractor = ExtractionWorker(self._backend, self)
        self._extractor.ready.connect(self._on_reextracted)

        self.setAcceptDrops(True)

        self._build_canvas()
        self._build_docks()
        self._build_toolbar()
        self._build_status()
        self._update_actions()
        self._restore_window_state()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(AUTOSAVE_SECONDS * 1000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

    # -- construction ------------------------------------------------------------

    def _build_canvas(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PlotCanvas()
        layout.addWidget(self.canvas, 1)

        self.selection_bar = SelectionBar()
        self.selection_bar.moveRequested.connect(self.move_selection_to)
        self.selection_bar.deleteRequested.connect(self.delete_selection)
        self.selection_bar.keepOnlyRequested.connect(self.keep_only_selection)
        self.selection_bar.invertRequested.connect(self.invert_selection)
        self.selection_bar.clearRequested.connect(self._clear_selection)
        layout.addWidget(self.selection_bar)

        self.setCentralWidget(central)

        self.canvas.cursorMoved.connect(self._on_cursor_moved)
        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        self.canvas.pointAdded.connect(self._on_point_added)
        self.canvas.pointMoved.connect(self._on_point_moved)
        self.canvas.pointRemoved.connect(self._on_point_removed)
        self.canvas.pointSelected.connect(self._on_point_selected)
        self.canvas.handleDragged.connect(self._on_handle_dragged)
        self.canvas.handleReleased.connect(lambda _: self.end_edit())
        self.canvas.pointTraced.connect(self._on_point_traced)
        self.canvas.pointsRemoved.connect(self._on_points_removed)
        self.canvas.pointNudged.connect(self._on_point_nudged)
        self.canvas.strokeSeeded.connect(self._on_stroke_seeded)
        self.canvas.sweepFinished.connect(self._on_sweep_finished)
        self.canvas.editFinished.connect(self.end_edit)
        self.canvas.selectionChanged.connect(self._on_selection_changed)

    def _build_docks(self) -> None:
        self.tabs = QTabWidget()

        self.calibrate_panel = CalibratePanel()
        self.calibrate_panel.calibrationEdited.connect(self._on_calibration_edited)
        self.calibrate_panel.recalibrateRequested.connect(self.auto_digitize)
        self.tabs.addTab(self.calibrate_panel, "1 Calibrate")

        self.series_panel = SeriesPanel()
        self.series_panel.activeChanged.connect(self._on_active_series)
        self.series_panel.visibilityChanged.connect(self._on_series_visibility)
        self.series_panel.nameChanged.connect(self._on_series_renamed)
        self.series_panel.settingsChanged.connect(self._on_series_settings)
        self.series_panel.addSeriesRequested.connect(self._on_add_series)
        self.series_panel.deleteSeriesRequested.connect(self._on_delete_series)
        self.series_panel.combineRequested.connect(self.combine_series_dialog)
        self.series_panel.splitRequested.connect(self.split_series_at)
        self.series_panel.selectSpikesRequested.connect(self.select_spikes)
        self.tabs.addTab(self.series_panel, "2 Series")

        self.data_panel = DataPanel()
        self.data_panel.valueEdited.connect(self._on_value_edited)
        self.data_panel.pointDeleted.connect(self._on_point_removed)
        self.data_panel.rowSelected.connect(self._on_row_selected)
        self.tabs.addTab(self.data_panel, "3 Data")

        self.export_panel = ExportPanel()
        self.export_panel.optionsChanged.connect(self._refresh_export_preview)
        self.export_panel.exportRequested.connect(self.export_csv)
        self.export_panel.copyRequested.connect(self.copy_to_clipboard)
        self.export_panel.saveProjectRequested.connect(self.save_session)
        self.export_panel.openProjectRequested.connect(self.open_session)
        self.export_panel.importCsvRequested.connect(self.import_csv)
        self.tabs.addTab(self.export_panel, "4 Export")

        dock = QDockWidget("Digitize", self)
        dock.setObjectName("steps")
        dock.setWidget(self.tabs)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.magnifier = Magnifier()
        zoom_dock = QDockWidget("Zoom", self)
        zoom_dock.setObjectName("zoom")
        zoom_dock.setWidget(self.magnifier)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, zoom_dock)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setObjectName("main-toolbar")
        bar.setMovable(False)
        self.addToolBar(bar)

        self.action_open = QAction("Open image", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self.open_image)
        bar.addAction(self.action_open)

        self.recent_menu = QMenu("Open recent", self)
        recent_button = QToolButton(self)
        recent_button.setText("Recent")
        recent_button.setMenu(self.recent_menu)
        recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bar.addWidget(recent_button)

        self.action_auto = QAction("Auto-digitize", self)
        self.action_auto.setShortcut("Ctrl+D")
        self.action_auto.triggered.connect(self.auto_digitize)
        bar.addAction(self.action_auto)
        bar.addSeparator()

        self.action_undo = QAction("Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.triggered.connect(self.undo)
        bar.addAction(self.action_undo)

        self.action_redo = QAction("Redo", self)
        self.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.action_redo.triggered.connect(self.redo)
        bar.addAction(self.action_redo)
        bar.addSeparator()

        fit = QAction("Fit", self)
        fit.setShortcut("Ctrl+0")
        fit.triggered.connect(self.canvas.zoom_to_fit)
        bar.addAction(fit)

        actual = QAction("100%", self)
        actual.setShortcut("Ctrl+1")
        actual.triggered.connect(self.canvas.zoom_to_actual)
        bar.addAction(actual)
        bar.addSeparator()

        self.action_snap = QAction("Snap", self)
        self.action_snap.setCheckable(True)
        self.action_snap.setChecked(True)
        self.action_snap.setToolTip(
            "Pull clicked points onto the centre of the nearest stroke")
        self.action_snap.toggled.connect(self._set_snap_enabled)
        bar.addAction(self.action_snap)

        self.lasso_menu = QMenu("Lasso shape", self)
        self._lasso_button = QToolButton(self)
        self._lasso_button.setText("Lasso: box")
        self._lasso_button.setMenu(self.lasso_menu)
        self._lasso_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._lasso_button.setToolTip(
            "Shape drawn by a right-drag to select points.\n"
            "Shift adds to the selection, Ctrl removes from it, Esc clears.")
        for label, shape in (("Box", "rectangle"), ("Freeform", "freeform")):
            action = self.lasso_menu.addAction(label)
            action.triggered.connect(
                lambda _=False, sh=shape, lb=label: self._set_lasso_shape(sh, lb))
        bar.addWidget(self._lasso_button)

        self.action_trace = QAction("Trace stroke", self)
        self.action_trace.setCheckable(True)
        self.action_trace.setShortcut("Ctrl+T")
        self.action_trace.setToolTip(
            "Click a curve to follow it into its own series (Ctrl+T)")
        self.action_trace.toggled.connect(self._set_trace_armed)
        bar.addAction(self.action_trace)
        bar.addSeparator()

        self.action_show_points = QAction("Points", self)
        self.action_show_points.setCheckable(True)
        self.action_show_points.setChecked(True)
        self.action_show_points.toggled.connect(
            lambda on: self.canvas.set_points_visible(
                on, self._result.series if self._result else None))
        bar.addAction(self.action_show_points)

        self.action_show_handles = QAction("Handles", self)
        self.action_show_handles.setCheckable(True)
        self.action_show_handles.setChecked(True)
        self.action_show_handles.toggled.connect(self.canvas.set_handles_visible)
        bar.addAction(self.action_show_handles)

        self.action_show_frame = QAction("Frame", self)
        self.action_show_frame.setCheckable(True)
        self.action_show_frame.setChecked(True)
        self.action_show_frame.toggled.connect(self.canvas.set_frame_visible)
        bar.addAction(self.action_show_frame)
        bar.addSeparator()

        self.action_export = QAction("Export CSV", self)
        self.action_export.setShortcut("Ctrl+E")
        self.action_export.triggered.connect(self.export_csv)
        bar.addAction(self.action_export)

        paste = QAction("Paste image", self)
        paste.setShortcut(QKeySequence.StandardKey.Paste)
        paste.triggered.connect(self.paste_image)
        self.addAction(paste)

        delete = QAction("Delete point", self)
        delete.setShortcut(QKeySequence.StandardKey.Delete)
        delete.triggered.connect(self._delete_selected_point)
        self.addAction(delete)

    def _build_status(self) -> None:
        self.status_position = QLabel("-")
        self.status_data = QLabel("-")
        self.status_zoom = QLabel("100%")
        self.status_device = QLabel(describe_devices().name)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(140)
        self.progress.hide()

        bar = self.statusBar()
        bar.addWidget(self.status_position)
        bar.addWidget(self.status_data, 1)
        bar.addPermanentWidget(self.progress)
        bar.addPermanentWidget(self.status_zoom)
        bar.addPermanentWidget(self.status_device)
        bar.showMessage("Open an image to begin.", 5000)

    # -- persistence -------------------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings("plotdigitizer", "plotdigitizer")

    def _restore_window_state(self) -> None:
        settings = self._settings()
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("window_state")
        if state:
            self.restoreState(state)
        self.export_panel.from_settings(settings.value("export_options"))
        self._refresh_recent_menu()

    def _save_window_state(self) -> None:
        settings = self._settings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("window_state", self.saveState())

    def _remember_recent(self, path: Path) -> None:
        settings = self._settings()
        recent = [str(path)] + [p for p in settings.value("recent", [], type=list)
                                if p != str(path)]
        settings.setValue("recent", recent[:RECENT_LIMIT])
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        menu = self.recent_menu
        menu.clear()
        entries = [p for p in self._settings().value("recent", [], type=list)
                   if Path(p).exists()]
        if not entries:
            action = menu.addAction("Nothing opened yet")
            action.setEnabled(False)
            return
        for entry in entries:
            path = Path(entry)
            action = menu.addAction(path.name)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _=False, p=path: self._open_recent(p))

    def _open_recent(self, path: Path) -> None:
        if path.suffix.lower() == PROJECT_SUFFIX:
            self.open_session(path)
        else:
            self.load_image(path)

    # -- autosave ----------------------------------------------------------------

    def _autosave_path(self) -> Path:
        cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        directory = Path(cache) / "plot_digitizer" / "autosave"
        stem = self._image_path.stem if self._image_path else "untitled"
        return directory / f"{stem}{PROJECT_SUFFIX}"

    def _autosave(self) -> None:
        """Write a recovery copy while there is unsaved work.

        Cheap insurance: a digitizing session is a lot of manual effort to lose to a
        crash or a stray close, and the project format is small enough that writing it
        every half minute costs nothing.
        """
        if not self._dirty or self._result is None:
            return
        try:
            save_project(self._autosave_path(), self._result, self._image_path)
        except Exception as exc:                          # noqa: BLE001
            log.debug("autosave failed: %s", exc)

    def _offer_recovery(self) -> None:
        """If an autosave is newer than the image, offer to pick up where we left off."""
        path = self._autosave_path()
        if not path.exists() or self._image_path is None:
            return
        if path.stat().st_mtime <= self._image_path.stat().st_mtime:
            return
        answer = QMessageBox.question(
            self, "Recover unsaved session?",
            f"An unsaved session for {self._image_path.name} was found from "
            f"{datetime.fromtimestamp(path.stat().st_mtime):%H:%M on %d %b}.\n\n"
            f"Restore it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.open_session(path)

    # -- modes -------------------------------------------------------------------

    def _set_snap_enabled(self, enabled: bool) -> None:
        self._snap_enabled = enabled
        self.statusBar().showMessage(
            "Snapping on: clicks land on the nearest stroke." if enabled
            else "Snapping off: points land exactly where you click.", 4000)

    def _set_lasso_shape(self, shape: str, label: str) -> None:
        self.canvas.set_lasso_shape(shape)
        self._lasso_button.setText(f"Lasso: {label.lower()}")

    def _set_trace_armed(self, armed: bool) -> None:
        self.canvas.begin_stroke_seed(armed)
        if armed:
            self.statusBar().showMessage(
                "Trace: click a curve to follow it into a new series.", 6000)

    # -- file actions ------------------------------------------------------------

    def dragEnterEvent(self, event):
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._dropped_path(event)
        if path is None:
            return
        event.acceptProposedAction()
        if path.suffix.lower() == PROJECT_SUFFIX:
            self.open_session(path)
        else:
            self.load_image(path)

    @staticmethod
    def _dropped_path(event) -> Path | None:
        data = event.mimeData()
        if not data.hasUrls():
            return None
        for url in data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in (*SUPPORTED_SUFFIXES, PROJECT_SUFFIX):
                return path
        return None

    def paste_image(self) -> None:
        """Load an image straight from the clipboard - screenshots need no detour."""
        image = QGuiApplication.clipboard().image()
        if image.isNull():
            self.statusBar().showMessage("No image on the clipboard.", 4000)
            return
        if not self._confirm_discard("Pasting an image"):
            return
        self._adopt_image(qimage_to_numpy(image), None)
        self.auto_digitize()

    def open_image(self) -> None:
        settings = QSettings("plotdigitizer", "plotdigitizer")
        start = settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open plot image", start,
                                              file_dialog_filter())
        if path:
            settings.setValue("last_dir", str(Path(path).parent))
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        if not self._confirm_discard("Opening another image"):
            return
        try:
            image = load_image(path)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, "Could not open image", f"{path.name}:\n{exc}")
            return
        self._adopt_image(image, Path(path))
        self._remember_recent(Path(path))
        self.auto_digitize()
        self._offer_recovery()

    def _adopt_image(self, image: np.ndarray, path: Path | None) -> None:
        """Install a new image and reset the per-image state around it."""
        self._image = image
        self._image_path = path
        self._undo.clear()
        self._redo.clear()
        self._set_dirty(False)
        # Kept for snapping and stroke tracing; a single pass over the image.
        self._ink = analyse_ink(image)
        self.canvas.set_image(image)
        self.magnifier.set_image(image)
        self.setWindowTitle(f"Plot Digitizer - {path.name if path else 'pasted image'}")

    def auto_digitize(self) -> None:
        if self._image is None or (self._worker is not None and self._worker.isRunning()):
            return
        if not self._confirm_replace_manual_work():
            return
        self.progress.show()
        self.statusBar().showMessage("Detecting axes and data...")
        self.action_auto.setEnabled(False)

        self._worker = DigitizeWorker(self._image, self._device)
        self._worker.finished_ok.connect(self._on_digitized)
        self._worker.failed.connect(self._on_digitize_failed)
        self._worker.finished.connect(lambda: (self.progress.hide(),
                                               self.action_auto.setEnabled(True)))
        self._worker.start()

    def _on_digitize_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Automatic digitization failed", message)
        self.statusBar().showMessage("Automatic digitization failed.", 8000)

    def _on_digitized(self, result: DigitizationResult) -> None:
        # Re-running detection over existing work is an edit like any other, so it goes
        # on the undo stack instead of clearing it. Wiping an hour of hand-tracing with
        # no way back is not a reasonable outcome for pressing "Re-detect axes".
        if self._result is not None:
            self._push_undo()
        self._result = result
        self._active = 0
        self._sync_all()

        summary = (f"Found {len(result.series)} series, {result.total_points} points "
                   f"in {result.elapsed_seconds:.2f}s on {result.device.upper()}.")
        self.statusBar().showMessage(summary, 10000)
        if result.warnings:
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "Check before exporting",
                                    summary + "\n\n- " + "\n- ".join(result.warnings))

    # -- synchronisation ---------------------------------------------------------

    def _sync_all(self) -> None:
        if self._result is None:
            return
        result = self._result
        self.canvas.set_frame(result.frame.left, result.frame.top,
                              result.frame.right, result.frame.bottom)
        self.canvas.set_series(result.series)
        self.canvas.set_active_series(self._active)
        self._sync_handles()
        self.calibrate_panel.set_calibration(result.calibration, result.confidence)
        self.series_panel.set_series(result.series, self._active)
        self._sync_data_panel()
        self._refresh_export_preview()
        self._update_actions()

    def _sync_handles(self) -> None:
        if self._result is None:
            return
        calibration = self._result.calibration
        frame = self._result.frame
        self.canvas.set_handles(
            {"x1": calibration.x.p1, "x2": calibration.x.p2,
             "y1": calibration.y.p1, "y2": calibration.y.p2},
            span_x=(frame.left, frame.right),
            span_y=(frame.top, frame.bottom),
        )

    def _sync_data_panel(self) -> None:
        if self._result is None or not self._result.series:
            self.data_panel.set_series(None, 0)
            return
        index = min(self._active, len(self._result.series) - 1)
        self.data_panel.set_series(
            self._result.series[index], index,
            scales=(self._result.calibration.x.scale.value,
                    self._result.calibration.y.scale.value),
        )

    def _sync_selection_bar(self) -> None:
        series = self._series_at(self._active)
        self.selection_bar.update_state(
            len(self._selection), series.count if series else 0,
            self._result.series if self._result else [], self._active)

    def _refresh_series_views(self) -> None:
        if self._result is None:
            return
        self.canvas.set_series(self._result.series)
        self.canvas.set_active_series(self._active)
        self.canvas.refresh_visibility(self._result.series)
        self.series_panel.refresh_counts(self._result.series)
        self.canvas.set_selection(sorted(self._selection))
        self._sync_selection_bar()
        self._sync_data_panel()
        self._refresh_export_preview()

    def _refresh_export_preview(self) -> None:
        if self._result is None:
            self.export_panel.set_preview("")
            return
        try:
            text = csv_string(self._result.series, self.export_panel.options())
        except Exception as exc:                          # noqa: BLE001
            text = f"(preview unavailable: {exc})"
        self.export_panel.set_preview(text)

    def _update_actions(self) -> None:
        has_result = self._result is not None
        self.action_export.setEnabled(has_result)
        self.action_auto.setEnabled(self._image is not None)
        self.action_undo.setEnabled(bool(self._undo))
        self.action_redo.setEnabled(bool(self._redo))

    # -- undo --------------------------------------------------------------------

    # -- selection ---------------------------------------------------------------

    def _clear_selection(self) -> None:
        if self._selection:
            self._selection = set()
        self.canvas.set_selection([])
        self._sync_selection_bar()

    def _set_selection(self, indices) -> None:
        series = self._series_at(self._active)
        limit = series.count if series else 0
        self._selection = {int(i) for i in indices if 0 <= int(i) < limit}
        self.canvas.set_selection(sorted(self._selection))
        self._sync_selection_bar()

    def _shift_selection_for_insert(self, position: int) -> None:
        """A point was inserted at ``position``; everything at or after it moved up one.

        The selection is remapped rather than dropped wherever the shift is known
        exactly. Guessing is the one thing that must not happen: a selection silently
        pointing at the wrong rows would delete the wrong data on the next action.
        """
        if self._selection:
            self._selection = {i + 1 if i >= position else i for i in self._selection}

    def _shift_selection_for_removal(self, removed, count_before: int) -> None:
        """Points at ``removed`` are gone; drop them and close the gaps.

        ``count_before`` is how many points there were *before* the deletion, because
        the renumbering has to be worked out against the indices the selection was
        recorded against.
        """
        if not self._selection:
            return
        gone = set(int(i) for i in removed)
        remaining = [i for i in range(count_before) if i not in gone]
        renumber = {old: new for new, old in enumerate(remaining)}
        self._selection = {renumber[i] for i in self._selection - gone if i in renumber}

    def _on_selection_changed(self, series_index: int, indices) -> None:
        if series_index != self._active:
            return
        self._set_selection(indices)

    def _snapshot(self) -> dict:
        """Everything an undo has to put back.

        Colour, mask and provenance belong here as much as the points do: without them
        undoing a deletion hands back a differently-coloured series that can no longer
        be re-extracted, and a combined one that can no longer be split - a "restore"
        that quietly loses part of what it restored.

        The mask is stored by reference rather than copied. Each is around a megabyte,
        and re-extraction replaces one wholesale rather than mutating it, so copying
        them into forty undo levels would cost hundreds of megabytes to preserve
        something that never changes underneath us.
        """
        assert self._result is not None
        return {
            "calibration": self._result.calibration.to_dict(),
            "series": [
                {
                    "name": s.name,
                    "color": tuple(s.color),
                    "visible": s.visible,
                    "settings": s.settings.copy(),
                    "points": s.pixel_points.copy(),
                    "mask": s.mask,
                    "sources": copy.deepcopy(s.sources),
                }
                for s in self._result.series
            ],
            "active": self._active,
        }

    def _restore(self, snapshot: dict) -> None:
        if self._result is None:
            return
        self._result.calibration = Calibration.from_dict(snapshot["calibration"])
        # Series may have been added or removed since; rebuild the list to match.
        while len(self._result.series) > len(snapshot["series"]):
            self._result.series.pop()
        for index, entry in enumerate(snapshot["series"]):
            if index < len(self._result.series):
                series = self._result.series[index]
            else:
                series = Series(name=entry["name"], color=tuple(entry["color"]),
                                settings=entry["settings"].copy(),
                                pixel_points=np.empty((0, 2)),
                                data_points=np.empty((0, 2)))
                self._result.series.append(series)
            series.name = entry["name"]
            series.color = tuple(entry["color"])
            series.visible = entry["visible"]
            series.settings = entry["settings"].copy()
            series.pixel_points = entry["points"].copy()
            series.mask = entry.get("mask")
            series.sources = copy.deepcopy(entry.get("sources"))
        self._active = snapshot["active"]
        # Point indices have just been rewritten wholesale, so any selection held
        # against the old ones is meaningless - and a selection pointing at the wrong
        # rows is worse than none at all.
        self._clear_selection()
        self._result.recompute()
        self._sync_all()

    def _push_undo(self, coalesce: str | None = None) -> None:
        """Record the state *before* the next edit. ``coalesce`` merges a drag."""
        if self._result is None:
            return
        self._set_dirty(True)
        if coalesce and self._undo and self._undo[-1].get("_tag") == coalesce:
            return
        snapshot = self._snapshot()
        snapshot["_tag"] = coalesce
        self._undo.append(snapshot)
        del self._undo[:-UNDO_DEPTH]
        self._redo.clear()
        self._update_actions()

    def end_edit(self) -> None:
        """Close the current coalescing group, so the next edit is its own undo step."""
        if self._undo:
            self._undo[-1]["_tag"] = None

    # -- unsaved work ------------------------------------------------------------

    def _set_dirty(self, dirty: bool) -> None:
        if dirty == self._dirty:
            return
        self._dirty = dirty
        title = self.windowTitle().removesuffix(" *")
        self.setWindowTitle(f"{title} *" if dirty else title)

    def _manual_point_count(self) -> int:
        return self._result.total_points if self._result else 0

    def _confirm_discard(self, action: str) -> bool:
        """Ask before throwing away unsaved edits. True means go ahead."""
        if not self._dirty or self._result is None:
            return True
        answer = QMessageBox.warning(
            self, "Unsaved changes",
            f"{action} will discard your unsaved edits "
            f"({self._manual_point_count()} points).\n\nSave the session first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            self.save_session()
            return not self._dirty            # a cancelled save aborts the whole action
        return True

    def _confirm_replace_manual_work(self) -> bool:
        """Re-running detection replaces every series; say so when that costs something."""
        if not self._dirty or self._result is None:
            return True
        answer = QMessageBox.question(
            self, "Re-detect from scratch?",
            f"Automatic detection replaces all {self._manual_point_count()} current "
            f"points, including anything you placed by hand.\n\n"
            f"This can be undone with Ctrl+Z.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Ok

    def undo(self) -> None:
        if not self._undo or self._result is None:
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self._update_actions()

    def redo(self) -> None:
        if not self._redo or self._result is None:
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self._update_actions()

    # -- canvas interaction ------------------------------------------------------

    def _on_cursor_moved(self, column: float, row: float) -> None:
        self.magnifier.set_centre(column, row)
        if np.isnan(column):
            self.status_position.setText("-")
            self.status_data.setText("-")
            return
        self.status_position.setText(f"px ({column:.1f}, {row:.1f})")
        if self._result is not None and self._result.calibration.is_valid:
            point = self._result.calibration.to_data(np.array([[column, row]]))[0]
            self.status_data.setText(f"data ({point[0]:.6g}, {point[1]:.6g})")
        else:
            self.status_data.setText("data (uncalibrated)")

    def _on_zoom_changed(self, zoom: float) -> None:
        self.status_zoom.setText(f"{zoom * 100:.0f}%")

    def _series_at(self, index: int) -> Series | None:
        if self._result is None or not (0 <= index < len(self._result.series)):
            return None
        return self._result.series[index]

    def _snap(self, column: float, row: float) -> tuple[float, float]:
        """Pull a hand-placed position onto the stroke it was aimed at."""
        if not self._snap_enabled or self._ink is None:
            return column, row
        result = snap_to_ink(self._ink, column, row)
        return result.column, result.row

    def _insert_point(self, series, column: float, row: float) -> None:
        points = np.vstack([series.pixel_points, [column, row]]) \
            if series.pixel_points.size else np.array([[column, row]], dtype=float)
        order = np.argsort(points[:, 0], kind="stable")
        series.pixel_points = points[order]
        series.recompute_data(self._result.calibration)
        if series is self._series_at(self._active):
            # The new point was appended then sorted into place; wherever it landed,
            # every selected index at or beyond it has shifted up by one.
            self._shift_selection_for_insert(int(np.flatnonzero(order == points.shape[0] - 1)[0]))

    def _on_point_added(self, index: int, column: float, row: float) -> None:
        series = self._series_at(index)
        if series is None:
            return
        self._push_undo()
        self._insert_point(series, *self._snap(column, row))
        self._refresh_series_views()

    def _on_point_traced(self, index: int, column: float, row: float) -> None:
        """A point laid down mid-drag; the whole drag is a single undo step."""
        series = self._series_at(index)
        if series is None:
            return
        self._push_undo(coalesce=f"trace:{index}")
        self._insert_point(series, *self._snap(column, row))
        self.canvas.update_series_points(index, series.pixel_points)
        self.series_panel.refresh_counts(self._result.series)

    def _on_point_moved(self, index: int, point_index: int, column: float, row: float) -> None:
        series = self._series_at(index)
        if series is None or not (0 <= point_index < series.pixel_points.shape[0]):
            return
        self._push_undo(coalesce=f"move:{index}:{point_index}")
        series.pixel_points[point_index] = self._snap(column, row)
        series.recompute_data(self._result.calibration)
        self.canvas.update_series_points(index, series.pixel_points)
        self.data_panel.refresh()

    def _on_point_nudged(self, index: int, point_index: int, dx: float, dy: float) -> None:
        """Arrow-key adjustment. Never snapped - the point of nudging is exact control."""
        series = self._series_at(index)
        if series is None or not (0 <= point_index < series.pixel_points.shape[0]):
            return
        self._push_undo(coalesce=f"nudge:{index}:{point_index}")
        series.pixel_points[point_index] += (dx, dy)
        series.recompute_data(self._result.calibration)
        self.canvas.update_series_points(index, series.pixel_points)
        self.data_panel.refresh()
        self._refresh_export_preview()

    def _on_points_removed(self, index: int, indices) -> None:
        """Delete several points in one step."""
        series = self._series_at(index)
        if series is None or not len(indices):
            return
        self._push_undo()
        before = series.count
        series.pixel_points = np.delete(series.pixel_points, list(indices), axis=0)
        series.recompute_data(self._result.calibration)
        if index == self._active:
            self._shift_selection_for_removal(indices, before)
        self._refresh_series_views()
        self.statusBar().showMessage(f"Deleted {len(indices)} points", 4000)

    # -- selection actions -------------------------------------------------------

    def delete_selection(self) -> None:
        if not self._selection:
            return
        count = len(self._selection)
        self._on_points_removed(self._active, sorted(self._selection))
        self.statusBar().showMessage(f"Deleted {count} selected points", 5000)

    def keep_only_selection(self) -> None:
        """Delete everything *except* the selection."""
        series = self._series_at(self._active)
        if series is None or not self._selection:
            return
        others = [i for i in range(series.count) if i not in self._selection]
        if not others:
            return
        kept = series.count - len(others)
        self._on_points_removed(self._active, others)
        self.statusBar().showMessage(f"Kept {kept} points, removed {len(others)}", 5000)

    def invert_selection(self) -> None:
        series = self._series_at(self._active)
        if series is None:
            return
        self._set_selection(set(range(series.count)) - self._selection)

    def move_selection_to(self, target_index: int) -> None:
        """Hand the selected points to another series.

        The direct remedy for a trace that strayed onto a neighbouring curve: the
        points are not judged wrong and destroyed, they are filed where they belong.
        ``target_index`` of -1 creates a new series for them.
        """
        source = self._series_at(self._active)
        if source is None or not self._selection:
            return

        indices = sorted(self._selection)
        moving = source.pixel_points[indices].copy()
        total_before = sum(s.count for s in self._result.series)

        self._push_undo()

        if target_index < 0:
            target = Series(
                name=f"Series {len(self._result.series) + 1}",
                color=NEW_SERIES_PALETTE[len(self._result.series) % len(NEW_SERIES_PALETTE)],
                settings=source.settings.copy(),
                pixel_points=np.empty((0, 2), dtype=float),
                data_points=np.empty((0, 2), dtype=float),
            )
            self._result.series.append(target)
        else:
            target = self._series_at(target_index)
            if target is None or target is source:
                return

        combined = np.vstack([target.pixel_points, moving]) if target.pixel_points.size \
            else moving
        target.pixel_points = combined[np.argsort(combined[:, 0], kind="stable")]
        target.recompute_data(self._result.calibration)

        before = source.count
        source.pixel_points = np.delete(source.pixel_points, indices, axis=0)
        source.recompute_data(self._result.calibration)
        self._shift_selection_for_removal(indices, before)

        assert sum(s.count for s in self._result.series) == total_before, \
            "moving points must not create or lose any"

        self._sync_all()
        self.statusBar().showMessage(
            f"Moved {len(indices)} points from {source.name} to {target.name}", 6000)

    def select_spikes(self, sensitivity: float) -> None:
        """Select points that look like they belong to a different curve."""
        from ..detect.outliers import MIN_POINTS

        series = self._series_at(self._active)
        if series is None:
            return
        if series.count < MIN_POINTS:
            self._clear_selection()
            self.statusBar().showMessage(
                f"{series.name} has too few points ({series.count}) to tell a stray "
                f"from the shape of the curve - select them by hand instead.", 6000)
            return

        indices = select_outliers(series.pixel_points, sensitivity=sensitivity)
        self._set_selection(indices)
        self.statusBar().showMessage(
            f"{len(indices)} points look out of place - nothing has changed yet", 5000)

    def _on_stroke_seeded(self, column: float, row: float) -> None:
        """Follow the stroke under the click into a new series."""
        if self._result is None or self._image is None or self._ink is None:
            return
        legend = self._result.content.legend_rect if self._result.content else None
        mask = stroke_mask(self._image, self._ink, self._result.frame,
                           column, row, legend_rect=legend)
        stats: dict = {}
        points = trace_stroke(mask, column, row, stats=stats)
        if points.shape[0] < 2:
            self.statusBar().showMessage(
                "No stroke found there - click directly on a curve.", 5000)
            return

        self._push_undo()
        series = Series(
            name=f"Series {len(self._result.series) + 1}",
            color=NEW_SERIES_PALETTE[len(self._result.series) % len(NEW_SERIES_PALETTE)],
            settings=ExtractionSettings(mode=ExtractionMode.CURVE),
            pixel_points=points,
            data_points=np.empty((0, 2), dtype=float),
        )
        series.recompute_data(self._result.calibration)
        self._result.series.append(series)
        self._active = len(self._result.series) - 1
        self._sync_all()

        # Report where the trace had to guess. Stated as a fact rather than as a
        # threshold alarm - overlapping curves are undetectable from a single column,
        # so a "clean" result is not a promise that the trace stayed on one curve.
        ambiguous = stats.get("ambiguous", 0)
        detail = (f" - {ambiguous} columns had another stroke alongside, so check those"
                  if ambiguous else "")
        self.statusBar().showMessage(
            f"Traced {points.shape[0]} points into {series.name}{detail}", 9000)

    def _on_point_removed(self, index: int, point_index: int) -> None:
        series = self._series_at(index)
        if series is None or not (0 <= point_index < series.pixel_points.shape[0]):
            return
        self._push_undo()
        before = series.count
        series.pixel_points = np.delete(series.pixel_points, point_index, axis=0)
        series.recompute_data(self._result.calibration)
        if index == self._active:
            self._shift_selection_for_removal([point_index], before)
        self._refresh_series_views()

    def _on_point_selected(self, index: int, point_index: int) -> None:
        self.canvas.set_selected_point(index, point_index)
        self.data_panel.select_row(point_index)

    def _on_row_selected(self, index: int, row: int) -> None:
        self.canvas.set_selected_point(index, row)

    def _delete_selected_point(self) -> None:
        if self._selection:
            self.delete_selection()
            return
        series = self._series_at(self._active)
        if series is None:
            return
        row = self.data_panel._table.currentRow()
        if row >= 0:
            self._on_point_removed(self._active, row)

    def _on_handle_dragged(self, handle_id: str, pixel: float) -> None:
        if self._result is None:
            return
        self._push_undo(coalesce="handle")
        calibration = self._result.calibration
        axis = calibration.x if handle_id.startswith("x") else calibration.y
        if handle_id.endswith("1"):
            axis.p1 = float(pixel)
        else:
            axis.p2 = float(pixel)
        self.calibrate_panel.set_handle_position(handle_id, pixel)
        self._result.recompute()
        self._refresh_series_views()

    # -- panel interaction -------------------------------------------------------

    def _on_calibration_edited(self) -> None:
        if self._result is None:
            return
        calibration = self.calibrate_panel.calibration()
        if calibration is None:
            self.statusBar().showMessage(
                "Calibration incomplete: each axis needs two different, valid values.", 6000)
            return
        self._push_undo()
        self._result.calibration = calibration
        self._result.recompute()
        self._sync_handles()
        self._refresh_series_views()

    def _on_active_series(self, index: int) -> None:
        if self._result is None or not (0 <= index < len(self._result.series)):
            return
        self._active = index
        self._clear_selection()
        self.canvas.set_active_series(index)
        self._sync_data_panel()

    def _on_series_visibility(self, index: int, visible: bool) -> None:
        series = self._series_at(index)
        if series is None or series.visible == visible:
            return
        series.visible = visible
        self.canvas.refresh_visibility(self._result.series)
        self._refresh_export_preview()

    def _on_series_renamed(self, index: int, name: str) -> None:
        series = self._series_at(index)
        if series is None:
            return
        series.name = name
        self._refresh_export_preview()

    def _on_series_settings(self, index: int, settings: ExtractionSettings) -> None:
        series = self._series_at(index)
        if series is None:
            return
        series.settings = settings
        if series.mask is None:
            # Hand-built series have no mask to re-extract from; the points are the data.
            self.statusBar().showMessage(
                "This series was created by hand, so there is nothing to re-extract.", 4000)
            return
        self._push_undo(coalesce=f"settings:{index}")
        self.statusBar().showMessage(f"Re-extracting {series.name}...", 2000)
        self._extractor.request(index, series.mask, settings)

    def _on_reextracted(self, index: int, points: np.ndarray) -> None:
        series = self._series_at(index)
        if series is None:
            return
        series.pixel_points = points
        series.recompute_data(self._result.calibration)
        if index == self._active:
            self._clear_selection()
        self._refresh_series_views()
        self.statusBar().showMessage(f"{series.name}: {series.count} points", 3000)

    def _on_sweep_finished(self, path) -> None:
        """Turn a drag along a curve into its own series.

        Each sweep lands in a separate series on purpose: a segment that came out wrong
        can be deleted whole without touching the others, and Combine merges the good
        ones at the end - recoverably, so a bad merge is not the end of the matter
        either.
        """
        if self._result is None or self._image is None or self._ink is None:
            return

        mask = self._ink.mask
        frame = self._result.frame
        interior = np.zeros_like(mask)
        r0, r1, c0, c1 = frame.interior_bounds(inset=2)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(mask.shape[0], r1), min(mask.shape[1], c1)
        if r1 > r0 and c1 > c0:
            interior[r0:r1, c0:c1] = True
        mask = mask & interior
        if self._result.content and self._result.content.legend_rect:
            x0, y0, x1, y1 = self._result.content.legend_rect
            mask = mask.copy()
            mask[max(0, y0):y1, max(0, x0):x1] = False

        points = sweep_along(mask, path)
        if points.shape[0] < 2:
            self.statusBar().showMessage(
                "Nothing found along that drag - follow the curve more closely.", 5000)
            return

        self._push_undo()
        index = len(self._result.series)
        series = Series(
            name=f"Segment {self._segment_counter()}",
            color=NEW_SERIES_PALETTE[index % len(NEW_SERIES_PALETTE)],
            settings=ExtractionSettings(mode=ExtractionMode.CURVE),
            pixel_points=points,
            data_points=np.empty((0, 2), dtype=float),
        )
        series.recompute_data(self._result.calibration)
        self._result.series.append(series)
        self._active = index
        self._clear_selection()
        self._sync_all()

        span = points[:, 0].max() - points[:, 0].min()
        self.statusBar().showMessage(
            f"{series.name}: {points.shape[0]} points across {span:.0f} px. "
            f"Sweep again for more, then Combine them.", 8000)

    def _segment_counter(self) -> int:
        existing = sum(1 for s in self._result.series if s.name.startswith("Segment "))
        return existing + 1

    def _on_add_series(self) -> None:
        if self._result is None:
            return
        self._push_undo()
        self._result.series.append(Series(
            name=f"Series {len(self._result.series) + 1}",
            color=NEW_SERIES_PALETTE[len(self._result.series) % len(NEW_SERIES_PALETTE)],
            settings=ExtractionSettings(),
            pixel_points=np.empty((0, 2), dtype=float),
            data_points=np.empty((0, 2), dtype=float),
        ))
        self._active = len(self._result.series) - 1
        self._sync_all()
        self.statusBar().showMessage("Empty series added - click the figure to add points.", 5000)

    def combine_series_dialog(self) -> None:
        """Merge several series into one, keeping them recoverable by default."""
        if self._result is None or len(self._result.series) < 2:
            QMessageBox.information(self, "Nothing to combine",
                                    "At least two series are needed.")
            return

        from .dialogs.combine import CombineDialog
        dialog = CombineDialog(self._result.series, preselected={self._active}, parent=self)
        if dialog.exec() != CombineDialog.DialogCode.Accepted:
            return

        chosen = dialog.selected_indices()
        if len(chosen) < 2:
            return

        members = [self._result.series[i] for i in chosen]
        keep = dialog.keep_sources()
        self._push_undo()

        combined = combine_series(members, name=dialog.chosen_name(),
                                  keep_sources=keep,
                                  calibration=self._result.calibration)
        # Drop the originals from the back so the earlier indices stay valid, then put
        # the result where the first of them used to be - combining should not reshuffle
        # the list out from under the user.
        for index in sorted(chosen, reverse=True):
            self._result.series.pop(index)
        self._result.series.insert(chosen[0], combined)

        self._active = chosen[0]
        self._clear_selection()
        self._sync_all()
        self.statusBar().showMessage(
            f"Combined {len(members)} series into {combined.name} ({combined.count} points)"
            + ("" if keep else " - permanently"), 8000)

    def split_series_at(self, index: int) -> None:
        """Put a combined series back into the ones it was made from."""
        series = self._series_at(index)
        if series is None:
            return
        if not series.is_combined:
            self.statusBar().showMessage(
                f"{series.name} was not combined from anything, so there is "
                f"nothing to split.", 5000)
            return

        restored = split_series(series, calibration=self._result.calibration)
        if not restored:
            return

        self._push_undo()
        self._result.series.pop(index)
        for offset, part in enumerate(restored):
            self._result.series.insert(index + offset, part)

        self._active = index
        self._clear_selection()
        self._sync_all()
        self.statusBar().showMessage(
            f"Split into {len(restored)} series: "
            f"{', '.join(s.name for s in restored)}", 8000)

    def _on_delete_series(self, index: int) -> None:
        if self._result is None or not (0 <= index < len(self._result.series)):
            return
        self._push_undo()
        self._result.series.pop(index)
        self._active = max(0, min(self._active, len(self._result.series) - 1))
        self._sync_all()

    def _on_value_edited(self, index: int, row: int, x: float, y: float) -> None:
        series = self._series_at(index)
        if series is None or self._result is None:
            return
        if not (0 <= row < series.pixel_points.shape[0]):
            return
        calibration = self._result.calibration
        if not calibration.is_valid:
            return
        # Push the edit back through the calibration so the marker on the figure moves
        # with it; a data value that no longer matches a pixel would be a lie.
        try:
            pixel = calibration.to_pixel(np.array([[x, y]]))[0]
        except Exception:                                  # noqa: BLE001
            return
        if not np.all(np.isfinite(pixel)):
            self.statusBar().showMessage(
                "That value cannot be placed on the current axes.", 4000)
            self.data_panel.refresh()
            return
        self._push_undo()
        series.pixel_points[row] = pixel
        series.recompute_data(calibration)
        self.canvas.update_series_points(index, series.pixel_points)
        self.data_panel.refresh()
        self._refresh_export_preview()

    # -- export ------------------------------------------------------------------

    def export_csv(self) -> None:
        if self._result is None or not self._result.series:
            QMessageBox.information(self, "Nothing to export", "No series have been extracted.")
            return
        if not self._result.calibrated:
            answer = QMessageBox.warning(
                self, "Axes not calibrated",
                "The axes were not detected automatically, so the exported numbers "
                "will not be in the figure's units.\n\nExport anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.tabs.setCurrentIndex(0)
                return

        default = str((self._image_path.with_suffix(".csv")) if self._image_path
                      else Path.home() / "digitized.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", default,
                                              "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            written = write_csv(path, self._result.series, self.export_panel.options())
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._set_dirty(False)
        self.statusBar().showMessage(
            f"Wrote {', '.join(Path(p).name for p in written)}", 8000)

    def import_csv(self) -> None:
        """Load a CSV of reference values back onto the figure as its own series."""
        if self._result is None:
            QMessageBox.information(self, "Open an image first",
                                    "A calibration is needed to place imported values.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV as series", str(self._image_path.parent if self._image_path
                                              else Path.home()),
            "CSV files (*.csv *.tsv *.txt);;All files (*)")
        if not path:
            return
        try:
            series = read_csv_series(path, self._result.calibration)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, "Could not import CSV", f"{Path(path).name}:\n{exc}")
            return

        self._push_undo()
        self._result.series.append(series)
        self._active = len(self._result.series) - 1
        self._sync_all()
        self.statusBar().showMessage(
            f"Imported {series.count} points from {Path(path).name}", 6000)

    def copy_to_clipboard(self) -> None:
        if self._result is None:
            return
        QGuiApplication.clipboard().setText(
            csv_string(self._result.series, self.export_panel.options()))
        self.statusBar().showMessage("Data copied to the clipboard.", 4000)

    def save_session(self) -> None:
        if self._result is None:
            return
        default = str(self._image_path.with_suffix(PROJECT_SUFFIX)) if self._image_path \
            else str(Path.home() / f"session{PROJECT_SUFFIX}")
        path, _ = QFileDialog.getSaveFileName(self, "Save session", default,
                                              f"Sessions (*{PROJECT_SUFFIX})")
        if not path:
            return
        try:
            save_project(path, self._result, self._image_path)
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.critical(self, "Could not save session", str(exc))
            return
        self._set_dirty(False)
        self.statusBar().showMessage(f"Session saved to {Path(path).name}", 6000)

    def open_session(self, path: Path | None = None) -> None:
        if not self._confirm_discard("Opening a session"):
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open session", str(Path.home()),
                                                  f"Sessions (*{PROJECT_SUFFIX})")
        if not path:
            return
        try:
            result, image_path = load_project(path)
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.critical(self, "Could not open session", str(exc))
            return

        if image_path and Path(image_path).exists():
            self._image = load_image(image_path)
            self._image_path = Path(image_path)
            self.canvas.set_image(self._image)
            self.magnifier.set_image(self._image)
        elif image_path:
            QMessageBox.warning(self, "Image not found",
                                f"The session refers to {image_path}, which is missing. "
                                f"The points and calibration were loaded without it.")
        self._result = result
        self._active = 0
        self._undo.clear()
        self._redo.clear()
        self._sync_all()
        self.statusBar().showMessage(f"Loaded {Path(path).name}", 6000)

    # -- lifecycle ---------------------------------------------------------------

    def closeEvent(self, event):
        if not self._confirm_discard("Closing"):
            event.ignore()
            return
        self._save_window_state()
        self._settings().setValue("export_options", self.export_panel.to_settings())
        self._extractor.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        super().closeEvent(event)
