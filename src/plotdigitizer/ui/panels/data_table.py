"""Step 3: the extracted numbers, and a plot of them.

This is where the user decides the digitization is right. The table is the data that
will be written; the preview redraws it as a chart, which is the fastest way to see a
stray point or a mis-set axis - a wrong log/linear choice is obvious as a shape here
long before it is obvious as a column of numbers.

Editing a cell writes back through the calibration, so a corrected value moves the
point on the canvas too, rather than creating a number that no longer matches the
figure.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

__all__ = ["DataPanel"]


class DataPanel(QWidget):
    """Editable table of the active series plus a preview chart."""

    valueEdited = Signal(int, int, float, float)     # series, row, x, y
    pointDeleted = Signal(int, int)
    rowSelected = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["x", "y"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.currentCellChanged.connect(self._on_cell_changed)
        layout.addWidget(self._table, 3)

        controls = QHBoxLayout()
        delete = QPushButton("Delete selected point")
        delete.clicked.connect(self._delete_current)
        controls.addWidget(delete)
        self._preview_toggle = QCheckBox("Preview")
        self._preview_toggle.setChecked(True)
        self._preview_toggle.toggled.connect(self._on_preview_toggled)
        controls.addWidget(self._preview_toggle)
        layout.addLayout(controls)

        self._canvas = None
        self._axes = None
        self._preview_placeholder = QLabel("Preview unavailable")
        self._preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_placeholder.setStyleSheet("color: #868e96;")
        self._preview_placeholder.hide()
        layout.addWidget(self._preview_placeholder, 2)
        self._build_preview(layout)

        self._series_index = 0
        self._loading = False
        self._series = None
        self._scales = ("linear", "linear")

    def _build_preview(self, layout) -> None:
        """Attach a matplotlib canvas, degrading to a note if the backend is missing."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as exc:                     # pragma: no cover - env dependent
            log.info("preview chart unavailable: %s", exc)
            self._preview_placeholder.show()
            return

        figure = Figure(figsize=(3.0, 2.2), layout="constrained")
        self._axes = figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(figure)
        # Both minimums matter: Agg raises a raster overflow trying to lay out tick
        # labels into a canvas that a layout pass has left with no width.
        self._canvas.setMinimumHeight(150)
        self._canvas.setMinimumWidth(160)
        layout.addWidget(self._canvas, 2)

    # -- state -------------------------------------------------------------------

    def set_series(self, series, index: int, scales=("linear", "linear")) -> None:
        self._series = series
        self._series_index = index
        self._scales = scales
        self._reload_table()
        self._redraw_preview()

    def _reload_table(self) -> None:
        self._loading = True
        try:
            points = (self._series.data_points if self._series is not None
                      else np.empty((0, 2)))
            self._table.setRowCount(points.shape[0])
            for row in range(points.shape[0]):
                for column in range(2):
                    item = QTableWidgetItem(f"{points[row, column]:.6g}")
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(row, column, item)
        finally:
            self._loading = False

    def _redraw_preview(self) -> None:
        if self._axes is None or self._canvas is None or not self._preview_toggle.isChecked():
            return
        # Before the first layout pass the canvas can still be zero-sized, and Agg
        # raises a raster overflow trying to lay out tick labels into no space at all.
        if self._canvas.width() < 40 or self._canvas.height() < 40:
            return
        self._axes.clear()
        if self._series is not None and self._series.data_points.shape[0]:
            points = self._series.data_points
            colour = self._series.hex_color
            order = np.argsort(points[:, 0])
            style = "o" if self._series.settings.mode.value == "scatter" else "-"
            self._axes.plot(points[order, 0], points[order, 1], style,
                            color=colour, markersize=3, linewidth=1.2)
            try:
                self._axes.set_xscale("log" if self._scales[0] == "log10" else "linear")
                self._axes.set_yscale("log" if self._scales[1] == "log10" else "linear")
            except ValueError:
                # Non-positive values on a log preview: fall back rather than raise.
                self._axes.set_xscale("linear")
                self._axes.set_yscale("linear")
        self._axes.grid(True, alpha=0.3)
        self._axes.tick_params(labelsize=7)
        self._canvas.draw_idle()

    def refresh(self) -> None:
        self._reload_table()
        self._redraw_preview()

    def select_row(self, row: int) -> None:
        if 0 <= row < self._table.rowCount():
            self._loading = True
            self._table.selectRow(row)
            self._loading = False

    # -- interaction -------------------------------------------------------------

    def _on_preview_toggled(self, enabled: bool) -> None:
        if self._canvas is not None:
            self._canvas.setVisible(enabled)
        if enabled:
            self._redraw_preview()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or self._series is None:
            return
        row = item.row()
        try:
            x = float(self._table.item(row, 0).text())
            y = float(self._table.item(row, 1).text())
        except (AttributeError, ValueError):
            self._reload_table()          # put back what was there
            return
        self.valueEdited.emit(self._series_index, row, x, y)

    def _on_cell_changed(self, row: int, column: int, *_) -> None:
        if not self._loading and row >= 0:
            self.rowSelected.emit(self._series_index, row)

    def _delete_current(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self.pointDeleted.emit(self._series_index, row)
