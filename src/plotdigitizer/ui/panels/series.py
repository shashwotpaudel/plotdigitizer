"""Step 2: the series panel - which curves were found, and how to trace them.

The list shows what the colour separation produced; the controls below re-extract the
selected series live. The mode selector is the important one: the detector's guess
between "markers" and "line" is right on ordinary figures but the distinction is
genuinely ambiguous on some, and flipping it is a single click rather than a reason to
give up on the tool.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...compose import describe_sources as _describe
from ...detect.extract import ExtractionMode, ExtractionSettings

__all__ = ["SeriesPanel"]


def _swatch(colour: tuple[int, int, int], size: int = 12) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*colour))
    return QIcon(pixmap)


def _mode_of(combo: QComboBox) -> ExtractionMode:
    """Read the selected mode back as an enum.

    Qt hands combo box user data back as a QVariant; these enums subclass ``str``, so
    without this the extractor would receive a bare string.
    """
    return ExtractionMode(combo.currentData())


class SeriesPanel(QWidget):
    """Series list plus the per-series extraction controls."""

    activeChanged = Signal(int)
    visibilityChanged = Signal(int, bool)
    nameChanged = Signal(int, str)
    settingsChanged = Signal(int, object)      # index, ExtractionSettings
    addSeriesRequested = Signal()
    deleteSeriesRequested = Signal(int)
    combineRequested = Signal()
    splitRequested = Signal(int)
    selectSpikesRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._list = QListWidget()
        self._list.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, 1)

        buttons = QHBoxLayout()
        add = QPushButton("Add series")
        add.clicked.connect(self.addSeriesRequested)
        remove = QPushButton("Delete")
        remove.clicked.connect(lambda: self.deleteSeriesRequested.emit(self._list.currentRow()))
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)

        layers = QHBoxLayout()
        combine = QPushButton("Combine...")
        combine.setToolTip("Merge several series into one, keeping them recoverable")
        combine.clicked.connect(self.combineRequested)
        self._split = QPushButton("Split")
        self._split.setEnabled(False)
        self._split.clicked.connect(lambda: self.splitRequested.emit(self._list.currentRow()))
        layers.addWidget(combine)
        layers.addWidget(self._split)
        layout.addLayout(layers)

        cleanup = QGroupBox("Find stray points")
        cleanup_form = QFormLayout(cleanup)
        spike_row = QWidget()
        spike_layout = QHBoxLayout(spike_row)
        spike_layout.setContentsMargins(0, 0, 0, 0)
        self._spike_sensitivity = QSlider(Qt.Orientation.Horizontal)
        self._spike_sensitivity.setRange(15, 80)     # sensitivity x10
        self._spike_sensitivity.setValue(30)
        self._spike_sensitivity.valueChanged.connect(self._emit_spikes)
        self._spike_label = QLabel("3.0")
        spike_layout.addWidget(self._spike_sensitivity, 1)
        spike_layout.addWidget(self._spike_label)
        cleanup_form.addRow("Sensitivity", spike_row)
        select_spikes = QPushButton("Select spikes")
        select_spikes.setToolTip("Highlight points that look like they belong to "
                                 "another curve. Nothing is changed - review first.")
        select_spikes.clicked.connect(self._emit_spikes)
        cleanup_form.addRow(select_spikes)
        layout.addWidget(cleanup)

        self._controls = QGroupBox("Extraction")
        form = QFormLayout(self._controls)

        self._mode = QComboBox()
        for mode in (ExtractionMode.SCATTER, ExtractionMode.CURVE):
            self._mode.addItem(mode.label, mode)
        self._mode.currentIndexChanged.connect(self._emit_settings)
        form.addRow("Mode", self._mode)

        self._x_step = QSpinBox()
        self._x_step.setRange(1, 40)
        self._x_step.setSuffix(" px")
        self._x_step.valueChanged.connect(self._emit_settings)
        form.addRow("X step", self._x_step)

        self._smoothing = QSpinBox()
        self._smoothing.setRange(1, 31)
        self._smoothing.setSingleStep(2)
        self._smoothing.valueChanged.connect(self._emit_settings)
        form.addRow("Smoothing", self._smoothing)

        self._max_gap = QSpinBox()
        self._max_gap.setRange(0, 400)
        self._max_gap.setSuffix(" px")
        self._max_gap.valueChanged.connect(self._emit_settings)
        form.addRow("Bridge gaps up to", self._max_gap)

        blob_row = QWidget()
        blob_layout = QHBoxLayout(blob_row)
        blob_layout.setContentsMargins(0, 0, 0, 0)
        self._min_blob = QSlider(Qt.Orientation.Horizontal)
        self._min_blob.setRange(0, 90)
        self._min_blob.valueChanged.connect(self._emit_settings)
        self._min_blob_label = QLabel("25%")
        blob_layout.addWidget(self._min_blob, 1)
        blob_layout.addWidget(self._min_blob_label)
        form.addRow("Min blob size", blob_row)

        self._max_points = QSpinBox()
        self._max_points.setRange(0, 10000)
        self._max_points.setSpecialValueText("all")
        self._max_points.valueChanged.connect(self._emit_settings)
        form.addRow("Limit points", self._max_points)

        layout.addWidget(self._controls)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #868e96;")
        layout.addWidget(self._summary)

        self._loading = False
        self._series: list = []
        self._controls.setEnabled(False)

    # -- state -------------------------------------------------------------------

    def set_series(self, series_list, active: int = 0) -> None:
        self._loading = True
        try:
            self._series = list(series_list)
            self._list.clear()
            for series in self._series:
                suffix = (f"  [{len(series.sources)} merged]" if series.is_combined else "")
                item = QListWidgetItem(_swatch(series.color),
                                       f"{series.name}  ({series.count} pts){suffix}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable
                              | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if series.visible
                                   else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, series.name)
                self._list.addItem(item)
            if self._series:
                self._list.setCurrentRow(min(active, len(self._series) - 1))
        finally:
            self._loading = False
        self._controls.setEnabled(bool(self._series))
        self._load_settings(self._list.currentRow())

    def _emit_spikes(self) -> None:
        value = self._spike_sensitivity.value() / 10.0
        self._spike_label.setText(f"{value:.1f}")
        if not self._loading:
            self.selectSpikesRequested.emit(value)

    def refresh_counts(self, series_list) -> None:
        """Update the point counts in place without disturbing selection or editing."""
        self._loading = True
        try:
            self._series = list(series_list)
            for row, series in enumerate(self._series):
                item = self._list.item(row)
                if item is not None:
                    suffix = (f"  [{len(series.sources)} merged]"
                              if series.is_combined else "")
                    item.setText(f"{series.name}  ({series.count} pts){suffix}")
                    item.setIcon(_swatch(series.color))
        finally:
            self._loading = False
        self._update_summary(self._list.currentRow())

    def current_index(self) -> int:
        return self._list.currentRow()

    def _load_settings(self, index: int) -> None:
        if not (0 <= index < len(self._series)):
            self._controls.setEnabled(False)
            return
        self._controls.setEnabled(True)
        current = self._series[index]
        self._split.setEnabled(current.is_combined)
        self._split.setToolTip(
            f"Restore {_describe(current)}" if current.is_combined
            else "Only a combined series can be split")
        settings = current.settings
        self._loading = True
        try:
            position = self._mode.findData(settings.mode.value)
            if position >= 0:
                self._mode.setCurrentIndex(position)
            self._x_step.setValue(settings.x_step)
            self._smoothing.setValue(settings.smoothing)
            self._max_gap.setValue(settings.max_gap)
            self._min_blob.setValue(int(round(settings.min_blob_fraction * 100)))
            self._min_blob_label.setText(f"{int(round(settings.min_blob_fraction * 100))}%")
            self._max_points.setValue(settings.max_points)
        finally:
            self._loading = False
        self._update_mode_availability()
        self._update_summary(index)

    def _update_mode_availability(self) -> None:
        curve = _mode_of(self._mode) is ExtractionMode.CURVE
        self._x_step.setEnabled(curve)
        self._smoothing.setEnabled(curve)
        self._max_gap.setEnabled(curve)
        self._min_blob.setEnabled(not curve)

    def _update_summary(self, index: int) -> None:
        if not (0 <= index < len(self._series)):
            self._summary.setText("")
            return
        series = self._series[index]
        self._summary.setText(
            f"{series.count} points - {series.hex_color}. "
            f"Left-click the figure to add a point, right-click one to delete it."
        )

    # -- signals -----------------------------------------------------------------

    def _on_row_changed(self, row: int) -> None:
        self._load_settings(row)
        if not self._loading:
            self.activeChanged.emit(row)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        row = self._list.row(item)
        visible = item.checkState() == Qt.CheckState.Checked
        self.visibilityChanged.emit(row, visible)

        # The list text carries the point count, so recover just the edited name.
        text = item.text()
        name = text.split("  (")[0].strip()
        if name and name != item.data(Qt.ItemDataRole.UserRole):
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.nameChanged.emit(row, name)

    def _emit_settings(self) -> None:
        self._min_blob_label.setText(f"{self._min_blob.value()}%")
        self._update_mode_availability()
        if self._loading:
            return
        index = self._list.currentRow()
        if not (0 <= index < len(self._series)):
            return
        settings = ExtractionSettings(
            mode=_mode_of(self._mode),
            x_step=self._x_step.value(),
            min_blob_fraction=self._min_blob.value() / 100.0,
            max_gap=self._max_gap.value(),
            smoothing=self._smoothing.value(),
            max_points=self._max_points.value(),
        )
        self.settingsChanged.emit(index, settings)
