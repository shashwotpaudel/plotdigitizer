"""Step 4: writing the CSV.

The panel deliberately shows a live preview of the first few lines. Export options are
easy to get subtly wrong - the wrong layout, a precision that quietly rounds away real
digits - and seeing the actual text that will be written removes the guesswork.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...export import ExportLayout, ExportOptions

__all__ = ["ExportPanel"]


class ExportPanel(QWidget):
    """Chooses the CSV shape and triggers the save."""

    optionsChanged = Signal()
    exportRequested = Signal()
    copyRequested = Signal()
    saveProjectRequested = Signal()
    openProjectRequested = Signal()
    importCsvRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("CSV options")
        form = QFormLayout(box)

        self._layout_combo = QComboBox()
        for option in ExportLayout:
            self._layout_combo.addItem(option.label, option)
        self._layout_combo.currentIndexChanged.connect(self.optionsChanged)
        form.addRow("Layout", self._layout_combo)

        self._delimiter = QComboBox()
        for label, value in (("Comma  ,", ","), ("Semicolon  ;", ";"), ("Tab", "\t")):
            self._delimiter.addItem(label, value)
        self._delimiter.currentIndexChanged.connect(self.optionsChanged)
        form.addRow("Delimiter", self._delimiter)

        self._precision = QSpinBox()
        self._precision.setRange(0, 17)
        self._precision.setSpecialValueText("full")
        self._precision.valueChanged.connect(self.optionsChanged)
        form.addRow("Significant digits", self._precision)

        self._header = QCheckBox("Write a header row")
        self._header.setChecked(True)
        self._header.toggled.connect(self.optionsChanged)
        form.addRow(self._header)

        self._pixels = QCheckBox("Also write pixel coordinates")
        self._pixels.setToolTip("Adds x_px / y_px columns showing where each value "
                                "was read from in the image")
        self._pixels.toggled.connect(self.optionsChanged)
        form.addRow(self._pixels)

        self._visible_only = QCheckBox("Only visible series")
        self._visible_only.setChecked(True)
        self._visible_only.toggled.connect(self.optionsChanged)
        form.addRow(self._visible_only)

        layout.addWidget(box)

        layout.addWidget(QLabel("Preview"))
        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._preview.setMaximumBlockCount(400)
        self._preview.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._preview, 1)

        buttons = QHBoxLayout()
        export = QPushButton("Export CSV...")
        export.setDefault(True)
        export.clicked.connect(self.exportRequested)
        copy = QPushButton("Copy")
        copy.clicked.connect(self.copyRequested)
        buttons.addWidget(export, 2)
        buttons.addWidget(copy, 1)
        layout.addLayout(buttons)

        session = QHBoxLayout()
        save = QPushButton("Save session")
        save.clicked.connect(self.saveProjectRequested)
        load = QPushButton("Open session")
        load.clicked.connect(self.openProjectRequested)
        session.addWidget(save)
        session.addWidget(load)
        layout.addLayout(session)

        import_csv = QPushButton("Import CSV as series...")
        import_csv.setToolTip("Load reference numbers back onto the figure to compare")
        import_csv.clicked.connect(self.importCsvRequested)
        layout.addWidget(import_csv)

    def options(self) -> ExportOptions:
        precision = self._precision.value()
        return ExportOptions(
            layout=ExportLayout(self._layout_combo.currentData()),
            delimiter=self._delimiter.currentData(),
            include_header=self._header.isChecked(),
            precision=None if precision == 0 else precision,
            visible_only=self._visible_only.isChecked(),
            include_pixels=self._pixels.isChecked(),
        )

    def to_settings(self) -> dict:
        """The current choices, in a form QSettings can store."""
        return {
            "layout": self._layout_combo.currentData(),
            "delimiter": self._delimiter.currentData(),
            "precision": self._precision.value(),
            "header": self._header.isChecked(),
            "visible_only": self._visible_only.isChecked(),
            "pixels": self._pixels.isChecked(),
        }

    def from_settings(self, stored) -> None:
        """Restore previously chosen options, ignoring anything unrecognised."""
        if not isinstance(stored, dict):
            return
        for combo, key in ((self._layout_combo, "layout"), (self._delimiter, "delimiter")):
            index = combo.findData(stored.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        if "precision" in stored:
            try:
                self._precision.setValue(int(stored["precision"]))
            except (TypeError, ValueError):
                pass
        for widget, key in ((self._header, "header"),
                            (self._visible_only, "visible_only"),
                            (self._pixels, "pixels")):
            if key in stored:
                widget.setChecked(str(stored[key]).lower() in ("true", "1"))

    def set_preview(self, text: str, max_lines: int = 12) -> None:
        lines = text.splitlines()
        shown = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            shown += f"\n... {len(lines) - max_lines} more rows"
        self._preview.setPlainText(shown or "(nothing to export yet)")
