"""Step 1: the axis calibration panel.

Four values and two scale choices, matching the x1/x2/y1/y2 model used by
plotdigitizer.com. The fields are filled in from the tick labels the detector read, so
in the normal case this panel is something to glance at rather than fill in - but every
field stays editable, because a figure the OCR cannot read must still be digitizable.

The confidence badge exists so a weak fit is visible before the data is exported rather
than after someone has drawn a conclusion from it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...calibration import AxisCalibration, AxisScale, Calibration

__all__ = ["CalibratePanel"]

_SCALES = [AxisScale.LINEAR, AxisScale.LOG10, AxisScale.LOGE,
           AxisScale.RECIPROCAL, AxisScale.DATE]


def _scale_of(combo: QComboBox) -> AxisScale:
    """Read the selected scale back as an enum.

    Qt stores combo box user data as a QVariant, and these enums subclass ``str``, so
    what comes back out is a bare string. Converting here keeps that detail from
    leaking into the calibration code.
    """
    return AxisScale(combo.currentData())


class _ValueEdit(QLineEdit):
    """A numeric field that shows when what it holds cannot be used."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("value")

    def value(self) -> float | None:
        text = self.text().strip().replace("−", "-")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def set_value(self, value: float) -> None:
        self.blockSignals(True)
        self.setText(f"{value:.6g}")
        self.blockSignals(False)

    def mark(self, ok: bool) -> None:
        self.setStyleSheet("" if ok else "border: 1px solid #e03131;")


class CalibratePanel(QWidget):
    """Edits the two axis calibrations."""

    calibrationEdited = Signal()
    recalibrateRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._badge = QLabel("No image loaded")
        self._badge.setWordWrap(True)
        self._badge.setFrameShape(QFrame.Shape.StyledPanel)
        self._badge.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._badge)

        self._fields: dict[str, _ValueEdit] = {}
        self._scales: dict[str, QComboBox] = {}

        for axis, title, keys in (("x", "X axis", ("x1", "x2")), ("y", "Y axis", ("y1", "y2"))):
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

            scale = QComboBox()
            for option in _SCALES:
                scale.addItem(option.label, option)
            scale.currentIndexChanged.connect(self._emit_change)
            self._scales[axis] = scale
            form.addRow("Type", scale)

            for key in keys:
                field = _ValueEdit()
                field.editingFinished.connect(self._emit_change)
                self._fields[key] = field
                form.addRow(f"{key} value", field)

            layout.addWidget(box)

        positions = QGroupBox("Handle positions (px)")
        grid = QGridLayout(positions)
        self._position_labels: dict[str, QLabel] = {}
        for column, key in enumerate(("x1", "x2", "y1", "y2")):
            grid.addWidget(QLabel(key), 0, column, alignment=Qt.AlignmentFlag.AlignHCenter)
            label = QLabel("-")
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            self._position_labels[key] = label
            grid.addWidget(label, 1, column)
        layout.addWidget(positions)

        hint = QLabel("Drag the orange handles on the figure to move a reference line.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #868e96;")
        layout.addWidget(hint)

        self._recalibrate = QPushButton("Re-detect axes")
        self._recalibrate.clicked.connect(self.recalibrateRequested)
        layout.addWidget(self._recalibrate)
        layout.addStretch(1)

        self._positions = {"x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0}
        self._loading = False

    # -- state -------------------------------------------------------------------

    def set_calibration(self, calibration: Calibration, confidence: dict | None = None) -> None:
        """Fill the panel from a calibration without echoing edit signals back out."""
        self._loading = True
        try:
            for axis, cal in (("x", calibration.x), ("y", calibration.y)):
                combo = self._scales[axis]
                index = combo.findData(cal.scale.value)
                if index >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(index)
                    combo.blockSignals(False)
            self._fields["x1"].set_value(calibration.x.v1)
            self._fields["x2"].set_value(calibration.x.v2)
            self._fields["y1"].set_value(calibration.y.v1)
            self._fields["y2"].set_value(calibration.y.v2)
            self._positions = {
                "x1": calibration.x.p1, "x2": calibration.x.p2,
                "y1": calibration.y.p1, "y2": calibration.y.p2,
            }
            self._refresh_positions()
        finally:
            self._loading = False
        if confidence is not None:
            self.set_confidence(confidence)
        self._validate()

    def set_handle_position(self, handle_id: str, pixel: float) -> None:
        self._positions[handle_id] = float(pixel)
        self._refresh_positions()

    def _refresh_positions(self) -> None:
        for key, label in self._position_labels.items():
            label.setText(f"{self._positions[key]:.1f}")

    def calibration(self) -> Calibration | None:
        """Build a calibration from the fields, or None if a value is unusable."""
        values = {key: field.value() for key, field in self._fields.items()}
        if any(v is None for v in values.values()):
            return None
        x = AxisCalibration(
            p1=self._positions["x1"], v1=values["x1"],
            p2=self._positions["x2"], v2=values["x2"],
            scale=_scale_of(self._scales["x"]),
        )
        y = AxisCalibration(
            p1=self._positions["y1"], v1=values["y1"],
            p2=self._positions["y2"], v2=values["y2"],
            scale=_scale_of(self._scales["y"]),
        )
        if not (x.is_valid and y.is_valid):
            return None
        return Calibration(x=x, y=y)

    def set_confidence(self, confidence: dict) -> None:
        x = confidence.get("x_axis", 0.0)
        y = confidence.get("y_axis", 0.0)
        if x <= 0 or y <= 0:
            self._badge.setText("Axes not detected - type the values for the four handles.")
            self._badge.setStyleSheet("background: #5c2b29; color: #ffc9c9;")
            return
        worst = min(x, y)
        if worst > 0.75:
            self._badge.setText(f"Axes detected automatically (confidence {worst:.0%}).")
            self._badge.setStyleSheet("background: #2b5c3b; color: #b2f2bb;")
        else:
            self._badge.setText(
                f"Axes detected, but the fit is uncertain (confidence {worst:.0%}). "
                f"Check the values against the figure.")
            self._badge.setStyleSheet("background: #5c4a29; color: #ffe8a1;")

    # -- validation --------------------------------------------------------------

    def _validate(self) -> bool:
        ok = True
        for axis, keys in (("x", ("x1", "x2")), ("y", ("y1", "y2"))):
            scale = _scale_of(self._scales[axis])
            for key in keys:
                value = self._fields[key].value()
                valid = value is not None and not (scale.requires_positive and value <= 0)
                self._fields[key].mark(valid)
                ok = ok and valid
        return ok

    def _emit_change(self) -> None:
        if self._loading:
            return
        self._validate()
        self.calibrationEdited.emit()
