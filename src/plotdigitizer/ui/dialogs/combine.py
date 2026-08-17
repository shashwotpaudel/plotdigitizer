"""The dialog for combining several series into one.

Combining is the one operation here that consumes other things, so the dialog is built
so that the consuming version cannot be chosen without meaning it: the recoverable
option is preselected, the permanent one says plainly that it cannot be undone later,
and a running summary states exactly what is about to happen before the button is
available at all.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QVBoxLayout,
)

__all__ = ["CombineDialog"]


def _swatch(colour, size: int = 12) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*colour))
    return QIcon(pixmap)


class CombineDialog(QDialog):
    """Pick the series to combine and whether the originals are kept."""

    def __init__(self, series_list, preselected=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine series")
        self.setMinimumWidth(420)
        self._series = list(series_list)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Combine these series into one:"))

        self._list = QListWidget()
        for index, series in enumerate(self._series):
            item = QListWidgetItem(_swatch(series.color),
                                   f"{series.name}  ({series.count} pts)")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if index in preselected
                               else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        self._list.itemChanged.connect(self._refresh)
        layout.addWidget(self._list)

        layout.addWidget(QLabel("Name for the combined series"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("leave blank to join the names")
        layout.addWidget(self._name)

        self._keep = QRadioButton("Keep the originals - this can be split apart again")
        self._keep.setChecked(True)
        self._permanent = QRadioButton("Combine permanently - the originals are not kept")
        self._permanent.setToolTip(
            "The combine itself can still be undone, but once you move on there is no "
            "record of which points came from which series.")
        layout.addWidget(self._keep)
        layout.addWidget(self._permanent)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #adb5bd;")
        layout.addWidget(self._summary)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                         QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._refresh()

    # -- state -------------------------------------------------------------------

    def selected_indices(self) -> list[int]:
        return [i for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.CheckState.Checked]

    def chosen_name(self) -> str | None:
        return self._name.text().strip() or None

    def keep_sources(self) -> bool:
        return self._keep.isChecked()

    def _refresh(self) -> None:
        chosen = self.selected_indices()
        points = sum(self._series[i].count for i in chosen)
        # Combining one series with nothing is a no-op, so the button stays unavailable
        # until the action would actually mean something.
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(len(chosen) >= 2)

        if len(chosen) < 2:
            self._summary.setText("Choose at least two series.")
            return
        names = ", ".join(self._series[i].name for i in chosen)
        self._summary.setText(
            f"{len(chosen)} series → 1 series, {points:,} points.\n{names}")
