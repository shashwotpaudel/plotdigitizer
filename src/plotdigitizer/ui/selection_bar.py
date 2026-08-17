"""The bar that appears under the canvas when points are selected.

It exists only while there is a selection, which is the whole safety argument: the
destructive buttons are not sitting there to be hit by accident, and when they do appear
they are next to a count saying exactly how much they will affect.

"Move to" comes first and is the default-styled button because it is the non-destructive
answer to the usual problem - a trace that picked up a neighbouring curve's points. They
belong to a different series, not to the bin.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QWidget,
)

__all__ = ["SelectionBar"]


class SelectionBar(QWidget):
    """Actions for the current point selection."""

    moveRequested = Signal(int)          # target series index, -1 for a new series
    deleteRequested = Signal()
    keepOnlyRequested = Signal()
    invertRequested = Signal()
    clearRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        self._label = QLabel("")
        self._label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._label)
        layout.addStretch(1)

        self._move_menu = QMenu(self)
        self._move = QToolButton()
        self._move.setText("Move to")
        self._move.setMenu(self._move_menu)
        self._move.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._move.setToolTip("Hand these points to another series instead of deleting them")
        self._move.setStyleSheet(
            "QToolButton { background: #364fc7; border: 1px solid #4c6ef5;"
            " border-radius: 4px; padding: 5px 12px; }")
        layout.addWidget(self._move)

        for text, signal, tip in (
            ("Delete", self.deleteRequested, "Remove the selected points"),
            ("Keep only", self.keepOnlyRequested, "Remove everything except the selection"),
            ("Invert", self.invertRequested, "Select the points that are not selected"),
            ("Clear", self.clearRequested, "Deselect everything (Esc)"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(signal)
            layout.addWidget(button)

        self.setStyleSheet("SelectionBar { background: #2b3038; border-top: 1px solid #495057; }")
        self.setVisible(False)

    def update_state(self, selected: int, total: int, series_list, active: int) -> None:
        """Show the bar when there is a selection, and rebuild the move targets."""
        if selected <= 0:
            self.setVisible(False)
            return

        self._label.setText(f"{selected} of {total} points selected")
        self._move_menu.clear()
        for index, series in enumerate(series_list):
            if index == active:
                continue
            action = self._move_menu.addAction(f"{series.name}  ({series.count} pts)")
            action.triggered.connect(lambda _=False, i=index: self.moveRequested.emit(i))
        if not self._move_menu.isEmpty():
            self._move_menu.addSeparator()
        new_action = self._move_menu.addAction("New series...")
        new_action.triggered.connect(lambda: self.moveRequested.emit(-1))

        self.setVisible(True)
