"""A zoom inset that follows the cursor.

Correcting a point by hand means placing it on a marker a few pixels across. At
fit-to-window zoom that is impossible to do accurately, and zooming the whole canvas in
far enough loses the context of where you are in the figure. The inset gives both: the
canvas stays at a readable zoom while this shows the pixels under the cursor
magnified, with a crosshair on the exact position that will be recorded.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

__all__ = ["Magnifier"]


class Magnifier(QWidget):
    """Shows a nearest-neighbour magnification of the image around a point."""

    def __init__(self, parent=None, factor: int = 8, size: int = 168):
        super().__init__(parent)
        self._factor = factor
        self._pixmap: QPixmap | None = None
        self._centre: QPointF | None = None
        self.setFixedSize(size, size)
        self.setAutoFillBackground(True)

    def set_image(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self._pixmap = None
        else:
            from .canvas import numpy_to_qimage
            self._pixmap = QPixmap.fromImage(numpy_to_qimage(rgb))
        self._centre = None
        self.update()

    def set_centre(self, x: float, y: float) -> None:
        self._centre = None if (np.isnan(x) or np.isnan(y)) else QPointF(x, y)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e2126"))

        if self._pixmap is None or self._centre is None:
            painter.setPen(QPen(QColor("#6c757d")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "move over\nthe figure")
            painter.end()
            return

        span = max(4, self.width() // self._factor)
        source = QRect(int(self._centre.x()) - span // 2,
                       int(self._centre.y()) - span // 2, span, span)

        # Nearest-neighbour: the point of the inset is to see individual pixels, and
        # smoothing them would hide exactly the detail being inspected.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawPixmap(self.rect(), self._pixmap, source)

        centre = self.rect().center()
        pen = QPen(QColor("#e8590c"), 1)
        painter.setPen(pen)
        painter.drawLine(centre.x(), 0, centre.x(), self.height())
        painter.drawLine(0, centre.y(), self.width(), centre.y())
        painter.setPen(QPen(QColor("#ffd43b"), 1))
        painter.drawRect(centre.x() - self._factor // 2, centre.y() - self._factor // 2,
                         self._factor, self._factor)

        painter.setPen(QPen(QColor("#495057")))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()
