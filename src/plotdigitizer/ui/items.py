"""Canvas overlay items: the plot frame, the calibration handles, the data points.

Series points are painted by a single item per series rather than one item per point.
A traced curve can hold several hundred points and a figure several series, and a
thousand individual QGraphicsItems make panning visibly stutter; one item that paints
an array does not. Hit-testing is done against the same array, so picking stays exact.

Everything is positioned in image pixel coordinates - the scene *is* the image - which
keeps the arithmetic between detection results and the display trivial.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

__all__ = ["FrameItem", "CalibrationHandleItem", "SeriesPointsItem", "LassoItem",
           "Z_FRAME", "Z_POINTS", "Z_HANDLES", "Z_LASSO"]

Z_FRAME = 10
Z_POINTS = 20
Z_HANDLES = 30
Z_LASSO = 40

_HANDLE_COLOR = QColor("#e8590c")
_HANDLE_ACTIVE = QColor("#ffa94d")
_FRAME_COLOR = QColor(64, 160, 255)


class FrameItem(QGraphicsItem):
    """A dashed rectangle showing the detected plot area."""

    def __init__(self):
        super().__init__()
        self._rect = QRectF()
        self.setZValue(Z_FRAME)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_frame(self, left: float, top: float, right: float, bottom: float) -> None:
        self.prepareGeometryChange()
        self._rect = QRectF(QPointF(left, top), QPointF(right, bottom))
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-2, -2, 2, 2)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._rect.isNull():
            return
        # Cosmetic pens keep a constant on-screen width whatever the zoom.
        pen = QPen(_FRAME_COLOR, 1.0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._rect)


class CalibrationHandleItem(QGraphicsObject):
    """One draggable axis reference marker, constrained to its own axis.

    These are the four handles a user would otherwise place by hand: two on the x axis
    and two on the y. Dragging one is a direct edit of the calibration, so the whole
    digitization follows it live.
    """

    moved = Signal(str, float)          # handle id, new pixel coordinate
    released = Signal(str)

    #: Half-width of the draggable band, in screen pixels.
    GRAB_PX = 7.0
    #: How far the line is drawn past the frame, and how much of that stub can be
    #: grabbed. These stubs sit outside the data area, so grabbing them can never
    #: compete with a data marker.
    OVERHANG_PX = 14.0
    #: Length of the draggable grip at the labelled end, in screen pixels.
    GRIP_PX = 16.0

    def __init__(self, handle_id: str, label: str, vertical: bool):
        super().__init__()
        self.handle_id = handle_id
        self.label = label
        self.vertical = vertical         # True: a vertical line marking an x position
        self._position = 0.0
        self._span = (0.0, 100.0)
        self._hover = False
        self._dragging = False
        # Set by the view so screen-sized features can be expressed in scene units.
        self._view_scale = 1.0
        self.setZValue(Z_HANDLES)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeHorCursor if vertical else Qt.CursorShape.SizeVerCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, False)

    # -- geometry ---------------------------------------------------------------

    def set_position(self, value: float) -> None:
        if value == self._position:
            return
        self.prepareGeometryChange()
        self._position = float(value)
        self.update()

    def position(self) -> float:
        return self._position

    def set_span(self, start: float, end: float) -> None:
        self.prepareGeometryChange()
        self._span = (float(start), float(end))
        self.update()

    def set_view_scale(self, scale: float) -> None:
        """Tell the item how many screen pixels one scene unit currently occupies."""
        scale = max(1e-6, float(scale))
        if abs(scale - self._view_scale) < 1e-9:
            return
        self.prepareGeometryChange()
        self._view_scale = scale
        self.update()

    def _scene(self, screen_px: float) -> float:
        """Convert a screen-pixel length into scene units at the current zoom."""
        return screen_px / self._view_scale

    def boundingRect(self) -> QRectF:
        start, end = self._span
        pad = self._scene(26.0)
        if self.vertical:
            return QRectF(self._position - pad, start - pad, 2 * pad, (end - start) + 2 * pad)
        return QRectF(start - pad, self._position - pad, (end - start) + 2 * pad, 2 * pad)

    def shape(self):
        """The draggable region: the grip and the stubs, never the span itself.

        The line is painted across the whole plot because it is a useful alignment
        guide, but making that whole length draggable turns it into an invisible wall:
        any data marker lying under it becomes unclickable, and on a real figure a
        handle very often sits right through the densest part of the data.

        So only the ends respond - the labelled grip, and the short stubs where the
        line overhangs the frame. Both are outside the data area, so grabbing a handle
        and picking a marker can no longer be the same gesture.
        """
        path = QPainterPath()
        start, end = self._span
        grab = self._scene(self.GRAB_PX)
        over = self._scene(self.OVERHANG_PX)
        grip = self._scene(self.GRIP_PX)

        if self.vertical:
            # Grip at the top end, plus the stubs above and below the frame.
            path.addRect(QRectF(self._position - grab, start - over, 2 * grab, over + grip))
            path.addRect(QRectF(self._position - grab, end, 2 * grab, over))
        else:
            path.addRect(QRectF(start - over, self._position - grab, over + grip, 2 * grab))
            path.addRect(QRectF(end, self._position - grab, over, 2 * grab))
        return path

    # -- painting ---------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:
        colour = _HANDLE_ACTIVE if (self._hover or self._dragging) else _HANDLE_COLOR
        pen = QPen(colour, 2.0 if (self._hover or self._dragging) else 1.4)
        pen.setCosmetic(True)
        painter.setPen(pen)

        # Keep the item's idea of the zoom in step with what is actually being painted,
        # so the grab regions match the drawn grips even if the view forgets to push it.
        self._view_scale = max(1e-6, painter.transform().m11())
        scale = self._view_scale
        over = self._scene(self.OVERHANG_PX)

        start, end = self._span
        if self.vertical:
            # Drawn past the frame at both ends: those stubs are the grab targets.
            painter.drawLine(QPointF(self._position, start - over),
                             QPointF(self._position, end + over))
            anchor = QPointF(self._position, start)
        else:
            painter.drawLine(QPointF(start - over, self._position),
                             QPointF(end + over, self._position))
            anchor = QPointF(start, self._position)

        # The grip and caption are sized in inverse proportion to the zoom so they stay
        # legible without growing into the figure when the user zooms in to place a
        # handle precisely.
        size = 5.0 / scale

        painter.setBrush(QBrush(colour))
        if self.vertical:
            grip = QPolygonF([
                QPointF(anchor.x() - size, anchor.y() - size),
                QPointF(anchor.x() + size, anchor.y() - size),
                QPointF(anchor.x(), anchor.y() + size),
            ])
        else:
            grip = QPolygonF([
                QPointF(anchor.x() - size, anchor.y() - size),
                QPointF(anchor.x() - size, anchor.y() + size),
                QPointF(anchor.x() + size, anchor.y()),
            ])
        painter.drawPolygon(grip)

        font = QFont()
        font.setPointSizeF(max(1.0, 8.0 / scale))
        painter.setFont(font)
        painter.setPen(QPen(colour))
        offset = 8.0 / scale
        painter.drawText(QPointF(anchor.x() + offset, anchor.y() + offset), self.label)

    # -- interaction ------------------------------------------------------------

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            event.accept()
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        point = event.scenePos()
        value = point.x() if self.vertical else point.y()
        self.set_position(value)
        self.moved.emit(self.handle_id, float(value))
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.update()
            self.released.emit(self.handle_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class LassoItem(QGraphicsItem):
    """The shape being dragged out to select points.

    One item serves both the rectangle and the freeform lasso so the two look and
    behave identically - only the path differs. It is drawn in the scene rather than as
    a ``QRubberBand`` because a rubber band can only ever be a rectangle.
    """

    #: What the gesture will do with what it encloses.
    MODE_COLOURS = {
        "replace": QColor("#4dabf7"),
        "add": QColor("#69db7c"),
        "subtract": QColor("#ff8787"),
        "sweep": QColor("#ffd43b"),
    }

    def __init__(self):
        super().__init__()
        self._path = QPainterPath()
        self._mode = "replace"
        self.setZValue(Z_LASSO)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setVisible(False)

    def set_path(self, path: QPainterPath, mode: str = "replace") -> None:
        self.prepareGeometryChange()
        self._path = QPainterPath(path)
        self._mode = mode
        self.setVisible(not path.isEmpty())
        self.update()

    def clear(self) -> None:
        self.prepareGeometryChange()
        self._path = QPainterPath()
        self.setVisible(False)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect().adjusted(-4, -4, 4, 4)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._path.isEmpty():
            return
        # Colour says what will happen - adding, subtracting or replacing - so the
        # outcome is visible before the button is released rather than after.
        colour = self.MODE_COLOURS.get(self._mode, self.MODE_COLOURS["replace"])
        if self._mode == "sweep":
            # An open trail, not an enclosed region: filling it would suggest the area
            # under the drag is being selected rather than the curve along it.
            pen = QPen(colour, 2.0, Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._path)
            return

        pen = QPen(colour, 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        fill = QColor(colour)
        fill.setAlpha(40)
        painter.setBrush(QBrush(fill))
        painter.drawPath(self._path)


class SeriesPointsItem(QGraphicsItem):
    """Paints every extracted point of one series."""

    def __init__(self, colour: QColor):
        super().__init__()
        self._points = np.empty((0, 2), dtype=float)
        self._colour = QColor(colour)
        self._radius = 3.0
        self._active = False
        self._selected = -1
        self._selection: set[int] = set()
        self.setZValue(Z_POINTS)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_points(self, points: np.ndarray) -> None:
        self.prepareGeometryChange()
        self._points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.update()

    def set_colour(self, colour) -> None:
        self._colour = QColor(colour)
        self.update()

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.update()

    def set_selected(self, index: int) -> None:
        if index != self._selected:
            self._selected = index
            self.update()

    def set_selection(self, indices) -> None:
        """The set of points currently selected for a bulk action."""
        new = set(int(i) for i in indices) if indices is not None else set()
        if new != self._selection:
            self._selection = new
            self.update()

    def selection(self) -> set[int]:
        return set(self._selection)

    def points(self) -> np.ndarray:
        return self._points

    def indices_within(self, path: QPainterPath) -> list[int]:
        """Indices of the points enclosed by a scene-coordinate path."""
        if self._points.shape[0] == 0 or path.isEmpty():
            return []
        # Test the path's bounding box first: containment is comparatively expensive
        # and a lasso usually encloses a small part of a long series.
        box = path.boundingRect()
        inside = []
        for index in range(self._points.shape[0]):
            x, y = self._points[index]
            if not box.contains(x, y):
                continue
            if path.contains(QPointF(x, y)):
                inside.append(index)
        return inside

    def nearest(self, x: float, y: float, max_distance: float) -> int:
        """Index of the closest point within ``max_distance`` scene units, else -1."""
        if self._points.shape[0] == 0:
            return -1
        deltas = self._points - np.array([x, y], dtype=float)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        index = int(np.argmin(distances))
        return index if distances[index] <= max_distance else -1

    def boundingRect(self) -> QRectF:
        if self._points.shape[0] == 0:
            return QRectF()
        pad = self._radius + 4.0
        x0, y0 = self._points.min(axis=0)
        x1, y1 = self._points.max(axis=0)
        return QRectF(QPointF(x0 - pad, y0 - pad), QPointF(x1 + pad, y1 + pad))

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._points.shape[0] == 0:
            return
        scale = max(1e-6, painter.transform().m11())
        radius = self._radius / scale

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outline = QPen(QColor(255, 255, 255, 220) if self._active else QColor(0, 0, 0, 90))
        outline.setCosmetic(True)
        outline.setWidthF(1.4 if self._active else 1.0)
        painter.setPen(outline)
        painter.setBrush(QBrush(self._colour))

        for index in range(self._points.shape[0]):
            x, y = self._points[index]
            painter.drawEllipse(QPointF(x, y), radius, radius)

        # Selected points are drawn over the top in a colour of their own, so what a
        # bulk action is about to affect is unmistakable against any figure.
        if self._selection:
            halo = QPen(QColor("#ffd43b"), 2.2)
            halo.setCosmetic(True)
            painter.setPen(halo)
            painter.setBrush(QBrush(QColor("#ffd43b")))
            for index in sorted(self._selection):
                if 0 <= index < self._points.shape[0]:
                    x, y = self._points[index]
                    painter.drawEllipse(QPointF(x, y), radius * 1.15, radius * 1.15)

        if 0 <= self._selected < self._points.shape[0]:
            highlight = QPen(QColor("#ffd43b"), 2.0)
            highlight.setCosmetic(True)
            painter.setPen(highlight)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            x, y = self._points[self._selected]
            painter.drawEllipse(QPointF(x, y), radius * 2.2, radius * 2.2)
