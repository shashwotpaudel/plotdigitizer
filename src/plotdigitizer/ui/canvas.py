"""The interactive canvas: the figure, the overlay, and the mouse verbs.

Interaction follows plotdigitizer.com so the muscle memory transfers: left-click adds a
point, dragging a point moves it, right-click deletes it, and the wheel zooms. Panning
is on the middle button or space-drag, so the left button stays free for editing.

The scene is one image pixel per unit, which makes every coordinate that crosses the
boundary between the detector and the display the same number.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .items import CalibrationHandleItem, FrameItem, LassoItem, SeriesPointsItem

__all__ = ["PlotCanvas", "numpy_to_qimage", "qimage_to_numpy"]

#: How close, in screen pixels, the cursor must be to grab a point.
PICK_RADIUS_PX = 9.0
#: Cursor travel, in scene units, between points laid down by a trace drag.
TRACE_SPACING_PX = 6.0
#: A right-drag shorter than this, in screen pixels, is a click and not a sweep.
BAND_THRESHOLD_PX = 3


def numpy_to_qimage(rgb: np.ndarray) -> QImage:
    """Wrap an (H, W, 3) uint8 array as a QImage that owns its own memory."""
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = array.shape
    image = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()


def qimage_to_numpy(image: QImage) -> np.ndarray:
    """Convert any QImage to an (H, W, 3) uint8 RGB array.

    Used for images pasted from the clipboard, which arrive in whatever format the
    source application happened to use.
    """
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = converted.width(), converted.height()
    pointer = converted.constBits()
    # Rows are padded to a 4-byte boundary, so copy row by row rather than reshaping.
    stride = converted.bytesPerLine()
    raw = np.frombuffer(memoryview(pointer)[:stride * height], dtype=np.uint8)
    return raw.reshape(height, stride)[:, : width * 3].reshape(height, width, 3).copy()


class PlotCanvas(QGraphicsView):
    """Displays the figure with an editable digitization overlay on top."""

    pointAdded = Signal(int, float, float)          # series, column, row
    pointTraced = Signal(int, float, float)         # series, column, row (drag trail)
    pointMoved = Signal(int, int, float, float)     # series, index, column, row
    pointRemoved = Signal(int, int)                 # series, index
    pointsRemoved = Signal(int, object)             # series, list of indices
    selectionChanged = Signal(int, object)         # series, list of selected indices
    pointNudged = Signal(int, int, float, float)    # series, index, dcol, drow
    pointSelected = Signal(int, int)                # series, index (-1 to clear)
    handleDragged = Signal(str, float)              # handle id, pixel coordinate
    handleReleased = Signal(str)
    cursorMoved = Signal(float, float)              # column, row (NaN when off-image)
    zoomChanged = Signal(float)
    strokeSeeded = Signal(float, float)             # column, row of a trace seed
    sweepFinished = Signal(object)                 # list of (column, row) path samples
    editFinished = Signal()                         # a drag or sweep ended

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#2b2f36"))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)

        self._frame_item = FrameItem()
        self._scene.addItem(self._frame_item)

        self._series_items: list[SeriesPointsItem] = []
        self._handles: dict[str, CalibrationHandleItem] = {}
        self._active_series = 0
        self._image_size = (0, 0)
        self._image_rgb: np.ndarray | None = None

        self._panning = False
        self._space_held = False
        self._pan_origin = QPointF()
        self._dragging_point: tuple[int, int] | None = None
        self._show_points = True
        self._selected: tuple[int, int] | None = None
        # Drag-to-trace: where the last point was laid down during this drag.
        self._trace_anchor: QPointF | None = None
        # Right-drag lasso selection.
        self._lasso_item = LassoItem()
        self._scene.addItem(self._lasso_item)
        self._lasso_origin: QPointF | None = None
        self._lasso_points: list[QPointF] = []
        self._lasso_hit = -1
        self._lasso_mode = "replace"
        self._lasso_shape = "rectangle"
        self._selection_indices: list[int] = []
        self._sweep_points: list[QPointF] = []
        self._seeding = False

    # -- content ----------------------------------------------------------------

    @property
    def image(self) -> np.ndarray | None:
        return self._image_rgb

    def set_image(self, rgb: np.ndarray) -> None:
        self._image_rgb = np.asarray(rgb)
        pixmap = QPixmap.fromImage(numpy_to_qimage(self._image_rgb))
        self._pixmap_item.setPixmap(pixmap)
        self._image_size = (pixmap.width(), pixmap.height())
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.clear_overlay()
        self.zoom_to_fit()

    def clear_overlay(self) -> None:
        for item in self._series_items:
            self._scene.removeItem(item)
        self._series_items.clear()
        for handle in self._handles.values():
            self._scene.removeItem(handle)
        self._handles.clear()
        self._frame_item.set_frame(0, 0, 0, 0)

    def set_frame(self, left: float, top: float, right: float, bottom: float) -> None:
        self._frame_item.set_frame(left, top, right, bottom)

    def set_series(self, series_list) -> None:
        """Rebuild the point overlays from the current series list."""
        while len(self._series_items) > len(series_list):
            self._scene.removeItem(self._series_items.pop())
        while len(self._series_items) < len(series_list):
            item = SeriesPointsItem(QColor("#1f77b4"))
            self._scene.addItem(item)
            self._series_items.append(item)

        for index, (item, series) in enumerate(zip(self._series_items, series_list)):
            item.set_colour(QColor(*series.color))
            item.set_points(series.pixel_points)
            item.setVisible(series.visible and self._show_points)
            item.set_active(index == self._active_series)

    def update_series_points(self, index: int, points: np.ndarray) -> None:
        if 0 <= index < len(self._series_items):
            self._series_items[index].set_points(points)

    def set_active_series(self, index: int) -> None:
        self._active_series = index
        for position, item in enumerate(self._series_items):
            item.set_active(position == index)
            if position != index:
                item.set_selected(-1)

    def set_selected_point(self, series_index: int, point_index: int) -> None:
        for position, item in enumerate(self._series_items):
            item.set_selected(point_index if position == series_index else -1)
        self.set_selected_series_point(series_index, point_index)

    def set_points_visible(self, visible: bool, series_list=None) -> None:
        """Global on/off for the point overlay, independent of per-series visibility."""
        self._show_points = visible
        self.refresh_visibility(series_list)

    def refresh_visibility(self, series_list=None) -> None:
        """Apply per-series visibility together with the global overlay toggle."""
        if series_list is None:
            for item in self._series_items:
                item.setVisible(self._show_points)
            return
        for item, series in zip(self._series_items, series_list):
            item.setVisible(bool(series.visible) and self._show_points)

    def set_frame_visible(self, visible: bool) -> None:
        self._frame_item.setVisible(visible)

    # -- calibration handles -----------------------------------------------------

    def set_handles(self, positions: dict[str, float], span_x, span_y) -> None:
        """Create or move the four calibration handles.

        ``span_x``/``span_y`` are the extents the handle lines are drawn across, so a
        handle reads as belonging to its axis rather than floating over the figure.
        """
        definitions = [("x1", "x1", True), ("x2", "x2", True),
                       ("y1", "y1", False), ("y2", "y2", False)]
        for handle_id, label, vertical in definitions:
            handle = self._handles.get(handle_id)
            if handle is None:
                handle = CalibrationHandleItem(handle_id, label, vertical)
                handle.moved.connect(self.handleDragged)
                handle.released.connect(self.handleReleased)
                self._scene.addItem(handle)
                self._handles[handle_id] = handle
            handle.set_span(*(span_y if vertical else span_x))
            handle.set_position(positions[handle_id])
        self._sync_handle_scales()

    def set_handles_visible(self, visible: bool) -> None:
        for handle in self._handles.values():
            handle.setVisible(visible)

    def _sync_handle_scales(self) -> None:
        """Keep the handles' screen-sized grab regions correct as the zoom changes."""
        scale = self.zoom()
        for handle in self._handles.values():
            handle.set_view_scale(scale)

    def set_selected_series_point(self, series_index: int, point_index: int) -> None:
        """Remember which point the keyboard should nudge."""
        self._selected = (series_index, point_index) if point_index >= 0 else None

    def begin_stroke_seed(self, armed: bool) -> None:
        """Arm (or disarm) the next left-click to seed a stroke trace."""
        self._seeding = armed
        self.setCursor(Qt.CursorShape.CrossCursor if armed else Qt.CursorShape.ArrowCursor)

    # -- zoom and pan ------------------------------------------------------------

    def zoom_to_fit(self) -> None:
        if self._image_size == (0, 0):
            return
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._sync_handle_scales()
        self.zoomChanged.emit(self.zoom())

    def zoom_to_actual(self) -> None:
        self.resetTransform()
        self._sync_handle_scales()
        self.zoomChanged.emit(self.zoom())

    def zoom(self) -> float:
        return float(self.transform().m11())

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        target = self.zoom() * factor
        if not (0.02 <= target <= 80.0):
            return
        if anchor is None:
            anchor = self.mapToScene(self.viewport().rect().center())
        before = self.mapFromScene(anchor)
        self.scale(factor, factor)
        after = self.mapFromScene(anchor)
        delta = after - before
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
        self._sync_handle_scales()
        self.zoomChanged.emit(self.zoom())

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.zoom_by(1.18 ** steps, self.mapToScene(event.position().toPoint()))
        event.accept()

    # -- mouse -------------------------------------------------------------------

    def _pick_radius(self) -> float:
        return PICK_RADIUS_PX / max(1e-6, self.zoom())

    def _hit(self, scene_pos: QPointF) -> int:
        if not (0 <= self._active_series < len(self._series_items)):
            return -1
        item = self._series_items[self._active_series]
        if not item.isVisible():
            return -1
        return item.nearest(scene_pos.x(), scene_pos.y(), self._pick_radius())

    def _over_handle(self, view_pos) -> bool:
        for item in self.items(view_pos):
            if isinstance(item, CalibrationHandleItem) and item.isVisible():
                return True
        return False

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.MouseButton.MiddleButton or self._space_held:
            self._panning = True
            self._pan_origin = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if (self._seeding and event.button() == Qt.MouseButton.LeftButton
                and self._scene.sceneRect().contains(scene_pos)):
            self.strokeSeeded.emit(scene_pos.x(), scene_pos.y())
            event.accept()
            return

        # A data marker under the cursor always wins. The handles are only grabbable at
        # their grips now, but this ordering is what guarantees that editing a point can
        # never be intercepted by a calibration line lying across the data.
        index = self._hit(scene_pos)

        if index < 0 and self._over_handle(event.position().toPoint()):
            super().mousePressEvent(event)      # let the handle take the drag
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_sweep_modifier(event.modifiers()) \
                    and self._scene.sceneRect().contains(scene_pos):
                # A guided sweep: follow the curve the user drags along, rather than
                # editing whatever happens to be under the press.
                self._sweep_points = [scene_pos]
                event.accept()
                return
            if index >= 0:
                self._dragging_point = (self._active_series, index)
                self._selected = (self._active_series, index)
                self.pointSelected.emit(self._active_series, index)
            elif self._scene.sceneRect().contains(scene_pos):
                self._trace_anchor = scene_pos
                self.pointAdded.emit(self._active_series, scene_pos.x(), scene_pos.y())
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            # A drag never destroys anything now: it selects. The release decides
            # whether this was a click on one point or a sweep over many, and what is
            # caught is shown before any action is taken on it.
            self._lasso_origin = scene_pos
            self._lasso_points = [scene_pos]
            self._lasso_hit = index
            self._lasso_mode = self._selection_mode(event.modifiers())
            event.accept()
            return

        super().mousePressEvent(event)

    @staticmethod
    def _is_sweep_modifier(modifiers) -> bool:
        """Alt starts a guided sweep - and so does Ctrl.

        Alt+drag is what was asked for, but a good many Linux desktops grab it to move
        windows, in which case the application never sees the event at all. Ctrl costs
        nothing here (on the left button it is otherwise unused) and gives the gesture a
        way to work on those systems.
        """
        return bool(modifiers & (Qt.KeyboardModifier.AltModifier
                                 | Qt.KeyboardModifier.ControlModifier))

    @staticmethod
    def _selection_mode(modifiers) -> str:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return "add"
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return "subtract"
        return "replace"

    def _lasso_path(self, current: QPointF) -> QPainterPath:
        """The shape enclosed so far, in scene coordinates."""
        path = QPainterPath()
        if self._lasso_origin is None:
            return path
        if self._lasso_shape == "rectangle":
            path.addRect(QRectF(self._lasso_origin, current).normalized())
        elif len(self._lasso_points) >= 3:
            path.addPolygon(QPolygonF(self._lasso_points))
            path.closeSubpath()
        return path

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._scene.sceneRect().contains(scene_pos):
            self.cursorMoved.emit(scene_pos.x(), scene_pos.y())
        else:
            self.cursorMoved.emit(float("nan"), float("nan"))

        if self._panning:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        if self._dragging_point is not None:
            series_index, point_index = self._dragging_point
            self.pointMoved.emit(series_index, point_index, scene_pos.x(), scene_pos.y())
            event.accept()
            return

        if self._lasso_origin is not None:
            self._lasso_points.append(scene_pos)
            if self._travelled(scene_pos):
                self._lasso_item.set_path(self._lasso_path(scene_pos), self._lasso_mode)
            event.accept()
            return

        if self._sweep_points:
            self._sweep_points.append(scene_pos)
            path = QPainterPath(self._sweep_points[0])
            for point in self._sweep_points[1:]:
                path.lineTo(point)
            self._lasso_item.set_path(path, "sweep")
            event.accept()
            return

        if self._trace_anchor is not None:
            # Lay a trail while the button is held. A small wobble never reaches the
            # spacing threshold, so an ordinary click still produces exactly one point.
            delta = scene_pos - self._trace_anchor
            if (delta.x() ** 2 + delta.y() ** 2) >= TRACE_SPACING_PX ** 2 \
                    and self._scene.sceneRect().contains(scene_pos):
                self._trace_anchor = scene_pos
                self.pointTraced.emit(self._active_series, scene_pos.x(), scene_pos.y())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def _travelled(self, current: QPointF) -> bool:
        """Has the cursor moved far enough for this to be a sweep rather than a click?"""
        if self._lasso_origin is None:
            return False
        delta = current - self._lasso_origin
        reach = BAND_THRESHOLD_PX / max(1e-6, self.zoom())
        return max(abs(delta.x()), abs(delta.y())) > reach

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space_held
                           else Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if self._sweep_points:
            path = [(p.x(), p.y()) for p in self._sweep_points]
            self._sweep_points = []
            self._lasso_item.clear()
            if len(path) >= 2:
                self.sweepFinished.emit(path)
            event.accept()
            return

        if self._dragging_point is not None:
            self._dragging_point = None
            self.editFinished.emit()
            event.accept()
            return

        if self._trace_anchor is not None:
            self._trace_anchor = None
            self.editFinished.emit()
            event.accept()
            return

        if self._lasso_origin is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            swept = self._travelled(scene_pos)
            mode = self._lasso_mode

            if swept:
                caught = self._points_within(self._lasso_path(scene_pos))
            elif self._lasso_hit >= 0:
                # A plain right-click toggles the one point under the cursor, so single
                # corrections do not need a drag.
                caught = [self._lasso_hit]
                if mode == "replace" and self._lasso_hit in self._selection_indices:
                    mode = "subtract"
            else:
                caught = []
                if mode == "replace":
                    # Clicking empty space clears, the way every canvas tool behaves.
                    self.selectionChanged.emit(self._active_series, [])

            self._lasso_item.clear()
            self._lasso_origin = None
            self._lasso_points = []
            self._lasso_hit = -1

            if caught or (swept and mode == "replace"):
                self.selectionChanged.emit(self._active_series,
                                           self._combine_selection(caught, mode))
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _points_within(self, path: QPainterPath) -> list[int]:
        if not (0 <= self._active_series < len(self._series_items)):
            return []
        item = self._series_items[self._active_series]
        if not item.isVisible():
            return []
        return item.indices_within(path)

    def _combine_selection(self, caught: list[int], mode: str) -> list[int]:
        current = set(self._selection_indices)
        if mode == "add":
            current |= set(caught)
        elif mode == "subtract":
            current -= set(caught)
        else:
            current = set(caught)
        return sorted(current)

    def set_selection(self, indices) -> None:
        """Display a selection decided by the window."""
        self._selection_indices = sorted(int(i) for i in indices)
        for position, item in enumerate(self._series_items):
            item.set_selection(self._selection_indices
                               if position == self._active_series else [])

    def set_lasso_shape(self, shape: str) -> None:
        """'rectangle' or 'freeform'."""
        self._lasso_shape = shape

    # -- keyboard ----------------------------------------------------------------

    #: Nudge distances in scene units: plain, fine (Shift), coarse (Ctrl).
    _NUDGE = {"plain": 1.0, "fine": 0.25, "coarse": 10.0}

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        if event.key() == Qt.Key.Key_Escape:
            self.selectionChanged.emit(self._active_series, [])
            event.accept()
            return

        arrows = {
            Qt.Key.Key_Left: (-1.0, 0.0), Qt.Key.Key_Right: (1.0, 0.0),
            Qt.Key.Key_Up: (0.0, -1.0), Qt.Key.Key_Down: (0.0, 1.0),
        }
        if event.key() in arrows and self._selected is not None:
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                step = self._NUDGE["fine"]
            elif modifiers & Qt.KeyboardModifier.ControlModifier:
                step = self._NUDGE["coarse"]
            else:
                step = self._NUDGE["plain"]
            dx, dy = arrows[event.key()]
            series_index, point_index = self._selected
            self.pointNudged.emit(series_index, point_index, dx * step, dy * step)
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)
