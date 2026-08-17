"""Background threads for the two slow operations, so the window never freezes.

Auto-digitizing a large figure takes a fraction of a second on a GPU and rather longer
on a CPU; re-extracting one series after a slider move is quicker but still enough to
stutter. Both run off the GUI thread.

Re-extraction is coalesced: while a run is in flight, only the most recent pending
request is kept. Dragging a tolerance slider would otherwise queue one job per pixel of
travel and the display would lag seconds behind the control.
"""

from __future__ import annotations

import logging
import traceback

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ..detect.extract import ExtractionSettings, extract_points
from ..pipeline import AutoDigitizer, DigitizationResult

log = logging.getLogger(__name__)

__all__ = ["DigitizeWorker", "ExtractionWorker"]


class DigitizeWorker(QThread):
    """Runs the full automatic pipeline on one image."""

    finished_ok = Signal(object)        # DigitizationResult
    failed = Signal(str)

    def __init__(self, image: np.ndarray, device: str = "auto",
                 max_series: int = 8, colour_tolerance: float = 28.0, parent=None):
        super().__init__(parent)
        self._image = image
        self._device = device
        self._max_series = max_series
        self._colour_tolerance = colour_tolerance

    def run(self) -> None:
        try:
            digitizer = AutoDigitizer(device=self._device, max_series=self._max_series,
                                      colour_tolerance=self._colour_tolerance)
            result: DigitizationResult = digitizer.run(self._image)
        except Exception as exc:                        # noqa: BLE001 - reported to the UI
            log.exception("automatic digitization failed")
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=3)}")
            return
        self.finished_ok.emit(result)


class ExtractionWorker(QObject):
    """Re-extracts a single series when its settings change."""

    ready = Signal(int, object)         # series index, (N, 2) pixel points

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._thread: QThread | None = None
        self._pending: tuple[int, np.ndarray, ExtractionSettings] | None = None
        self._busy = False

    def request(self, index: int, mask: np.ndarray, settings: ExtractionSettings) -> None:
        """Ask for a re-extraction, replacing any request that has not started yet."""
        self._pending = (index, mask, settings.copy())
        if not self._busy:
            self._start_next()

    def _start_next(self) -> None:
        if self._pending is None:
            return
        index, mask, settings = self._pending
        self._pending = None
        self._busy = True

        worker = self

        class _Job(QThread):
            def run(self) -> None:
                try:
                    points = extract_points(mask, settings, worker._backend)
                except Exception:                        # noqa: BLE001
                    log.exception("re-extraction failed")
                    points = np.empty((0, 2), dtype=float)
                worker.ready.emit(index, points)

        self._thread = _Job()
        self._thread.finished.connect(self._on_finished)
        self._thread.start()

    def _on_finished(self) -> None:
        self._busy = False
        self._thread = None
        # Anything that arrived while we were working runs now, and only the last one.
        if self._pending is not None:
            self._start_next()

    def stop(self) -> None:
        self._pending = None
        if self._thread is not None:
            self._thread.wait(2000)
