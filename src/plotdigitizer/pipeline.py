"""The automatic digitization pipeline: an image in, a reviewable session out.

This is the piece that does the clicking. Where a person would place four calibration
handles and then click every marker, :meth:`AutoDigitizer.run` detects the frame, reads
the tick labels, fits the axes, separates the series by colour and extracts their
points - producing exactly the state a hand-driven session would have reached.

It is deliberately total: every stage has a fallback, so a figure it cannot fully
understand still opens with handles on the frame corners and whatever series it did
find, ready to be corrected by hand. Refusing to produce a result would just be a
worse starting point than an imperfect one. What it will not do is hide its own
uncertainty - each stage reports a confidence, and anything doubtful lands in
``warnings`` for the UI to show.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend, select_backend
from .calibration import AxisCalibration, AxisScale, Calibration, fit_axis
from .detect.extract import ExtractionSettings, choose_mode, extract_points
from .detect.frame import InkImage, PlotFrame, analyse_ink, detect_frame
from .detect.labels import AxisLabels, detect_labels
from .detect.ocr import get_default_ocr
from .detect.series import PlotContent, discover_series
from .detect.ticks import TickSet, detect_ticks

log = logging.getLogger(__name__)

__all__ = ["Series", "DigitizationResult", "AutoDigitizer"]


@dataclass
class Series:
    """One extracted data series, in both pixel and data coordinates."""

    name: str
    color: tuple[int, int, int]
    settings: ExtractionSettings
    pixel_points: np.ndarray                  # (N, 2) as (column, row)
    data_points: np.ndarray                   # (N, 2) as (x, y)
    visible: bool = True
    mask: np.ndarray | None = field(default=None, repr=False)
    #: When this series was combined from others, everything needed to restore them.
    #: Its presence is what makes a combine reversible rather than destructive.
    sources: list[dict] | None = field(default=None, repr=False)

    @property
    def hex_color(self) -> str:
        r, g, b = self.color
        return f"#{r:02x}{g:02x}{b:02x}"

    @property
    def count(self) -> int:
        return int(self.pixel_points.shape[0])

    @property
    def is_combined(self) -> bool:
        """True when this series can be split back into the ones it came from."""
        return bool(self.sources)

    def recompute_data(self, calibration: Calibration | None) -> None:
        """Refresh the data coordinates after a calibration change."""
        if calibration is None or not calibration.is_valid or self.pixel_points.size == 0:
            self.data_points = np.empty((0, 2), dtype=float)
            return
        self.data_points = calibration.to_data(self.pixel_points)


@dataclass
class DigitizationResult:
    """Everything the automatic pass worked out about one image."""

    image_shape: tuple[int, int]
    frame: PlotFrame
    calibration: Calibration
    series: list[Series] = field(default_factory=list)
    x_labels: AxisLabels | None = None
    y_labels: AxisLabels | None = None
    x_ticks: TickSet | None = None
    y_ticks: TickSet | None = None
    content: PlotContent | None = None
    warnings: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    device: str = "cpu"

    @property
    def calibrated(self) -> bool:
        """True when the axes were read from the figure rather than guessed."""
        return bool(self.confidence.get("x_axis", 0.0) > 0 and self.confidence.get("y_axis", 0.0) > 0)

    @property
    def total_points(self) -> int:
        return sum(s.count for s in self.series)

    def recompute(self) -> None:
        for series in self.series:
            series.recompute_data(self.calibration)


def _inward_reach(tickset: TickSet) -> int:
    """How far this axis's ticks intrude into the plot area, 0 if they point outwards."""
    if tickset is None or tickset.side != "inside":
        return 0
    marks = list(tickset.major) + list(tickset.minor)
    return int(max((m.length for m in marks), default=0))


def _fallback_axis(p1: float, p2: float, scale: AxisScale = AxisScale.LINEAR) -> AxisCalibration:
    """A placeholder axis spanning the frame, for the user to type real values into."""
    return AxisCalibration(p1=p1, v1=0.0, p2=p2, v2=1.0, scale=scale)


class AutoDigitizer:
    """Runs the full detection stack over an image."""

    def __init__(self, device: str = "auto", ocr=None, max_series: int = 8,
                 colour_tolerance: float = 28.0, backend: Backend | None = None,
                 ocr_engine: str = "template"):
        self.backend = backend or select_backend(device)
        # "template" is the default because it reads rendered axis labels perfectly
        # and needs nothing downloaded; "neural" is for scans and odd typefaces and
        # falls back to the template engine if the model cannot be loaded.
        self.ocr = ocr or get_default_ocr(prefer_neural=(ocr_engine == "neural"))
        self.max_series = max_series
        self.colour_tolerance = colour_tolerance

    # -- stages ------------------------------------------------------------------

    def _calibrate_axis(self, labels: AxisLabels, axis: str, frame: PlotFrame,
                        warnings: list[str]) -> tuple[AxisCalibration, float]:
        pairs = labels.pairs if labels else []
        if len(pairs) >= 2:
            pixels = [p for p, _ in pairs]
            values = [v for _, v in pairs]
            fitted = fit_axis(pixels, values)
            if fitted is not None and fitted.is_valid:
                confidence = fitted.fit.confidence if fitted.fit else 0.5
                if fitted.fit and fitted.fit.outliers:
                    warnings.append(
                        f"{axis} axis: ignored {len(fitted.fit.outliers)} tick label(s) that "
                        f"did not fit the scale - check them before exporting"
                    )
                if confidence < 0.5:
                    warnings.append(f"{axis} axis calibration is weak; verify the limits")
                return fitted, confidence
            warnings.append(f"{axis} axis: tick labels did not fit any supported scale")
        else:
            warnings.append(f"{axis} axis: could not read enough tick labels; enter the limits by hand")

        if axis == "x":
            return _fallback_axis(frame.left, frame.right), 0.0
        return _fallback_axis(frame.bottom, frame.top), 0.0

    def _build_series(self, content: PlotContent, calibration: Calibration) -> list[Series]:
        series: list[Series] = []
        for candidate in content.series:
            settings = ExtractionSettings(mode=choose_mode(candidate.mask))
            points = extract_points(candidate.mask, settings, self.backend)
            item = Series(
                name=candidate.name,
                color=candidate.color_rgb,
                settings=settings,
                pixel_points=points,
                data_points=np.empty((0, 2), dtype=float),
                mask=candidate.mask,
            )
            item.recompute_data(calibration)
            series.append(item)
        return series

    # -- entry point -------------------------------------------------------------

    def run(self, rgb: np.ndarray) -> DigitizationResult:
        """Detect axes, calibration and data for one image."""
        started = time.perf_counter()
        rgb = np.asarray(rgb)
        warnings: list[str] = []

        ink: InkImage = analyse_ink(rgb)
        frame = detect_frame(rgb, ink)
        if frame.confidence < 0.4:
            warnings.append("Plot frame was hard to find; drag the calibration handles onto the axes")

        x_ticks = detect_ticks(ink, frame, "x")
        y_ticks = detect_ticks(ink, frame, "y")

        x_labels = detect_labels(ink, frame, x_ticks, "x", self.ocr)
        y_labels = detect_labels(ink, frame, y_ticks, "y", self.ocr)

        x_axis, x_confidence = self._calibrate_axis(x_labels, "x", frame, warnings)
        y_axis, y_confidence = self._calibrate_axis(y_labels, "y", frame, warnings)
        calibration = Calibration(x=x_axis, y=y_axis)

        content = discover_series(
            rgb, ink, frame,
            x_ticks=x_ticks.positions, y_ticks=y_ticks.positions,
            backend=self.backend, max_series=self.max_series,
            colour_tolerance=self.colour_tolerance,
            inward_reach_x=_inward_reach(x_ticks),
            inward_reach_y=_inward_reach(y_ticks),
        )
        series = self._build_series(content, calibration)
        if not series:
            warnings.append("No data series found inside the plot area")

        result = DigitizationResult(
            image_shape=(rgb.shape[0], rgb.shape[1]),
            frame=frame,
            calibration=calibration,
            series=series,
            x_labels=x_labels, y_labels=y_labels,
            x_ticks=x_ticks, y_ticks=y_ticks,
            content=content,
            warnings=warnings,
            confidence={
                "frame": frame.confidence,
                "x_axis": x_confidence,
                "y_axis": y_confidence,
                "x_labels": x_labels.confidence if x_labels else 0.0,
                "y_labels": y_labels.confidence if y_labels else 0.0,
            },
            elapsed_seconds=time.perf_counter() - started,
            device=self.backend.kind,
        )
        return result


def digitize(rgb: np.ndarray, device: str = "auto", **kwargs) -> DigitizationResult:
    """Convenience wrapper for one-shot use."""
    return AutoDigitizer(device=device, **kwargs).run(rgb)
