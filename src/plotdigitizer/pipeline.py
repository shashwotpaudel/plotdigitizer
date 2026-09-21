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

import colorsys

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .backend import Backend, select_backend
from .calibration import AxisCalibration, AxisScale, Calibration, fit_axis
from .detect.extract import ExtractionMode, ExtractionSettings, choose_mode, extract_points
from .detect.frame import InkImage, PlotFrame, analyse_ink, detect_frame
from .detect.labels import AxisLabels, detect_labels
from .detect.ocr import get_default_ocr
from .detect.separate import separate_series
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
    #: How the series is drawn - "dotted line", "hollow triangle markers" - when that
    #: is what told it apart from others of the same colour. Empty otherwise.
    style: str = ""
    #: Colour to show it in, when its own ink colour would not do: series separated
    #: from one black mask would otherwise all be drawn black, on top of black ink.
    display: tuple[int, int, int] | None = None
    #: The points the separator found, and the mode they are for. Its mask alone cannot
    #: always give them back - markers merged into a band have one centre each that
    #: the band's blobs do not show - so a change of spacing thins these instead.
    found: np.ndarray | None = field(default=None, repr=False)
    found_mode: ExtractionMode | None = field(default=None, repr=False)

    @property
    def hex_color(self) -> str:
        r, g, b = self.color
        return f"#{r:02x}{g:02x}{b:02x}"

    @property
    def display_color(self) -> tuple[int, int, int]:
        """The colour the interface draws this series in."""
        return tuple(self.display) if self.display else tuple(self.color)

    @property
    def display_hex(self) -> str:
        r, g, b = self.display_color
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


#: Colours for series separated from one ink. Strong, distinct hues that read against
#: black ink and against each other.
SEPARATED_PALETTE = [
    (174, 62, 201), (47, 158, 68), (214, 51, 108), (12, 133, 153),
    (245, 159, 0), (92, 148, 13), (140, 90, 40), (40, 60, 160),
]
#: Colours the canvas already uses for its own overlays: the calibration handles and
#: the plot frame. A series drawn in either would be lost against them.
_OVERLAY_COLOURS = [(232, 89, 12), (255, 169, 77), (64, 160, 255)]


def distinct_colour(existing: list[tuple[int, int, int]], palette: list[tuple[int, int, int]],
                    start: int = 0, distance: float = 90.0) -> tuple[int, int, int]:
    """A colour at least ``distance`` (RGB) from every one in ``existing``.

    The palette is tried first, from ``start`` round; when all of it is taken, hues are
    generated a golden angle apart until one is free. Only when even that fails - a
    screen crowded with colours - does the least crowded candidate win.
    """
    taken = [np.asarray(c, dtype=float) for c in list(existing) + _OVERLAY_COLOURS]

    def room(colour) -> float:
        rgb = np.asarray(colour, dtype=float)
        return min((float(np.linalg.norm(rgb - t)) for t in taken), default=np.inf)

    candidates = [palette[(start + i) % len(palette)] for i in range(len(palette))]
    for i in range(36):
        hue = (0.07 + 0.618034 * i) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.75 if i % 2 else 0.6)
        candidates.append(tuple(int(round(255 * v)) for v in rgb))
    for colour in candidates:
        if room(colour) > distance:
            return colour
    return max(candidates, key=room)


def _display_colours(count: int, avoid: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """``count`` colours to draw separated series in, none close to what is on screen."""
    chosen: list[tuple[int, int, int]] = []
    for _ in range(count):
        chosen.append(distinct_colour(list(avoid) + chosen, SEPARATED_PALETTE))
    return chosen


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
                 ocr_engine: str = "template", separate: bool = True):
        self.backend = backend or select_backend(device)
        # "template" is the default because it reads rendered axis labels perfectly
        # and needs nothing downloaded; "neural" is for scans and odd typefaces and
        # falls back to the template engine if the model cannot be loaded.
        self.ocr = ocr or get_default_ocr(prefer_neural=(ocr_engine == "neural"))
        self.max_series = max_series
        self.colour_tolerance = colour_tolerance
        #: Split series that share a colour by marker shape, line style and continuity.
        self.separate = separate

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

    def _split(self, content: PlotContent, frame: PlotFrame) -> list:
        """Each colour's separated parts, or ``None`` where it stays one series."""
        candidates = content.series
        if not self.separate:
            return [None] * len(candidates)
        splits = []
        # Largest colours first, as discover_series orders them; each may split only
        # while the total stays within max_series, which is a promise to the caller.
        budget = self.max_series - len(candidates)
        for index, candidate in enumerate(candidates):
            others = np.zeros_like(candidate.mask)
            for j, other in enumerate(candidates):
                if j != index:
                    others |= other.mask
            # Another colour's anti-aliased edge often lands in this colour, as a thin
            # broken trail beside that curve, and a curve drawn over markers of this
            # colour cuts them into new shapes. The separator is told where the other
            # colours are, so that it trusts neither.
            parts = separate_series(candidate.mask, plot_width=int(frame.width),
                                    foreign=others, ink=content.data_mask)
            if parts is not None and len(parts) - 1 > budget:
                log.info("not splitting %s: %d series would exceed max_series=%d",
                         candidate.hex_color, len(parts), self.max_series)
                parts = None
            if parts is not None:
                budget -= len(parts) - 1
            splits.append(parts)
        return splits

    def _build_series(self, content: PlotContent, calibration: Calibration,
                      frame: PlotFrame, warnings: list[str] | None = None) -> list[Series]:
        series: list[Series] = []
        inks = [c.color_rgb for c in content.series]
        used: list[tuple[int, int, int]] = []
        for candidate, parts in zip(content.series, self._split(content, frame)):
            unassigned = getattr(parts, "unassigned", None)
            if unassigned is not None and warnings is not None:
                _, first, last = unassigned
                lo, hi = sorted(calibration.x.to_data(np.array([first, last], dtype=float)))
                warnings.append(
                    f"Some {candidate.hex_color} ink between x = {lo:.4g} and {hi:.4g} is in "
                    f"none of the series found in it; check that stretch against the figure")
            if parts is None:
                settings = ExtractionSettings(mode=choose_mode(candidate.mask))
                pieces = [(candidate.mask, settings, "")]
            else:
                # The separator knows what each part is, which is better evidence than
                # the shape statistics choose_mode works from: the dots of a dotted
                # line are identical glyphs too, but they are still a line.
                # A curve's points come one per column, so the spacing says 1.
                pieces = [(part.mask,
                           ExtractionSettings(mode=ExtractionMode.SCATTER)
                           if part.kind == "markers"
                           else ExtractionSettings(mode=ExtractionMode.CURVE, x_step=1),
                           part.description, part.points)
                          for part in parts]
            shades = _display_colours(len(pieces), inks + used) if parts else [None]
            used.extend(c for c in shades if c)
            for (mask, settings, style, *given), shade in zip(pieces, shades):
                found = None
                if given and given[0] is not None:
                    points = np.asarray(given[0], dtype=float)
                    found = points.copy()
                else:
                    points = extract_points(mask, settings, self.backend)
                item = Series(
                    name=candidate.name,
                    color=candidate.color_rgb,
                    settings=settings,
                    pixel_points=points,
                    data_points=np.empty((0, 2), dtype=float),
                    mask=mask,
                    style=style,
                    display=shade,
                    found=found,
                    found_mode=settings.mode if found is not None else None,
                )
                item.recompute_data(calibration)
                series.append(item)
        for position, item in enumerate(series):
            item.name = f"Series {position + 1}"
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
            shared_ink=self.separate,
        )
        series = self._build_series(content, calibration, frame, warnings)
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
