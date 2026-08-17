"""Turn a series mask into a list of points.

Two modes, matching what the drawing actually is:

``SCATTER`` treats each blob as one measurement and returns its centroid. Markers that
overlap are split back apart by clustering the blob's pixels, because a merged pair
would otherwise report a single point halfway between two real ones.

``CURVE`` scans column by column and averages the ink in each, the same idea as
WebPlotDigitizer's averaging window. Gaps shorter than a threshold are bridged so a
dashed line comes back as one series rather than forty fragments.

The mode is chosen by looking at the spacing between blobs relative to their size:
markers stand apart, whereas the dashes of a dashed line nearly touch. It is only ever
a default - the UI exposes the choice, because some figures are genuinely ambiguous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ..backend import Backend, NumpyBackend

log = logging.getLogger(__name__)

__all__ = ["ExtractionMode", "ExtractionSettings", "extract_points", "choose_mode"]


class ExtractionMode(str, Enum):
    SCATTER = "scatter"
    CURVE = "curve"

    @property
    def label(self) -> str:
        return "Scatter / markers" if self is ExtractionMode.SCATTER else "Line / curve"


@dataclass
class ExtractionSettings:
    """The knobs the Series panel exposes."""

    mode: ExtractionMode = ExtractionMode.SCATTER
    #: Column step for curve mode, in pixels.
    x_step: int = 2
    #: Ignore blobs smaller than this fraction of the median blob.
    min_blob_fraction: float = 0.25
    #: Bridge gaps in a curve shorter than this many pixels.
    max_gap: int = 24
    #: Median-filter window for curve mode, in samples. 1 disables smoothing.
    smoothing: int = 3
    #: Cap on returned points; 0 keeps every sample.
    max_points: int = 0

    def copy(self) -> "ExtractionSettings":
        return ExtractionSettings(**vars(self))


def _components(mask: np.ndarray):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return count, labels, stats, centroids


#: How much the blob heights may vary, relative to the median, and still be markers.
_MARKER_HEIGHT_SPREAD = 0.15
#: Width may vary further, because adjacent markers merge into wider blobs.
_MARKER_WIDTH_SPREAD = 1.0
#: Floor on solidity, to reject a straight dashed line whose dashes are uniform too.
_MARKER_MIN_THICKNESS = 0.4


def _spread(values: np.ndarray) -> float:
    """Interquartile range relative to the median - robust to a few merged blobs."""
    median = float(np.median(values))
    if median <= 0:
        return float("inf")
    return float(np.percentile(values, 75) - np.percentile(values, 25)) / median


def choose_mode(mask: np.ndarray) -> ExtractionMode:
    """Guess whether this mask is a set of markers or a line.

    The giveaway is that markers are *identical to each other*. A scatter plot stamps
    the same glyph at every point, so every blob has the same bounding box. The pieces
    of a line do not: each fragment of a curve is as tall as that stretch of the curve
    is steep, so their heights vary with the local slope.

    Spacing is the tempting discriminator and it does not work - on a dense scatter the
    markers' boxes already overlap in x, so the gaps are indistinguishable from those
    of a dashed line. Uniformity does work, and it does not care how the points are
    laid out.

    Solidity is kept as a secondary check for the one genuinely ambiguous case: a
    perfectly straight dashed line has uniform dashes too. Those dashes are thin
    slivers, whereas a marker is a filled glyph roughly as thick as it is wide.
    """
    count, _, stats, _ = _components(mask)
    blobs = [stats[i] for i in range(1, count) if stats[i][cv2.CC_STAT_AREA] >= 3]
    if len(blobs) <= 2:
        # One or two blobs is a continuous stroke, not a plausible scatter.
        return ExtractionMode.CURVE

    areas = np.array([b[cv2.CC_STAT_AREA] for b in blobs], dtype=float)
    widths = np.maximum(1.0, np.array([b[cv2.CC_STAT_WIDTH] for b in blobs], dtype=float))
    heights = np.maximum(1.0, np.array([b[cv2.CC_STAT_HEIGHT] for b in blobs], dtype=float))

    # A line broken by a few specks: one blob dwarfs the rest.
    if areas.max() > 6.0 * np.median(areas) and widths.max() > 8.0 * np.median(widths):
        return ExtractionMode.CURVE

    thickness = float(np.median(areas / widths)) / float(np.median(widths))
    if (_spread(heights) <= _MARKER_HEIGHT_SPREAD
            and _spread(widths) <= _MARKER_WIDTH_SPREAD
            and thickness >= _MARKER_MIN_THICKNESS):
        return ExtractionMode.SCATTER
    return ExtractionMode.CURVE


def _split_blob(coords: np.ndarray, parts: int, backend: Backend) -> list[np.ndarray]:
    """Break one merged blob into ``parts`` clusters of pixels."""
    if parts <= 1 or coords.shape[0] < parts:
        return [coords]
    points = coords.astype(np.float32)
    centres = backend.kmeans(points, parts, iters=15, seed=0)
    labels, _ = backend.nearest_center(points, centres)
    return [coords[labels == k] for k in range(parts) if np.any(labels == k)]


def _extract_scatter(mask: np.ndarray, settings: ExtractionSettings,
                     backend: Backend) -> np.ndarray:
    count, labels, stats, centroids = _components(mask)
    if count <= 1:
        return np.empty((0, 2), dtype=float)

    areas = np.array([stats[i][cv2.CC_STAT_AREA] for i in range(1, count)], dtype=float)
    if areas.size == 0:
        return np.empty((0, 2), dtype=float)
    floor = max(3.0, settings.min_blob_fraction * float(np.median(areas)))

    # Size of a *single* marker, taken from a low percentile of the surviving blobs.
    # The median would be wrong: merging only ever makes a blob bigger, so on an image
    # where many markers touch the median is itself a merged pair, and then nothing
    # looks large enough to split.
    singles = areas[areas >= floor]
    unit_area = float(np.percentile(singles, 25)) if singles.size else 0.0

    points: list[tuple[float, float]] = []
    for i in range(1, count):
        area = float(stats[i][cv2.CC_STAT_AREA])
        if area < floor:
            continue
        # Markers that touch merge into one blob whose centroid is between them, so
        # split anything substantially larger than a single marker.
        parts = int(round(area / unit_area)) if unit_area > 0 else 1
        if parts <= 1:
            points.append((float(centroids[i][0]), float(centroids[i][1])))
            continue
        coords = np.argwhere(labels == i)  # (row, col)
        for cluster in _split_blob(coords, min(parts, 8), backend):
            if cluster.size == 0:
                continue
            points.append((float(cluster[:, 1].mean()), float(cluster[:, 0].mean())))

    if not points:
        return np.empty((0, 2), dtype=float)
    array = np.asarray(points, dtype=float)
    return array[np.argsort(array[:, 0])]


def _column_centres(mask: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean ink row for every sampled column that contains any."""
    columns = np.arange(0, mask.shape[1], max(1, step))
    rows = np.arange(mask.shape[0], dtype=float)
    sub = mask[:, columns]
    counts = sub.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centres = (sub * rows[:, None]).sum(axis=0) / counts
    valid = counts > 0
    return columns[valid].astype(float), centres[valid]


def _bridge_and_smooth(xs: np.ndarray, ys: np.ndarray,
                       settings: ExtractionSettings) -> tuple[np.ndarray, np.ndarray]:
    """Join dash gaps and take out single-sample jitter."""
    if xs.size == 0:
        return xs, ys

    # Split where the horizontal gap is too wide to be a dash gap; a real break in the
    # data (a curve leaving and re-entering the axes) must stay a break.
    gaps = np.diff(xs)
    breaks = np.flatnonzero(gaps > settings.max_gap)
    segments = np.split(np.arange(xs.size), breaks + 1)

    out_x: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    for segment in segments:
        if segment.size == 0:
            continue
        sx, sy = xs[segment], ys[segment]
        window = settings.smoothing
        if window > 1 and sy.size >= window:
            if window % 2 == 0:
                window += 1
            padded = np.pad(sy, window // 2, mode="edge")
            strided = np.lib.stride_tricks.sliding_window_view(padded, window)
            sy = np.median(strided, axis=1)
        out_x.append(sx)
        out_y.append(sy)

    return np.concatenate(out_x), np.concatenate(out_y)


def _extract_curve(mask: np.ndarray, settings: ExtractionSettings) -> np.ndarray:
    xs, ys = _column_centres(mask, settings.x_step)
    if xs.size == 0:
        return np.empty((0, 2), dtype=float)
    xs, ys = _bridge_and_smooth(xs, ys, settings)
    return np.column_stack([xs, ys])


def _resample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Thin an over-sampled curve down to a requested number of points, evenly in x."""
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points)
    return points[np.round(indices).astype(int)]


def extract_points(mask: np.ndarray, settings: ExtractionSettings,
                   backend: Backend | None = None) -> np.ndarray:
    """Extract (column, row) pixel points from one series mask."""
    backend = backend or NumpyBackend()
    if not mask.any():
        return np.empty((0, 2), dtype=float)

    if settings.mode is ExtractionMode.SCATTER:
        points = _extract_scatter(mask, settings, backend)
    else:
        points = _extract_curve(mask, settings)
    return _resample(points, settings.max_points)
