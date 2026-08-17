"""Separate the data from everything else drawn inside the plot frame.

Colour is what distinguishes one series from another, so the work is: collect the ink
inside the frame, throw away the parts that are structure rather than data, and cluster
what remains by colour.

Two things must not survive into a series, because both would read as perfectly
plausible data points:

  * gridlines, which are recognised by *where* they are - a gridline runs the length of
    the plot at a tick position, which no data curve does; and
  * the legend, whose colour patches are drawn in exactly the series colours.

Clusters are found by peak-picking in Lab space rather than by k-means with a guessed
k, because the number of series is not known in advance and a plot's palette is a
handful of well-separated colours, not a continuum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..backend import Backend, NumpyBackend
from .frame import InkImage, PlotFrame

log = logging.getLogger(__name__)

__all__ = ["SeriesCandidate", "PlotContent", "detect_legend", "discover_series"]

#: Minimum separation between two colours in Lab before they count as different series.
MIN_COLOUR_SEPARATION = 22.0
#: A cluster holding less of the plot's ink than this is noise, not a series.
MIN_CLUSTER_FRACTION = 0.015


@dataclass
class SeriesCandidate:
    """One colour-separated series, with the pixels that belong to it."""

    index: int
    name: str
    color_rgb: tuple[int, int, int]
    color_lab: np.ndarray
    mask: np.ndarray               # full-image bool
    pixel_count: int
    structural_score: float = 0.0  # 1.0 means "this looks like gridlines"

    @property
    def hex_color(self) -> str:
        r, g, b = self.color_rgb
        return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class PlotContent:
    """Everything found inside the frame."""

    data_mask: np.ndarray
    series: list[SeriesCandidate] = field(default_factory=list)
    legend_rect: tuple[int, int, int, int] | None = None   # (x0, y0, x1, y1)
    rejected: list[SeriesCandidate] = field(default_factory=list)


def _interior_mask(ink: InkImage, frame: PlotFrame, inset: int = 2) -> np.ndarray:
    """Ink strictly inside the frame, with the spines themselves excluded."""
    mask = np.zeros(ink.shape, dtype=bool)
    r0, r1, c0, c1 = frame.interior_bounds(inset=inset)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(ink.shape[0], r1), min(ink.shape[1], c1)
    if r1 <= r0 or c1 <= c0:
        return mask
    mask[r0:r1, c0:c1] = ink.mask[r0:r1, c0:c1]
    return mask


def _clear_inward_ticks(mask: np.ndarray, frame: PlotFrame,
                        x_ticks: np.ndarray, y_ticks: np.ndarray,
                        reach_x: int, reach_y: int, half_width: int = 3) -> None:
    """Erase tick marks that point into the plot area.

    Inward ticks are drawn in the axis colour inside the frame, so they cluster into a
    convincing extra "series" of evenly spaced dots. They are cleared where they are
    known to be - a short stub at each tick position on each side - rather than by
    insetting the whole interior, which would throw away any data sitting on the axes.
    Ticks are mirrored onto the opposite side because a figure with inward ticks
    usually draws all four.
    """
    height, width = mask.shape
    if reach_x > 0:
        for position in x_ticks:
            lo = max(0, int(round(position)) - half_width)
            hi = min(width, int(round(position)) + half_width + 1)
            bottom = int(round(frame.bottom))
            top = int(round(frame.top))
            mask[max(0, bottom - reach_x):bottom + 1, lo:hi] = False
            mask[top:min(height, top + reach_x + 1), lo:hi] = False
    if reach_y > 0:
        for position in y_ticks:
            lo = max(0, int(round(position)) - half_width)
            hi = min(height, int(round(position)) + half_width + 1)
            left = int(round(frame.left))
            right = int(round(frame.right))
            mask[lo:hi, left:min(width, left + reach_y + 1)] = False
            mask[lo:hi, max(0, right - reach_y):right + 1] = False


def detect_legend(ink: InkImage, frame: PlotFrame,
                  min_area_fraction: float = 0.004,
                  max_area_fraction: float = 0.45) -> tuple[int, int, int, int] | None:
    """Find a framed legend box inside the plot, if there is one.

    A legend is a rectangle whose *inside* is mostly empty - that is what separates it
    from a dense cluster of data. Legends drawn without a frame are not found, and that
    is fine: the cost is a couple of stray points the user can delete, and inventing a
    box that is not there would be worse.
    """
    r0, r1, c0, c1 = frame.interior_bounds(inset=3)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(ink.shape[0], r1), min(ink.shape[1], c1)
    if r1 - r0 < 20 or c1 - c0 < 20:
        return None

    region = ink.mask[r0:r1, c0:c1].astype(np.uint8)
    interior_area = float((r1 - r0) * (c1 - c0))

    # Close small gaps so a rounded or dashed border reads as one contour.
    closed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if not (min_area_fraction * interior_area <= area <= max_area_fraction * interior_area):
            continue
        if w < 12 or h < 12:
            continue
        # Rectangular outline?
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) < 4 or len(approx) > 8:
            continue
        if cv2.contourArea(contour) < 0.55 * area:
            continue
        # Mostly hollow? A legend is a frame around whitespace and a little text.
        inner = region[y + 3:y + h - 3, x + 3:x + w - 3]
        if inner.size == 0 or inner.mean() > 0.35:
            continue
        if area > best_area:
            best, best_area = (c0 + x, r0 + y, c0 + x + w, r0 + y + h), area

    return best


def _structural_score(mask: np.ndarray, frame: PlotFrame,
                      x_ticks: np.ndarray, y_ticks: np.ndarray) -> float:
    """How much of this colour lies in full-length lines sitting at tick positions.

    Both halves of that test matter. "Long straight line" alone would condemn a
    genuinely linear dataset; "at a tick position" alone would condemn any curve that
    happens to pass through one.
    """
    total = int(mask.sum())
    if total == 0:
        return 0.0

    height = max(8, int(0.55 * frame.height))
    width = max(8, int(0.55 * frame.width))
    as_u8 = mask.astype(np.uint8)

    vertical = cv2.morphologyEx(as_u8, cv2.MORPH_OPEN, np.ones((height, 1), np.uint8))
    horizontal = cv2.morphologyEx(as_u8, cv2.MORPH_OPEN, np.ones((1, width), np.uint8))

    aligned = np.zeros_like(as_u8)
    if vertical.any() and x_ticks.size:
        columns = np.zeros(mask.shape[1], dtype=bool)
        for position in x_ticks:
            lo = max(0, int(round(position)) - 2)
            hi = min(mask.shape[1], int(round(position)) + 3)
            columns[lo:hi] = True
        aligned |= vertical * columns[None, :]
    if horizontal.any() and y_ticks.size:
        rows = np.zeros(mask.shape[0], dtype=bool)
        for position in y_ticks:
            lo = max(0, int(round(position)) - 2)
            hi = min(mask.shape[0], int(round(position)) + 3)
            rows[lo:hi] = True
        aligned |= horizontal * rows[:, None]

    return float(aligned.sum()) / float(total)


def _is_blend(colour: np.ndarray, parent_colour: np.ndarray, background: np.ndarray,
              tolerance: float = 20.0) -> bool:
    """Is this colour a mix of the parent's colour and the background?

    Tested in RGB, because that is the space compositing actually happens in.
    """
    axis = parent_colour - background
    length_sq = float(axis @ axis)
    if length_sq < 1e-6:
        return False
    alpha = float((colour - background) @ axis / length_sq)
    if not (0.1 < alpha < 0.98):
        return False
    return float(np.linalg.norm((colour - background) - alpha * axis)) <= tolerance


def _blob_boxes(mask: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    boxes = [(stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3])
             for i in range(1, count) if stats[i][cv2.CC_STAT_AREA] >= 3]
    keep = np.array([centroids[i] for i in range(1, count)
                     if stats[i][cv2.CC_STAT_AREA] >= 3], dtype=float).reshape(-1, 2)
    return boxes, keep


def _encloses(parent_boxes, centroids: np.ndarray) -> float:
    """Fraction of the given centroids that fall inside one of the parent's boxes."""
    if centroids.shape[0] == 0 or not parent_boxes:
        return 0.0
    inside = 0
    for cx, cy in centroids:
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in parent_boxes):
            inside += 1
    return inside / float(centroids.shape[0])


def _merge_marker_parts(candidates: list[SeriesCandidate], background: np.ndarray
                        ) -> list[SeriesCandidate]:
    """Fold the extra colour clusters one drawn series produces back into it.

    A single series routinely paints more than one colour:

      * **anti-aliasing** - every drawn edge is blended with what is behind it, so a
        blue curve on white paper also lays down a band of pale blues;
      * **boundary fringes** - where two colours of one marker meet, the transition is
        a third colour that is a mix of neither one and the background alone;
      * **outlined markers** - ``markeredgecolor`` differs from ``markerfacecolor``, so
        a plain scatter genuinely contains two strong, unrelated colours.

    Left alone each becomes a phantom series sitting exactly on top of the real one.

    Every rule here demands spatial evidence as well as colour evidence, because colour
    alone is not enough to be safe: in a sequential palette a light blue really *is* a
    blend of a dark blue and white, and merging two such series would silently destroy
    data. Requiring the pixels to hug - or to be enclosed by - the parent distinguishes
    "the fringe of that curve" from "a second, paler curve".
    """
    if len(candidates) < 2:
        return candidates

    ordered = sorted(candidates, key=lambda c: c.pixel_count, reverse=True)
    merged: list[SeriesCandidate] = []
    background = np.asarray(background, dtype=float)
    neighbourhoods: dict[int, np.ndarray] = {}
    boxes: dict[int, list] = {}

    for candidate in ordered:
        colour = np.array(candidate.color_rgb, dtype=float)
        candidate_boxes, candidate_centroids = _blob_boxes(candidate.mask)
        absorbed = False

        for parent in merged:
            key = id(parent)
            if key not in neighbourhoods:
                neighbourhoods[key] = cv2.dilate(
                    parent.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                boxes[key] = _blob_boxes(parent.mask)[0]

            touching = float((candidate.mask & neighbourhoods[key]).sum()) / max(
                1, candidate.pixel_count)
            parent_colour = np.array(parent.color_rgb, dtype=float)

            reason = None
            if touching >= 0.7 and _is_blend(colour, parent_colour, background):
                # The colour test has already established that this is the parent's ink
                # at partial coverage, so adjacency only has to rule out a genuinely
                # separate series - which would score near zero, not 0.8. A stricter
                # bar fails on thin strokes: a dotted line or a small open marker is
                # never solid enough to have a fully-inked core for its edges to hug,
                # so a monochrome figure splits into a fan of greys instead of one ink.
                reason = "anti-aliasing"
            elif touching >= 0.9 and candidate.pixel_count < 0.25 * parent.pixel_count:
                # A thin skin of a third colour where two colours of one marker meet.
                reason = "boundary fringe"
            elif (len(candidate_boxes) >= 3
                  and abs(len(candidate_boxes) - len(boxes[key])) <= 0.25 * max(
                      len(candidate_boxes), len(boxes[key]))
                  and _encloses(boxes[key], candidate_centroids) >= 0.9):
                # One blob of this colour sits inside each blob of the parent's: the
                # fill of an outlined marker, or its outline.
                reason = "marker outline"

            if reason is None:
                continue

            log.debug("merging %s into %s (%s)", candidate.hex_color, parent.hex_color, reason)
            parent.mask |= candidate.mask
            parent.pixel_count = int(parent.mask.sum())
            neighbourhoods.pop(key, None)
            boxes.pop(key, None)
            absorbed = True
            break

        if not absorbed:
            merged.append(candidate)
    return merged


def _lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to Lab, where euclidean distance tracks perception."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def _peak_colours(samples: np.ndarray, max_series: int) -> np.ndarray:
    """Pick well-separated dominant colours by histogram peak-picking in Lab.

    Choosing seeds this way means the number of series falls out of the image instead
    of being guessed: a plot uses a few deliberate, well-separated colours, so the
    peaks are unambiguous and a k that is too large simply finds nothing to put in the
    extra clusters.
    """
    if samples.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    quantised = np.floor(samples / 8.0).astype(np.int32)
    keys = (quantised[:, 0] << 20) | (quantised[:, 1] << 10) | quantised[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1]

    total = float(samples.shape[0])
    seeds: list[np.ndarray] = []
    for index in order:
        if len(seeds) >= max_series:
            break
        if counts[index] / total < MIN_CLUSTER_FRACTION:
            break
        members = samples[keys == unique[index]]
        centre = members.mean(axis=0)
        if all(np.linalg.norm(centre - s) >= MIN_COLOUR_SEPARATION for s in seeds):
            seeds.append(centre)

    if not seeds:
        seeds = [samples.mean(axis=0)]
    return np.asarray(seeds, dtype=np.float32)


def discover_series(
    rgb: np.ndarray,
    ink: InkImage,
    frame: PlotFrame,
    x_ticks: np.ndarray | None = None,
    y_ticks: np.ndarray | None = None,
    backend: Backend | None = None,
    max_series: int = 8,
    colour_tolerance: float = 28.0,
    inward_reach_x: int = 0,
    inward_reach_y: int = 0,
) -> PlotContent:
    """Find the colour-separated data series inside the plot frame."""
    backend = backend or NumpyBackend()
    x_ticks = np.asarray(x_ticks if x_ticks is not None else [], dtype=float)
    y_ticks = np.asarray(y_ticks if y_ticks is not None else [], dtype=float)

    mask = _interior_mask(ink, frame)
    _clear_inward_ticks(mask, frame, x_ticks, y_ticks, inward_reach_x, inward_reach_y)
    legend_rect = detect_legend(ink, frame)
    if legend_rect is not None:
        x0, y0, x1, y1 = legend_rect
        mask[max(0, y0):y1, max(0, x0):x1] = False

    content = PlotContent(data_mask=mask, legend_rect=legend_rect)
    if not mask.any():
        return content

    # Cluster on the solidly-inked pixels only: anti-aliased edges are blends of a
    # series colour with the background and would otherwise seed phantom clusters.
    strong = mask & (ink.distance > max(ink.threshold, 0.45 * float(ink.distance[mask].max())))
    if strong.sum() < 20:
        strong = mask

    lab = _lab(rgb)
    samples = lab[strong]
    if samples.shape[0] > 200_000:
        step = samples.shape[0] // 200_000 + 1
        samples = samples[::step]

    seeds = _peak_colours(samples, max_series)
    if seeds.shape[0] == 0:
        return content
    centres = backend.kmeans(samples, seeds.shape[0], iters=12, seed=0) \
        if samples.shape[0] > seeds.shape[0] else seeds

    # Assign every masked pixel - including the anti-aliased ones - to its nearest colour.
    coords = np.argwhere(mask)
    pixels = lab[mask]
    labels, distances = backend.nearest_center(pixels, centres)
    within = distances <= colour_tolerance

    total_ink = float(mask.sum())
    for index in range(centres.shape[0]):
        selected = (labels == index) & within
        count = int(selected.sum())
        if count < max(12, MIN_CLUSTER_FRACTION * total_ink):
            continue

        series_mask = np.zeros_like(mask)
        chosen = coords[selected]
        series_mask[chosen[:, 0], chosen[:, 1]] = True

        centre_lab = centres[index].reshape(1, 1, 3).astype(np.uint8)
        centre_rgb = cv2.cvtColor(centre_lab, cv2.COLOR_LAB2RGB).reshape(3)

        candidate = SeriesCandidate(
            index=index,
            name=f"Series {index + 1}",
            color_rgb=(int(centre_rgb[0]), int(centre_rgb[1]), int(centre_rgb[2])),
            color_lab=centres[index],
            mask=series_mask,
            pixel_count=count,
            structural_score=_structural_score(series_mask, frame, x_ticks, y_ticks),
        )
        if candidate.structural_score > 0.5:
            content.rejected.append(candidate)
        else:
            content.series.append(candidate)

    content.series = _merge_marker_parts(content.series, ink.background)
    content.series.sort(key=lambda s: s.pixel_count, reverse=True)
    for position, series in enumerate(content.series):
        series.index = position
        series.name = f"Series {position + 1}"
    return content
