"""Locate the tick labels, read them, and pair each with the tick it belongs to.

The search is deliberately narrow. Labels live in a strip immediately beyond the
ticks, and within that strip the tick labels form the text block *closest to the axis* -
anything further out is the axis title. Picking that block by proximity, instead of
trying to classify text semantically, is what keeps "response" and "time (s)" out of
the calibration.

Scientific-notation multipliers ('1e-3' above the y axis) are hunted separately in the
plot corners and applied to every value on that axis. Missing one is a three-orders-of-
magnitude error that otherwise looks perfectly plausible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .frame import InkImage, PlotFrame
from .ocr import Glyph, TextLine, parse_number, parse_offset, split_superscript
from .ticks import TickSet

log = logging.getLogger(__name__)

__all__ = ["AxisLabels", "detect_labels"]


@dataclass
class AxisLabels:
    """Everything read beside one axis."""

    axis: str
    lines: list[TextLine] = field(default_factory=list)
    multiplier: float = 1.0
    multiplier_text: str = ""
    #: (tick pixel position, data value) pairs ready for the calibration fit.
    pairs: list[tuple[float, float]] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        if not self.lines:
            return 0.0
        read = [ln for ln in self.lines if ln.value is not None]
        if not read:
            return 0.0
        return float(np.mean([ln.confidence for ln in read]) * len(read) / len(self.lines))


def _extract_glyphs(ink: InkImage, row0: int, row1: int, col0: int, col1: int,
                    min_area: int = 2) -> list[Glyph]:
    """Connected blobs of ink in a region, as glyph crops in image coordinates."""
    h, w = ink.shape
    row0, row1 = max(0, row0), min(h, row1)
    col0, col1 = max(0, col0), min(w, col1)
    if row1 - row0 < 2 or col1 - col0 < 2:
        return []

    sub_mask = ink.mask[row0:row1, col0:col1].astype(np.uint8)
    if not sub_mask.any():
        return []
    sub_dist = ink.distance[row0:row1, col0:col1]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub_mask, connectivity=8)
    glyphs: list[Glyph] = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        # Take the intensities inside the component's own box, masked to a slightly
        # grown component so the anti-aliased halo is kept but a neighbouring
        # character's ink is not. The box itself stays tight, because the templates
        # are measured from tight boxes too and a one-pixel disagreement about where
        # a glyph starts is enough to break the match.
        component = (labels == i).astype(np.uint8)
        grown = cv2.dilate(component, np.ones((3, 3), np.uint8))
        bitmap = (sub_dist[y:y + ch, x:x + cw] * grown[y:y + ch, x:x + cw]).astype(np.float32)
        peak = float(bitmap.max())
        if peak <= 0:
            continue
        glyphs.append(Glyph(
            bitmap=bitmap / peak,
            x0=col0 + x, y0=row0 + y, x1=col0 + x + cw, y1=row0 + y + ch,
        ))
    return glyphs


def _cluster_intervals(items, lo, hi, gap: float) -> list[list]:
    """Group items whose [lo, hi) intervals are within ``gap`` of each other."""
    if not items:
        return []
    ordered = sorted(items, key=lo)
    groups = [[ordered[0]]]
    edge = hi(ordered[0])
    for item in ordered[1:]:
        if lo(item) - edge <= gap:
            groups[-1].append(item)
            edge = max(edge, hi(item))
        else:
            groups.append([item])
            edge = hi(item)
    return groups


def _median_glyph_height(glyphs: list[Glyph]) -> float:
    if not glyphs:
        return 0.0
    return float(np.median([g.height for g in glyphs]))


def _drop_clipped(glyphs: list[Glyph], near: int, far: int, axis: str) -> list[Glyph]:
    """Remove glyphs cut off by either edge of the search strip.

    A real tick label sits wholly inside its strip. Anything touching a boundary is the
    tail of something living on the other side of it - the bottom half of the corner
    y-axis label spilling under the x axis, or the left half of the corner x label
    reaching into the y strip. Either would be read as an extra character on the
    neighbouring label.
    """
    if not glyphs:
        return glyphs
    if axis == "x":
        kept = [g for g in glyphs if g.y0 > near and g.y1 < far]
    else:
        kept = [g for g in glyphs if g.x0 > near and g.x1 < far]
    return kept or glyphs


def _drop_rotated_runs(glyphs: list[Glyph], min_run: int = 3) -> list[Glyph]:
    """Remove vertically stacked, centre-aligned glyph chains: rotated axis titles.

    A rotated y-axis title overlaps the tick labels horizontally, so no left/right
    split can separate them. What does separate them is direction: the title's letters
    are stacked one above another with their centres in a line and only a hair of space
    between, whereas tick labels are a tick-spacing apart.
    """
    n = len(glyphs)
    if n < min_run:
        return list(glyphs)

    height = _median_glyph_height(glyphs)
    max_gap = 1.6 * height

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = glyphs[i], glyphs[j]
            if abs(a.cx - b.cx) > 0.6 * max(a.width, b.width):
                continue
            if abs(a.cy - b.cy) > max_gap:
                continue
            parent[find(i)] = find(j)

    chains: dict[int, list[int]] = {}
    for i in range(n):
        chains.setdefault(find(i), []).append(i)

    rotated = {i for members in chains.values() if len(members) >= min_run for i in members}
    kept = [g for k, g in enumerate(glyphs) if k not in rotated]
    return kept or list(glyphs)


def _tick_reach(tickset: TickSet) -> int:
    """How far the ticks stick out, so the label strip can start beyond them."""
    marks = list(tickset.major) + list(tickset.minor)
    if not marks or tickset.side == "inside":
        return 0
    return int(max(m.length for m in marks))


def _x_axis_label_rows(ink: InkImage, frame: PlotFrame, tickset: TickSet) -> list[Glyph]:
    """Glyphs from the tick-label row under the x axis (not the axis title)."""
    start = int(round(frame.bottom)) + _tick_reach(tickset) + 1
    depth = int(max(20, 0.35 * frame.height))
    margin = int(0.06 * frame.width)
    glyphs = _extract_glyphs(ink, start, start + depth,
                             int(frame.left) - margin, int(frame.right) + margin + 1)
    glyphs = _drop_clipped(glyphs, near=start, far=start + depth, axis="x")
    glyphs = _drop_rotated_runs(glyphs)
    if not glyphs:
        return []

    # Rows of text, nearest the axis first; the tick labels are the first row.
    height = _median_glyph_height(glyphs)
    rows = _cluster_intervals(glyphs, lambda g: g.y0, lambda g: g.y1, gap=max(2.0, 0.35 * height))
    rows.sort(key=lambda grp: min(g.y0 for g in grp))
    return rows[0] if rows else []


def _y_axis_label_glyphs(ink: InkImage, frame: PlotFrame, tickset: TickSet) -> list[Glyph]:
    """Glyphs from the tick-label column left of the y axis (not the rotated title)."""
    end = int(round(frame.left)) - _tick_reach(tickset)
    depth = int(max(24, 0.35 * frame.width))
    margin = int(0.06 * frame.height)
    glyphs = _extract_glyphs(ink, int(frame.top) - margin, int(frame.bottom) + margin + 1,
                             end - depth, end)
    glyphs = _drop_clipped(glyphs, near=end - depth, far=end, axis="y")
    glyphs = _drop_rotated_runs(glyphs)
    if not glyphs:
        return []

    # Split into vertical columns of text and keep the one nearest the axis. The tick
    # labels form a single such column - they are right-aligned against the axis, so
    # their boxes all overlap horizontally - while the rotated axis title forms its own
    # column further out. Grouping by column beats testing each glyph individually,
    # because a title glyph can sit on the same rows as a label and inherit its edge.
    height = _median_glyph_height(glyphs)
    columns = _cluster_intervals(glyphs, lambda g: g.x0, lambda g: g.x1,
                                 gap=max(3.0, 0.8 * height))
    return max(columns, key=lambda col: max(g.x1 for g in col))


def _read_groups(groups: list[list[Glyph]], ocr) -> list[TextLine]:
    """Recognise each label, handling raised exponents before the engine sees them.

    The superscript split is done here rather than inside an engine so that every
    engine gets it. A general-purpose recogniser flattens 10 with a raised 2 into the
    string "102" - a plausible-looking number that is wrong by a factor of one - and no
    amount of downstream parsing can recover the exponent once it has been lost.
    """
    lines: list[TextLine] = []
    for group in groups:
        if not group:
            continue
        base, exponent = split_superscript(group)
        text, confidence = ocr.read_line(base)
        if exponent:
            exponent_text, exponent_confidence = ocr.read_line(exponent)
            text = f"{text}^{exponent_text}"
            confidence = 0.5 * (confidence + exponent_confidence)
        line = TextLine(glyphs=list(group), text=text, confidence=confidence)
        line.value = parse_number(text)
        lines.append(line)
    return lines


def _find_multiplier(ink: InkImage, frame: PlotFrame, axis: str, ocr,
                     label_box: tuple[int, int] | None) -> tuple[float, str]:
    """Look for a scientific-notation multiplier in the corner beside the axis.

    ``label_box`` is the extent the tick labels already occupy, and the search starts
    beyond it. Without that exclusion the last x tick label - very often something like
    '10' - reads as a perfectly good power of ten and scales the whole axis by it.
    """
    h, w = ink.shape
    reference = max(8, int(0.03 * min(frame.width, frame.height)))
    if axis == "y":
        # matplotlib puts the y multiplier above the top-left corner of the axes,
        # starting at the axis line itself - to the right of every tick label.
        row0, row1 = max(0, int(frame.top) - 4 * reference), max(0, int(frame.top) - 2)
        col0, col1 = int(frame.left) - 2, int(frame.left + 0.45 * frame.width)
        if label_box is not None:
            col0 = max(col0, label_box[1] + 1)
    else:
        # ...and the x multiplier below the bottom-right corner, under the labels.
        row0 = int(frame.bottom) + reference
        if label_box is not None:
            row0 = max(row0, label_box[1] + 1)
        row1 = h
        col0, col1 = int(frame.right - 0.35 * frame.width), w

    glyphs = _extract_glyphs(ink, row0, row1, col0, col1)
    if not glyphs:
        return 1.0, ""

    height = _median_glyph_height(glyphs)
    rows = _cluster_intervals(glyphs, lambda g: g.y0, lambda g: g.y1, gap=max(2.0, 0.35 * height))
    for row in rows:
        blocks = _cluster_intervals(row, lambda g: g.x0, lambda g: g.x1,
                                    gap=max(3.0, 0.9 * height))
        for block in blocks:
            text, _ = ocr.read_line(block)
            factor = parse_offset(text)
            # A plot title also lives up here; only something that reads as a power
            # of ten is treated as a multiplier, and 1.0 would be a no-op anyway.
            if factor is not None and factor > 0 and abs(np.log10(factor)) >= 1.0:
                return float(factor), text
    return 1.0, ""


def _pair_with_ticks(lines: list[TextLine], tickset: TickSet, axis: str,
                     multiplier: float) -> list[tuple[float, float]]:
    """Match each readable label to its nearest tick, one label per tick."""
    positions = tickset.positions
    if positions.size == 0:
        return []
    spacing = float(np.median(np.diff(np.sort(positions)))) if positions.size > 1 else float("inf")
    tolerance = 0.5 * spacing if np.isfinite(spacing) else 1e9

    taken: dict[int, tuple[float, float]] = {}
    for line in lines:
        if line.value is None:
            continue
        centre = line.cx if axis == "x" else line.cy
        distances = np.abs(positions - centre)
        index = int(np.argmin(distances))
        if distances[index] > tolerance:
            continue
        # If two labels claim one tick, keep the better-centred one.
        previous = taken.get(index)
        if previous is not None and previous[0] <= distances[index]:
            continue
        taken[index] = (float(distances[index]), float(line.value) * multiplier)

    return [(float(positions[i]), value) for i, (_, value) in sorted(taken.items())]


def detect_labels(ink: InkImage, frame: PlotFrame, tickset: TickSet, axis: str,
                  ocr) -> AxisLabels:
    """Read the tick labels for one axis and pair them with tick positions."""
    if axis == "x":
        glyphs = _x_axis_label_rows(ink, frame, tickset)
        grouping = (lambda g: g.x0, lambda g: g.x1)
    else:
        glyphs = _y_axis_label_glyphs(ink, frame, tickset)
        grouping = (lambda g: g.y0, lambda g: g.y1)

    result = AxisLabels(axis=axis)
    if not glyphs:
        return result

    height = _median_glyph_height(glyphs)
    if axis == "x":
        # Characters within a number nearly touch; separate labels are far apart.
        groups = _cluster_intervals(glyphs, *grouping, gap=max(3.0, 0.55 * height))
    else:
        groups = _cluster_intervals(glyphs, *grouping, gap=max(2.0, 0.35 * height))

    result.lines = _read_groups(groups, ocr)

    # The band the labels occupy, so the multiplier search can start past it.
    if axis == "x":
        label_box = (min(g.y0 for g in glyphs), max(g.y1 for g in glyphs))
    else:
        label_box = (min(g.x0 for g in glyphs), max(g.x1 for g in glyphs))

    result.multiplier, result.multiplier_text = _find_multiplier(ink, frame, axis, ocr, label_box)
    result.pairs = _pair_with_ticks(result.lines, tickset, axis, result.multiplier)
    return result
