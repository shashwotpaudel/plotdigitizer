"""Find tick marks along the bottom and left spines.

Ticks may point outwards (matplotlib's default) or inwards, so both bands beside the
spine are tried and whichever produces a more convincing set wins.

The defining property of a tick, and the one that separates it from axis labels and
from the data itself, is that it is *attached to the spine*: a short stub starting at
the axis line and running perpendicular to it. Searching a narrow band and keeping only
the components that begin at the band's inner edge encodes exactly that.

Major and minor ticks are then separated by length, because on a log axis the minor
ticks are the unevenly spaced ones and only the majors carry labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .frame import InkImage, PlotFrame

__all__ = ["TickMark", "TickSet", "detect_ticks"]


@dataclass
class TickMark:
    """One tick: where it sits along the axis, and how far it sticks out."""

    position: float   # column for an x axis, row for a y axis
    length: int       # extent perpendicular to the axis
    width: int        # extent along the axis
    strength: float


@dataclass
class TickSet:
    axis: str                      # "x" or "y"
    side: str                      # "outside" or "inside"
    major: list[TickMark] = field(default_factory=list)
    minor: list[TickMark] = field(default_factory=list)
    regularity: float = 0.0        # 1.0 when the majors are perfectly evenly spaced

    @property
    def positions(self) -> np.ndarray:
        return np.array([t.position for t in self.major], dtype=float)

    @property
    def count(self) -> int:
        return len(self.major)

    @property
    def quality(self) -> float:
        """How much this set should be trusted, used to choose between bands."""
        if self.count < 2:
            return 0.0
        return self.regularity * min(1.0, self.count / 4.0)


def _spine_free_offset(ink: InkImage, frame: PlotFrame, axis: str, direction: int) -> int:
    """How far past the nominal spine position the line actually ends.

    Spines are 1-2 px thick and anti-aliased; starting the search band inside that
    smear would merge every tick into the spine as one component.
    """
    h, w = ink.shape
    if axis == "x":
        origin = int(round(frame.bottom))
        lo, hi = int(round(frame.left)), int(round(frame.right))
        span = max(1, hi - lo + 1)
        for step in range(1, 8):
            row = origin + direction * step
            if not (0 <= row < h):
                return step
            if ink.mask[row, lo:hi + 1].sum() < 0.5 * span:
                return step
    else:
        origin = int(round(frame.left))
        lo, hi = int(round(frame.top)), int(round(frame.bottom))
        span = max(1, hi - lo + 1)
        for step in range(1, 8):
            col = origin + direction * step
            if not (0 <= col < w):
                return step
            if ink.mask[lo:hi + 1, col].sum() < 0.5 * span:
                return step
    return 2


def _extract_band(ink: InkImage, frame: PlotFrame, axis: str, side: str,
                  depth: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (mask, distance, origin) for a band beside the spine.

    Both arrays are oriented as (depth, along): row 0 is the pixel nearest the spine
    and the column index runs along the axis, so one component-analysis routine can
    serve all four combinations of axis and side.
    """
    h, w = ink.shape
    direction = 1 if side == "outside" else -1
    if axis == "y":
        direction = -direction  # "outside" a left spine means towards smaller columns

    offset = _spine_free_offset(ink, frame, axis, direction)

    if axis == "x":
        origin_row = int(round(frame.bottom)) + direction * offset
        lo, hi = int(round(frame.left)), int(round(frame.right))
        lo, hi = max(0, lo), min(w - 1, hi)
        if direction > 0:
            r0, r1 = origin_row, min(h, origin_row + depth)
            if r1 <= r0:
                return None
            sub_mask = ink.mask[r0:r1, lo:hi + 1]
            sub_dist = ink.distance[r0:r1, lo:hi + 1]
        else:
            r1, r0 = origin_row + 1, max(0, origin_row + 1 - depth)
            if r1 <= r0:
                return None
            sub_mask = ink.mask[r0:r1, lo:hi + 1][::-1, :]
            sub_dist = ink.distance[r0:r1, lo:hi + 1][::-1, :]
        return sub_mask, sub_dist, float(lo)

    origin_col = int(round(frame.left)) + direction * offset
    lo, hi = int(round(frame.top)), int(round(frame.bottom))
    lo, hi = max(0, lo), min(h - 1, hi)
    if direction > 0:
        c0, c1 = origin_col, min(w, origin_col + depth)
        if c1 <= c0:
            return None
        sub_mask = ink.mask[lo:hi + 1, c0:c1].T
        sub_dist = ink.distance[lo:hi + 1, c0:c1].T
    else:
        c1, c0 = origin_col + 1, max(0, origin_col + 1 - depth)
        if c1 <= c0:
            return None
        sub_mask = ink.mask[lo:hi + 1, c0:c1][:, ::-1].T
        sub_dist = ink.distance[lo:hi + 1, c0:c1][:, ::-1].T
    return np.ascontiguousarray(sub_mask), np.ascontiguousarray(sub_dist), float(lo)


def _regularity(positions: np.ndarray) -> float:
    """1.0 for perfectly even spacing, decaying as the spacing scatters.

    Major ticks are evenly spaced on both linear and log axes - it is the *minor*
    ticks that bunch up on a log axis - so this is a fair test in either case.
    """
    if positions.size < 3:
        return 0.6 if positions.size == 2 else 0.0
    gaps = np.diff(np.sort(positions))
    if gaps.size == 0 or gaps.mean() <= 0:
        return 0.0
    return float(max(0.0, 1.0 - gaps.std() / gaps.mean()))


def _candidates_from_band(band: tuple[np.ndarray, np.ndarray, float],
                          max_width: int) -> tuple[list[TickMark], list[TickMark]]:
    """Components attached to the spine, split into stubs and full-depth intruders."""
    mask, distance, origin = band
    if mask.size == 0 or not mask.any():
        return [], []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    band_depth, along_extent = mask.shape

    marks: list[TickMark] = []
    spanning: list[TickMark] = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if y > 1:                      # not attached to the spine
            continue
        if ch < 2:                     # a single row is noise, not a stub
            continue
        if area < 2:
            continue

        # A tick stops short of the far edge of the band. Anything running the full
        # depth is a line passing through - most often the perpendicular spine at a
        # corner, which would otherwise pose as the longest tick on the axis and drag
        # every real tick into the "minor" bucket.
        spans_depth = y + ch >= band_depth
        at_start, at_end = x <= 1, x + cw >= along_extent - 1
        # At a plot corner the tick, the spine it sits on and the perpendicular
        # spine all fuse into one blob, so the centroid is meaningless - but such a
        # tick is by definition at the frame edge, which is where we place it.
        is_corner = spans_depth and (at_start or at_end)

        if not is_corner and cw > max_width:
            continue                   # too wide along the axis to be a tick

        position = origin + (0.0 if at_start else float(along_extent - 1)) if is_corner \
            else origin + float(centroids[i][0])
        mark = TickMark(
            position=position,
            length=int(ch),
            width=int(cw),
            strength=float(distance[labels == i].mean()),
        )
        (spanning if spans_depth else marks).append(mark)

    if not marks:
        # The band was simply too shallow to clear the ticks.
        return spanning, []
    return marks, spanning


def _split_major_minor(marks: list[TickMark]) -> tuple[list[TickMark], list[TickMark]]:
    """Longer stubs are major ticks; shorter ones are minors and carry no label."""
    if not marks:
        return [], []
    lengths = np.array([m.length for m in marks], dtype=float)
    longest = float(lengths.max())
    if longest <= 0:
        return list(marks), []
    cutoff = 0.7 * longest
    major = [m for m in marks if m.length >= cutoff]
    minor = [m for m in marks if m.length < cutoff]
    # A handful of stragglers among many uniform ticks is more likely noise than a
    # real minor-tick system, so only treat the split as meaningful when it is clean.
    if len(major) < 2:
        return list(marks), []
    return major, minor


def _merge_adjacent(marks: list[TickMark], min_gap: float) -> list[TickMark]:
    """Collapse ticks whose anti-aliased edges split them into neighbours."""
    if not marks:
        return []
    ordered = sorted(marks, key=lambda m: m.position)
    merged = [ordered[0]]
    for mark in ordered[1:]:
        prev = merged[-1]
        if mark.position - prev.position < min_gap:
            if mark.length > prev.length or (mark.length == prev.length and mark.strength > prev.strength):
                merged[-1] = mark
        else:
            merged.append(mark)
    return merged


def _recover_corner_ticks(major: list[TickMark], spanning: list[TickMark],
                          spacing_tolerance: float) -> list[TickMark]:
    """Rescue ticks that merged with a perpendicular spine at a corner.

    A tick sitting exactly on the plot corner is fused with the spine crossing there,
    so it looks like a line running through the band rather than a stub. It is a real
    tick only if it continues the even progression of the ticks we are already sure
    about - which is a much safer test than accepting every intruder.
    """
    if not spanning or len(major) < 2:
        return major

    positions = np.sort(np.array([m.position for m in major], dtype=float))
    spacing = float(np.median(np.diff(positions)))
    if spacing <= 0:
        return major
    tolerance = max(spacing_tolerance, 0.03 * spacing)
    median_length = int(np.median([m.length for m in major]))

    recovered = list(major)
    for candidate in spanning:
        offset = (candidate.position - positions[0]) / spacing
        if abs(offset - round(offset)) * spacing > tolerance:
            continue
        # Only extend the sequence; a hit inside the existing run is already covered.
        if positions[0] - 0.5 * spacing < candidate.position < positions[-1] + 0.5 * spacing:
            continue
        recovered.append(TickMark(
            position=candidate.position, length=median_length,
            width=candidate.width, strength=candidate.strength,
        ))
    return recovered


def detect_ticks(ink: InkImage, frame: PlotFrame, axis: str,
                 max_depth: int | None = None) -> TickSet:
    """Detect the ticks on one axis, trying both the outward and inward bands."""
    axis_span = frame.width if axis == "x" else frame.height
    if max_depth is None:
        # Ticks are a small fraction of the plot; a generous cap still excludes labels.
        max_depth = int(max(6, min(28, 0.06 * axis_span)))
    max_width = int(max(3, 0.02 * axis_span))

    best = TickSet(axis=axis, side="outside")
    for side in ("outside", "inside"):
        band = _extract_band(ink, frame, axis, side, max_depth)
        if band is None:
            continue
        stubs, spanning = _candidates_from_band(band, max_width)
        min_gap = max(2.0, 0.004 * axis_span)
        stubs = _merge_adjacent(stubs, min_gap)
        major, minor = _split_major_minor(stubs)
        major = _merge_adjacent(_recover_corner_ticks(major, spanning, min_gap), min_gap)
        candidate = TickSet(
            axis=axis, side=side, major=major, minor=minor,
            regularity=_regularity(np.array([m.position for m in major], dtype=float)),
        )
        if candidate.quality > best.quality:
            best = candidate
    return best
