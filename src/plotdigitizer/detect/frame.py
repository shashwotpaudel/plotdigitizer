"""Find the plot frame - the rectangle the data lives inside.

Everything downstream depends on this: ticks are searched for in bands hugging the
frame, labels in strips beyond it, and data only inside it.

The hard part is not finding long straight lines, it is deciding which of them are
axis spines. A figure with a grid contains a dozen full-width lines and only one of
them is the x axis. Two signals separate them:

  * spines are drawn darker and thicker than gridlines, and
  * a spine's *extent* is exactly the plot box, so the left spine's vertical run
    tells you where the top and bottom of the box are.

That second property is what lets this work on figures drawn with only a left and a
bottom spine, where there is no line at all along two sides of the rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["PlotFrame", "InkImage", "analyse_ink", "detect_frame", "detect_panel"]


@dataclass
class InkImage:
    """An image reduced to 'how far is this pixel from the background'.

    Working in distance-from-background instead of raw darkness is what makes the
    dark-background figures work without a special case.
    """

    distance: np.ndarray      # (H, W) float32, 0 = background
    mask: np.ndarray          # (H, W) bool, distance above the noise floor
    background: np.ndarray    # (3,) float32 RGB
    threshold: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.distance.shape


def _background_colour(rgb: np.ndarray) -> np.ndarray:
    """The most common colour, found on a coarse quantisation of the image.

    The mode beats the border median because some figures carry a coloured margin,
    and it beats the overall mean because a mean of white paper and black ink is grey.
    """
    quant = (rgb.astype(np.int32) >> 3)  # 32 levels per channel
    keys = (quant[..., 0] << 10) | (quant[..., 1] << 5) | quant[..., 2]
    flat = keys.reshape(-1)
    counts = np.bincount(flat)
    dominant = int(np.argmax(counts))
    members = rgb.reshape(-1, 3)[flat == dominant]
    return members.mean(axis=0).astype(np.float32)


def analyse_ink(rgb: np.ndarray, threshold: float = 60.0) -> InkImage:
    """Compute the distance-from-background field and its mask."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    bg = _background_colour(rgb)
    diff = rgb.astype(np.float32) - bg[None, None, :]
    distance = np.sqrt((diff**2).sum(axis=2)).astype(np.float32)
    return InkImage(distance=distance, mask=distance > threshold, background=bg, threshold=threshold)


@dataclass
class LineCandidate:
    """A long straight run of ink, either horizontal or vertical."""

    position: float     # row for a horizontal line, column for a vertical one
    start: int          # first pixel along the line's own direction
    end: int            # last pixel, inclusive
    thickness: int
    strength: float     # mean distance-from-background along the line
    coverage: float     # fraction of [start, end] that is actually inked

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def score(self) -> float:
        # Darkness dominates: it is what separates a spine from a gridline.
        return self.strength * self.coverage * min(1.0, self.thickness / 1.5)


def _find_lines(ink: InkImage, axis: str, min_fraction: float = 0.30) -> list[LineCandidate]:
    """Locate long horizontal (axis='h') or vertical (axis='v') runs of ink.

    A morphological opening with a long 1-D kernel keeps only pixels belonging to a
    run of at least that length, which discards dashed gridlines and text for free.
    """
    mask = ink.mask.astype(np.uint8)
    h, w = mask.shape
    span = w if axis == "h" else h
    min_len = max(12, int(min_fraction * span))
    kernel = np.ones((1, min_len), np.uint8) if axis == "h" else np.ones((min_len, 1), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if not opened.any():
        return []

    # Collapse to a 1-D profile along the direction perpendicular to the lines.
    profile = opened.sum(axis=1 if axis == "h" else 0)
    present = profile > 0

    candidates: list[LineCandidate] = []
    idx = 0
    n = present.size
    while idx < n:
        if not present[idx]:
            idx += 1
            continue
        start_band = idx
        while idx < n and present[idx]:
            idx += 1
        end_band = idx - 1  # inclusive band of adjacent rows/cols forming one line

        band = opened[start_band:end_band + 1, :] if axis == "h" else opened[:, start_band:end_band + 1]
        along = band.any(axis=0) if axis == "h" else band.any(axis=1)
        cols = np.flatnonzero(along)
        if cols.size == 0:
            continue
        first, last = int(cols[0]), int(cols[-1])
        coverage = float(along[first:last + 1].mean())

        # Weight the sub-pixel centre by how much ink each row of the band carries.
        weights = profile[start_band:end_band + 1].astype(np.float64)
        positions = np.arange(start_band, end_band + 1, dtype=np.float64)
        centre = float((positions * weights).sum() / max(weights.sum(), 1e-9))

        region = ink.distance[start_band:end_band + 1, first:last + 1] if axis == "h" \
            else ink.distance[first:last + 1, start_band:end_band + 1]
        sub_mask = band[:, first:last + 1] if axis == "h" else band[first:last + 1, :]
        strength = float(region[sub_mask > 0].mean()) if (sub_mask > 0).any() else 0.0

        candidates.append(LineCandidate(
            position=centre, start=first, end=last,
            thickness=end_band - start_band + 1,
            strength=strength, coverage=coverage,
        ))
    return candidates


@dataclass
class PlotFrame:
    """The data rectangle, in pixel coordinates."""

    left: float
    right: float
    top: float
    bottom: float
    has_left: bool = True
    has_right: bool = True
    has_top: bool = True
    has_bottom: bool = True
    confidence: float = 0.0

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def interior_bounds(self, inset: int = 2) -> tuple[int, int, int, int]:
        """(row0, row1, col0, col1) slice bounds just inside the frame lines."""
        return (
            int(round(self.top + inset)), int(round(self.bottom - inset)) + 1,
            int(round(self.left + inset)), int(round(self.right - inset)) + 1,
        )

    def contains(self, col: float, row: float, tolerance: float = 0.0) -> bool:
        return (self.left - tolerance <= col <= self.right + tolerance
                and self.top - tolerance <= row <= self.bottom + tolerance)

    def to_dict(self) -> dict:
        return {
            "left": self.left, "right": self.right, "top": self.top, "bottom": self.bottom,
            "has_left": self.has_left, "has_right": self.has_right,
            "has_top": self.has_top, "has_bottom": self.has_bottom,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlotFrame":
        return cls(**d)


def _pick_spine(candidates: list[LineCandidate], prefer: str, extent: int) -> LineCandidate | None:
    """Choose the spine on one side.

    Restricting to the outer third of the image is what stops a strong gridline or a
    zero rule through the middle of the plot from being mistaken for an axis.
    """
    if not candidates:
        return None
    limit_lo, limit_hi = 0.35 * extent, 0.65 * extent
    if prefer in ("low",):
        pool = [c for c in candidates if c.position <= limit_lo]
    else:
        pool = [c for c in candidates if c.position >= limit_hi]
    if not pool:
        return None
    best = max(pool, key=lambda c: c.score)
    # Among lines of comparable darkness, take the outermost - a boxed plot drawn
    # inside a figure border should calibrate against the axes, not the border.
    threshold = best.score * 0.75
    strong = [c for c in pool if c.score >= threshold]
    return min(strong, key=lambda c: c.position) if prefer == "low" \
        else max(strong, key=lambda c: c.position)


def detect_panel(rgb: np.ndarray, ink: InkImage,
                 min_area_fraction: float = 0.08) -> tuple[PlotFrame, float] | None:
    """Find the plot area from a *filled* panel rather than from drawn spines.

    Several popular styles - ggplot, seaborn's darkgrid - draw no spines at all. The
    plot area is delimited by a coloured panel with light gridlines on it, so there is
    no long dark line anywhere to find, and spine detection has nothing to work with.

    What those figures do have is two background colours: the figure's and the panel's.
    Comparing the most common colour in the image against the most common colour around
    its border separates them, and the panel's own extent is then the plot rectangle.
    """
    height, width = ink.shape
    if height < 40 or width < 40:
        return None

    # The colour around the edge of the image is the figure background; the dominant
    # colour overall is the panel when a panel exists, and the same thing when it does not.
    border = max(2, int(0.02 * min(height, width)))
    ring = np.concatenate([
        rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3),
        rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3),
    ])
    outer = _background_colour(ring.reshape(1, -1, 3))
    panel_colour = ink.background
    if float(np.linalg.norm(outer - panel_colour)) < 18.0:
        return None                      # one background: an ordinary figure

    # Everything close to the panel colour, as one solid region.
    difference = np.linalg.norm(rgb.astype(np.float32) - panel_colour[None, None, :], axis=2)
    panel = (difference < 26.0).astype(np.uint8)
    panel = cv2.morphologyEx(panel, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(panel, connectivity=8)
    if count <= 1:
        return None
    index = 1 + int(np.argmax([stats[i][cv2.CC_STAT_AREA] for i in range(1, count)]))
    x, y, w, h, area = stats[index]
    if w < 20 or h < 20 or area < min_area_fraction * height * width:
        return None

    # A panel is a rectangle: nearly all of its bounding box should be filled. This is
    # what stops a large irregular patch of background colour from posing as one.
    fill = area / float(w * h)
    if fill < 0.9:
        return None

    frame = PlotFrame(
        left=float(x), right=float(x + w - 1),
        top=float(y), bottom=float(y + h - 1),
        has_left=True, has_right=True, has_top=True, has_bottom=True,
        confidence=float(min(1.0, fill)),
    )
    return frame, fill


def detect_frame(rgb: np.ndarray, ink: InkImage | None = None) -> PlotFrame:
    """Locate the plot rectangle.

    Falls back to the bounding box of all ink when no convincing spines exist, so the
    rest of the pipeline always has something to work with and the user can drag the
    handles into place themselves.
    """
    if ink is None:
        ink = analyse_ink(rgb)
    h, w = ink.shape

    # A filled panel is the stronger signal when there is one: it is an unambiguous
    # rectangle, whereas spine detection on such a figure has no lines to find.
    panel = detect_panel(rgb, ink)

    horizontals = _find_lines(ink, "h")
    verticals = _find_lines(ink, "v")

    top_line = _pick_spine(horizontals, "low", h)
    bottom_line = _pick_spine(horizontals, "high", h)
    left_line = _pick_spine(verticals, "low", w)
    right_line = _pick_spine(verticals, "high", w)

    # Prefer explicit spines; otherwise infer the missing side from the extent of the
    # spines we do have, since a spine spans exactly the plot box.
    left = left_line.position if left_line else None
    right = right_line.position if right_line else None
    top = top_line.position if top_line else None
    bottom = bottom_line.position if bottom_line else None

    if left is None or right is None:
        span = bottom_line or top_line
        if span is not None:
            left = float(span.start) if left is None else left
            right = float(span.end) if right is None else right
    if top is None or bottom is None:
        span = left_line or right_line
        if span is not None:
            top = float(span.start) if top is None else top
            bottom = float(span.end) if bottom is None else bottom

    if None in (left, right, top, bottom):
        # Spines were incomplete. A detected panel answers this exactly; falling back to
        # the bounding box of all ink would swallow the tick labels and axis titles.
        if panel is not None:
            return panel[0]
        rows, cols = np.nonzero(ink.mask)
        if rows.size == 0:
            return PlotFrame(0.0, float(w - 1), 0.0, float(h - 1), confidence=0.0)
        left = float(cols.min()) if left is None else left
        right = float(cols.max()) if right is None else right
        top = float(rows.min()) if top is None else top
        bottom = float(rows.max()) if bottom is None else bottom
        confidence = 0.15
    else:
        found = sum(x is not None for x in (left_line, right_line, top_line, bottom_line))
        coverages = [c.coverage for c in (left_line, right_line, top_line, bottom_line) if c]
        confidence = float(np.mean(coverages)) * (0.6 + 0.1 * found)

    # A degenerate box means the detection went wrong; hand back the whole image.
    if right - left < 8 or bottom - top < 8:
        if panel is not None:
            return panel[0]
        return PlotFrame(0.0, float(w - 1), 0.0, float(h - 1), confidence=0.0)

    # Both methods produced an answer. They normally agree to within a pixel or two;
    # when they do not, the panel is the one to trust, because the "spines" that
    # disagree with a solid rectangle are usually gridlines drawn on top of it.
    if panel is not None:
        candidate = panel[0]
        drift = max(abs(candidate.left - left), abs(candidate.right - right),
                    abs(candidate.top - top), abs(candidate.bottom - bottom))
        if drift > 3.0:
            return candidate
        confidence = max(confidence, candidate.confidence)

    return PlotFrame(
        left=float(left), right=float(right), top=float(top), bottom=float(bottom),
        has_left=left_line is not None, has_right=right_line is not None,
        has_top=top_line is not None, has_bottom=bottom_line is not None,
        confidence=float(min(1.0, confidence)),
    )
