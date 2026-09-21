"""Split one colour's ink into the series drawn in it.

Colour separation hands over one mask per colour. On a figure printed in black that is
one mask for the whole plot, and averaging it column by column produces a curve that
runs between the real ones and belongs to none of them. What a reader uses instead is
what this module uses:

  * **marker glyphs.** A scatter series stamps the same shape at every point, so a
    shape that recurs many times at the same size is a marker. Each recurring shape is
    then searched for across the whole mask, which also finds the markers that touch a
    line and so never formed a component of their own.
  * **line style.** Once the markers are lifted out, what remains is strokes. A dotted
    line is many tiny components, a dashed line many components of one length, and a
    solid line a few long ones - so the components sort themselves by style.
  * **continuity.** Several curves in one style are followed column by column, each
    track predicting where its curve goes next from its recent slope. Where two cross,
    both predictions fall inside the merged ink and both tracks carry straight on, so
    they come out of the crossing on the curves they went in on.

Then the pieces are put back together the way they were drawn: a dashed and a dotted
track that follow the same path are one dash-dot line, and markers whose centres sit on
a line are that line's markers rather than a series of their own.

The contract with the caller is conservative. If the ink is one series, ``None`` comes
back and the caller carries on with the mask exactly as it was, so a figure that was
digitized correctly before is digitized identically now. Only when there is positive
evidence of several series is anything split.

The limit is the same one a person faces: where two curves in the same style run on
top of each other, nothing in the image says which ink belongs to which. The tracks
carry on through such a stretch on their predictions, but what they report there is
the shared ink, not a separation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Part", "Separation", "SeparationSettings", "separate_series", "stroke_thickness"]


@dataclass
class SeparationSettings:
    """Thresholds for :func:`separate_series`, in pixels unless noted."""

    #: How many times a shape must recur before it counts as a marker.
    min_markers: int = 4
    #: Correlation between two glyph bitmaps for them to be the same marker.
    glyph_similarity: float = 0.8
    #: Correlation needed to find a marker that touches other ink.
    match_threshold: float = 0.55
    #: Fraction of a marker's own pixels that must be present at a match.
    match_recall: float = 0.85
    #: How long a stroke may go without ink before its track ends, in columns.
    max_gap: int = 25
    #: How long an unexplained stroke must persist before it starts a track, in columns.
    birth_length: int = 10
    #: A track shorter than this fraction of the plot width is noise, not a series.
    min_track_fraction: float = 0.15
    #: Two tracks of different styles this often on top of each other are one line.
    colocated_fraction: float = 0.7
    #: A marker whose centre is this close to a line is drawn on that line.
    on_line_px: float = 2.5
    #: This fraction of a marker shape on lines makes it those lines' markers.
    on_line_fraction: float = 0.7
    #: A split that leaves more than this share of the colour's ink in no series is
    #: not trusted, and the colour is kept whole.
    max_unexplained: float = 0.35


class Separation(list):
    """The series found in one colour, and any ink left over that fits none of them."""

    #: (pixels, first column, last column) of ink no series accounts for, if much.
    unassigned: tuple[int, int, int] | None = None


@dataclass(eq=False)
class Part:
    """One series found inside a single colour's ink."""

    kind: str                       # "curve" or "markers"
    style: str                      # "solid", "dashed", "dotted", "dash-dot" or "markers"
    mask: np.ndarray                # full-image bool
    marker: str | None = None       # a short description of the glyph, if any
    #: For a curve: the track's centreline as (column, row), used to judge overlaps.
    centreline: np.ndarray | None = field(default=None, repr=False)
    #: Points to report directly, as (column, row), when the mask alone cannot give
    #: them - markers so crowded they merge into one blob, for instance.
    points: np.ndarray | None = field(default=None, repr=False)
    #: Pixels contributed by each stroke style, to name a line assembled from pieces.
    tally: dict[str, int] = field(default_factory=dict, repr=False)
    #: How many separate pieces of ink each style contributed.
    pieces: dict[str, int] = field(default_factory=dict, repr=False)
    #: A curve that turns back on itself - a loop, a fold - is not a function of x, so
    #: it has no single row per column to read; its points come from its mask instead.
    folded: bool = field(default=False, repr=False)
    #: The separate branches of a folded curve, and which of their ends are still free.
    branches: list = field(default_factory=list, repr=False)
    ends: list = field(default_factory=list, repr=False)

    @property
    def description(self) -> str:
        if self.kind == "markers":
            return f"{self.marker or 'marker'} markers"
        if self.marker:
            return f"{self.style} line with {self.marker} markers"
        return f"{self.style} line"


# -- basic measurements ------------------------------------------------------------


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """For every inked pixel, the length of the vertical run it belongs to."""
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1]), dtype=bool)
    padded[1:-1] = mask
    flat = padded.T.ravel()             # columns laid end to end, separated by False
    edges = np.diff(flat.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    lengths = ends - starts
    out = np.zeros(flat.shape, dtype=np.int32)
    out[np.flatnonzero(flat)] = np.repeat(lengths, lengths)
    return out.reshape(padded.shape[1], padded.shape[0]).T[1:-1]


def _ridge(mask: np.ndarray) -> np.ndarray:
    """Pixels along the middle of each stroke: where the distance to paper peaks.

    A cheap stand-in for a skeleton. Along a line it is the line's centre; in a filled
    blob it is only a short cross through the middle, so blobs weigh little in anything
    measured along it.
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    pad = np.pad(dist, 1)
    centre = pad[1:-1, 1:-1]
    across = (centre >= pad[1:-1, :-2]) & (centre >= pad[1:-1, 2:])
    along = (centre >= pad[:-2, 1:-1]) & (centre >= pad[2:, 1:-1])
    return mask & (across | along)


def _median_by_label(labels: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """Median of ``values`` within each label, 0 where a label has none."""
    out = np.zeros(count)
    if labels.size == 0:
        return out
    order = np.lexsort((values, labels))
    labels, values = labels[order], values[order]
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], labels.size]
    out[labels[starts]] = values[(starts + ends - 1) // 2]
    return out


def stroke_thickness(mask: np.ndarray) -> float:
    """Typical stroke width of the ink's lines, in pixels.

    Each pixel is measured across the narrower of its vertical and horizontal runs,
    which is the stroke's width whether the stroke is flat, upright or diagonal. Only
    elongated ink is measured - lines and dashes - and only along the middle of each
    stroke, so a marker drawn on a line adds a few pixels of its own width rather than
    all of its area. With no elongated ink at all there is no stroke width to measure,
    and a typical two pixels stands in.

    Filled markers would otherwise report their own width. So would a clump of them:
    in a dense scatter, three or four overlapping markers make a component as long as
    a dash. A clump is as wide as the markers around it, where a line is much thinner,
    and that is how clumps are kept out.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 2.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    ridge = _ridge(mask)
    width = np.minimum(_run_lengths(mask), _run_lengths(mask.T).T).astype(float)
    typical = _median_by_label(labels[ridge], width[ridge], count)
    elongated = np.zeros(count, dtype=bool)
    long_sides = np.zeros(count)
    blobs = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if max(w, h) < 6:
            continue
        rows, cols = np.nonzero(labels[y:y + h, x:x + w] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32))
        long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
        long_sides[i] = long_side
        # Long and thin, or long outright - a line curving through the plot, which is
        # long for its own width even where its bounding box is not.
        elongated[i] = (long_side >= 3 * short_side
                        or long_side >= max(40, 8 * typical[i]))
        if not elongated[i] and typical[i] >= 0.6 * max(w, h):
            blobs.append(max(w, h))
    if len(blobs) >= 4:
        # Filled markers are about. Anything elongated that is as wide as they are, and
        # not many times longer, is a few of them run together - unless there are many
        # such pieces of one length, which is a dash pattern drawn in a heavy stroke.
        size = float(np.median(blobs))
        clump = elongated & (typical >= 0.6 * size) & (long_sides < 8 * np.maximum(typical, 1))
        lengths = long_sides[clump]
        dashes = (lengths.size >= 6 and np.mean(np.abs(lengths / np.median(lengths) - 1)
                                                <= 0.25) >= 0.7)
        if not dashes:
            elongated &= ~clump
    lines = elongated[labels] & ridge
    if lines.sum() < 12:
        return 2.0
    return float(max(1.0, np.percentile(width[lines], 30)))


def _column_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Inked spans in every column, each as an (n, 2) array of [start, end) rows."""
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1]), dtype=np.int8)
    padded[1:-1] = mask
    edges = np.diff(padded, axis=0)
    runs = []
    for column in range(mask.shape[1]):
        starts = np.flatnonzero(edges[:, column] == 1)
        ends = np.flatnonzero(edges[:, column] == -1)
        runs.append(np.column_stack([starts, ends]) if starts.size else np.empty((0, 2), int))
    return runs


# -- text --------------------------------------------------------------------------


def _text(mask: np.ndarray) -> np.ndarray:
    """Lettering inside the plot - an unframed legend, an annotation - and its legend
    samples.

    A framed legend is removed before this ever sees the ink, but text drawn in the
    data's colour without a frame stays. Letters are compact and marker-sized, so left
    in they would be matched as markers, or strung together as a dotted line.

    A line of text is recognised by what a row of markers never has: glyphs of mixed
    shapes, a word apart or closer, sitting on one baseline. The lines and markers a
    legend draws beside its labels go with them.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros(mask.shape, dtype=bool)
    if count < 4:
        return out
    x, y, w, h = (stats[:, k].astype(float) for k in range(4))
    tallest = max(16.0, 0.035 * mask.shape[0])
    candidate = (h >= 4) & (h <= tallest) & (w <= 3 * h + 2)
    candidate[0] = False
    index = np.flatnonzero(candidate)
    if index.size < 3:
        return out
    order = index[np.argsort(x[index], kind="stable")]
    parent = {int(i): int(i) for i in index}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for position, i in enumerate(order):
        right = x[i] + w[i]
        for j in order[position + 1:]:
            gap = x[j] - right
            if gap > tallest:
                break
            if gap < -2:
                continue
            overlap = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
            if (overlap >= 0.5 * min(h[i], h[j]) and max(h[i], h[j]) <= 2.5 * min(h[i], h[j])
                    and gap <= 0.9 * max(h[i], h[j])):
                parent[root(int(j))] = root(int(i))
                break

    groups: dict[int, list[int]] = {}
    for i in index:
        groups.setdefault(root(int(i)), []).append(int(i))
    for members in groups.values():
        if len(members) < 3 or not _reads_as_text(members, labels, stats):
            continue
        members = np.asarray(members)
        x0, y0 = x[members].min(), y[members].min()
        x1, y1 = (x[members] + w[members]).max(), (y[members] + h[members]).max()
        height = float(np.median(h[members]))
        middle = 0.5 * (y0 + y1)
        # Punctuation, accents and superscripts within the line; and to its left, the
        # legend's sample of the line or marker it labels.
        inside = ((x >= x0 - 0.3 * height) & (x + w <= x1 + 0.3 * height)
                  & (y >= y0 - 0.5 * height) & (y + h <= y1 + 0.5 * height)
                  & (np.maximum(w, h) <= 1.5 * height))
        # A legend sample is two font sizes long, set most of one before its label,
        # and a font size is about twice the height of the median lower-case letter.
        sample = ((x + w <= x0) & (x + w >= x0 - 6 * height) & (x >= x0 - 7.5 * height)
                  & (np.abs(y + 0.5 * h - middle) <= 0.7 * height) & (h <= 1.6 * height))
        chosen = inside | sample
        chosen[members] = True
        chosen[0] = False
        out |= chosen[labels]
    return out


def _reads_as_text(members: list[int], labels: np.ndarray, stats: np.ndarray) -> bool:
    """Mixed shapes on a shared baseline, as letters are and a row of markers is not."""
    bottoms = np.array([stats[i, 1] + stats[i, 3] for i in members], dtype=float)
    heights = np.array([stats[i, 3] for i in members], dtype=float)
    # Descenders (g, p, y) drop below the line, so most, not all, must share it.
    on_line = np.abs(bottoms - np.median(bottoms)) <= max(1.5, 0.12 * np.median(heights))
    if on_line.mean() < 0.6:
        return False
    vectors = []
    for i in members:
        x, y, w, h, _ = stats[i]
        vectors.append(_descriptor((labels[y:y + h, x:x + w] == i).astype(np.float32)))
    shapes: list[list[np.ndarray]] = []
    for vector in vectors:
        for group in shapes:
            if float(vector @ group[0]) >= 0.85:
                group.append(vector)
                break
        else:
            shapes.append([vector])
    return max(len(g) for g in shapes) <= 0.75 * len(vectors)


# -- error bars --------------------------------------------------------------------


def _bars_of(bars: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """The error bars that run into these markers."""
    if not bars.any() or not markers.any():
        return np.zeros(bars.shape, dtype=bool)
    count, labels = cv2.connectedComponents(bars.astype(np.uint8), connectivity=8)
    grown = cv2.dilate(markers.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    touching = np.unique(labels[grown & bars])
    return np.isin(labels, touching[touching > 0])


def _error_bars(mask: np.ndarray, thickness: float) -> np.ndarray:
    """Error bars: thin, exactly upright or level strokes with a cap or a marker.

    A marker and its error bar are drawn as one piece of ink, so left in they would make
    every marker a shape of its own - a circle on a stick - and the sticks, all alike,
    would be taken for the dashes of a line. A bar is told from a steep stretch of a
    curve by being exactly upright (or level) and thin, and from the riser of a step
    plot by what is at its ends: a short cap centred on it, or a marker it runs
    through. A riser meets a line going off to one side.
    """
    upright = _upright_bars(mask, thickness)
    level = _upright_bars(mask.T, thickness).T
    return upright | level


def _upright_bars(mask: np.ndarray, thickness: float) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    # Error bars are often drawn heavier than the plot's thinnest line.
    width = max(2, int(round(2 * thickness)) + 1)
    across = _run_lengths(mask.T).T              # horizontal run through each pixel
    thin = mask & (across <= width)
    length = max(4, int(round(1.5 * thickness)))
    pieces = thin & (_run_lengths(thin) >= length)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pieces.astype(np.uint8), 8)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        # A bar is a straight upright stroke: a solid rectangle, taller than wide, the
        # same width all the way along - where the point of a triangle widens.
        if w > width + 1 or h < w or area < 0.85 * w * h:
            continue
        runs = across[y:y + h, int(x + w // 2)]
        if np.mean(np.abs(runs - np.median(runs)) <= 1) < 0.7:
            continue
        span = (int(x), int(x + w - 1))
        ends = [_bar_end(mask, y, -1, span, thickness),
                _bar_end(mask, y + h - 1, 1, span, thickness)]
        # A cap at one end and a marker at the other; caps at both ends with the marker
        # in between, where the bar runs straight through a hollow one; or, for bars
        # drawn without caps, a marker at one end and paper at the other. A line ending
        # on an upright line - the corner of a step plot - looks like a capped bar, but
        # it has no marker.
        if (sorted(ends) == ["cap", "marker"]
                or (ends == ["cap", "cap"] and _marker_around(mask, y, h, span, thickness))
                or (sorted(ends) == ["free", "marker"] and h >= 2 * w + 2)):
            # Only where the bar stands clear: through a hollow marker it runs inside
            # the outline, and taking it there would cut the outline open.
            piece = labels == i
            for r in range(y, y + h):
                left, right = _extent(mask, r, span, gaps=True)
                if max(left, right) >= 1.5:
                    piece[r] = False
            out |= piece
            for (row, beyond), kind in zip(((y, -1), (y + h - 1, 1)), ends):
                if kind == "cap":
                    out |= _cap_pixels(mask, row + beyond, beyond, span)
    return out


def _marker_around(mask: np.ndarray, top: int, height: int, span: tuple[int, int],
                   thickness: float) -> bool:
    """Is there a marker around the middle of a bar - its outline either side of it?"""
    reach = max(2.0, thickness)
    middle = range(top + height // 3, top + height - height // 3)
    wide = 0
    for r in middle:
        left, right = _extent(mask, r, span, gaps=True)
        wide += min(left, right) >= reach
    return wide >= 3


def _bar_end(mask: np.ndarray, row: int, beyond: int, span: tuple[int, int],
             thickness: float) -> str:
    """What a thin upright stroke runs into past one end: a cap, a marker, or neither."""
    height = mask.shape[0]
    reach = max(2.0, thickness)
    lo, hi = max(0, span[0] - 1), span[1] + 2
    rows = [row + beyond * k for k in range(1, int(max(4, 3 * (span[1] - span[0] + 1))) + 1)]
    inked = [0 <= r < height and bool(mask[r, lo:hi].any()) for r in rows]
    if not inked[0]:
        return "free"
    wide = []
    for r in rows:
        left, right = _extent(mask, r, span, gaps=True)
        wide.append(min(left, right) >= reach and max(left, right) <= 3 * min(left, right) + 3)
    # A cap is a level stroke no thicker than the bar is wide, the same length all the
    # way through - not a shape widening away from it, like a filled triangle below its
    # point - with paper past it all along its length, where the arms of an x or a star
    # carry on outwards. The bar's own rounded end may poke through; only ink off to the
    # sides counts.
    stroke = next((k for k, v in enumerate(wide) if not v), len(wide))
    if 1 <= stroke <= max(2, span[1] - span[0] + 1):
        lengths = [sum(_extent(mask, r, span)) for r in rows[:stroke]]
        left, right = _extent(mask, rows[0], span)
        c0, c1 = max(0, int(span[0] - left) - 1), int(span[1] + right) + 2
        beside = np.ones(max(0, c1 - c0), dtype=bool)
        beside[max(0, span[0] - 1 - c0):max(0, span[1] + 2 - c0)] = False
        past = [r for r in rows[stroke:stroke + 2] if 0 <= r < height]
        clear = all(not (mask[r, c0:c1] & beside[:mask[r, c0:c1].size]).any() for r in past)
        # ...but a bar that carries on past a crossing stroke - a triangle's base, a
        # line - does not end there.
        ends = len(past) < 2 or not mask[past[1], lo:hi].any()
        if max(lengths) - min(lengths) <= 2 and clear and ends:
            return "cap"
    # A marker spreads out within a few rows - slowly, where a bar meets a triangle's
    # point. Through a hollow marker the bar carries on inside the outline, so the end
    # may already be within it: the rows just before it count too.
    look = max(6, 2 * (span[1] - span[0] + 1))
    inside = []
    for r in (row - beyond * k for k in range(0, look // 2)):
        left, right = _extent(mask, r, span, gaps=True)
        inside.append(min(left, right) >= reach and max(left, right) <= 3 * min(left, right) + 3)
    if sum(wide[:look]) >= 3 or sum(inside) + sum(wide[:look // 2]) >= 3:
        return "marker"
    return "other"


def _extent(mask: np.ndarray, row: int, span: tuple[int, int],
            gaps: bool = False) -> tuple[float, float]:
    """How far ink reaches left and right of a bar's columns along one row.

    With ``gaps``, ink a pixel or two away still counts - the outline of a hollow marker
    is separate from the bar running through it.
    """
    if not (0 <= row < mask.shape[0]):
        return 0.0, 0.0
    line = mask[row]
    step = 3 if gaps else 1

    def reach(start: int, direction: int) -> float:
        last, gap, col = start, 0, start
        while 0 <= col + direction < line.size:
            col += direction
            if line[col]:
                last, gap = col, 0
            else:
                gap += 1
                if gap >= step:
                    break
        return float(abs(last - start))

    return reach(span[0], -1), reach(span[1], 1)


def _cap_pixels(mask: np.ndarray, row: int, beyond: int, span: tuple[int, int]) -> np.ndarray:
    """The cap at the end of a bar: every row of the level stroke there."""
    out = np.zeros(mask.shape, dtype=bool)
    left, right = _extent(mask, row, span)
    c0, c1 = max(0, int(span[0] - left)), int(span[1] + right) + 1
    r = row
    while 0 <= r < mask.shape[0] and mask[r, c0:c1].mean() >= 0.8:
        out[r, c0:c1] = mask[r, c0:c1]
        r += beyond
    return out


# -- markers -----------------------------------------------------------------------


@dataclass(eq=False)
class _MarkerClass:
    template: np.ndarray            # uint8 0/1, the glyph's own pixels
    hull: np.ndarray                # uint8 0/1, the glyph with its inside filled
    vector: np.ndarray
    size: tuple[int, int]           # (w, h)
    members: int
    centres: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Classes of one shape at different sizes share a family, and are one series.
    family: int = -1

    def same_shape(self, other: "_MarkerClass") -> bool:
        """The same marker, possibly drawn at another size."""
        w, h = self.size
        ow, oh = other.size
        return (self.describe() == other.describe()
                and abs(w / max(1, h) - ow / max(1, oh)) <= 0.15 * max(1.0, ow / max(1, oh))
                and float(self.vector @ other.vector) >= 0.7)

    @property
    def anchor(self) -> tuple[float, float]:
        """Where the data point is within the glyph, as (row, column).

        A marker is drawn centred on its point - the middle of its bounding box, not the
        middle of its ink. For a triangle the two differ by a third of its half-height,
        which at high resolution is several pixels, enough to put it off its own line.
        """
        h, w = self.template.shape
        return (h - 1) / 2.0, (w - 1) / 2.0

    @property
    def filled(self) -> bool:
        return float(self.template.sum()) >= 0.8 * float(self.hull.sum())

    def describe(self) -> str:
        """A name a reader would use: "hollow triangle", "filled square", "cross"."""
        hull = self.hull.astype(np.uint8)
        area = float(hull.sum())
        if area == 0:
            return "marker"
        h, w = hull.shape
        solid = float(self.template.sum()) / area
        # Markers made of strokes - +, x, * - enclose nothing, and fill much less of
        # their own outline than any filled shape does.
        if solid > 0.9 and _convexity(self.template) < 0.8:
            return _stroke_marker(self.template)
        fill = "filled" if solid >= 0.8 else "hollow"
        # Counting corners does not work at marker sizes, so coarser measures. A
        # triangle's weight sits towards its base, which nothing symmetric does.
        rows, cols = np.nonzero(hull)
        down = (rows.mean() - (h - 1) / 2) / max(1.0, h)
        across = (cols.mean() - (w - 1) / 2) / max(1.0, w)
        if max(abs(down), abs(across)) > 0.06:
            if abs(down) >= abs(across):
                shape = "triangle" if down > 0 else "down triangle"
            else:
                shape = "left triangle" if across > 0 else "right triangle"
            return f"{fill} {shape}"
        # A square fills its box. A circle and a diamond both leave its corners empty,
        # and at marker sizes no single measure tells them apart once the outline's
        # own width has rounded the diamond's points. Together three do: how much of
        # the box is covered, how wide the outline is at its extremes (a diamond comes
        # to a point), and how far it reaches along the diagonals - 0.71 of the way to
        # the corner for a diamond, all of it for a circle. The weights were fitted to
        # matplotlib's markers from 7 to 54 pixels across.
        # A lone pixel at an extreme - the top of a circle drawn without anti-aliasing -
        # would read as the point of a diamond.
        hull = _trim_points(hull)
        h, w = hull.shape
        area = float(hull.sum())
        box = area / float(w * h)
        if box >= 0.93:
            return f"{fill} square"
        rows, cols = hull.sum(axis=1), hull.sum(axis=0)
        tips = float(rows[0] + rows[-1] + cols[0] + cols[-1]) / float(
            2 * (rows.max() + cols.max()))
        score = box + 0.5 * tips + 1.25 * _diagonal_reach(hull)
        return f"{fill} {'circle' if score >= 1.895 else 'diamond'}"


def _convexity(bitmap: np.ndarray) -> float:
    """A shape's area over the area of its convex hull, pixels taken as squares."""
    rows, cols = np.nonzero(bitmap)
    if rows.size == 0:
        return 1.0
    corners = np.concatenate([np.column_stack([cols + dx, rows + dy])
                              for dx in (0, 1) for dy in (0, 1)]).astype(np.float32)
    hull = cv2.convexHull(corners)
    return float(rows.size) / max(1.0, float(cv2.contourArea(hull)))


def _trim_points(hull: np.ndarray) -> np.ndarray:
    """Drop an outermost row or column that is a single pixel, on each side."""
    if min(hull.shape) < 5:
        return hull
    rows, cols = hull.sum(axis=1), hull.sum(axis=0)
    r0, r1 = int(rows[0] <= 1), hull.shape[0] - int(rows[-1] <= 1)
    c0, c1 = int(cols[0] <= 1), hull.shape[1] - int(cols[-1] <= 1)
    return hull[r0:r1, c0:c1]


def _diagonal_reach(hull: np.ndarray) -> float:
    """How far a shape extends along its box's diagonals, as a fraction of half its size."""
    h, w = hull.shape
    rows, cols = np.nonzero(hull)
    u = (cols - (w - 1) / 2) / (w / 2)
    v = (rows - (h - 1) / 2) / (h / 2)
    near = np.abs(np.abs(u) - np.abs(v)) <= 2.0 / min(w, h)
    reach = []
    for su in (-1, 1):
        for sv in (-1, 1):
            k = near & (np.sign(u) == su) & (np.sign(v) == sv)
            reach.append(float(np.sqrt(u[k] ** 2 + v[k] ** 2).max()) if k.any() else 0.0)
    return float(np.mean(reach))


def _stroke_marker(template: np.ndarray) -> str:
    """Tell a plus from a cross from a star by where the strokes run."""
    h, w = template.shape
    rows, cols = np.nonzero(template)
    y = (rows - (h - 1) / 2) / max(1.0, h / 2)
    x = (cols - (w - 1) / 2) / max(1.0, w / 2)
    band = 0.3
    straight = np.mean((np.abs(x) <= band) | (np.abs(y) <= band))
    diagonal = np.mean(np.abs(np.abs(x) - np.abs(y)) <= band)
    if straight >= 0.85 and straight > diagonal:
        return "plus"
    if diagonal >= 0.85:
        return "cross"
    return "star"


def _descriptor(bitmap: np.ndarray, size: int = 15) -> np.ndarray:
    """A size-free, unit-length vector of the glyph's shape.

    The glyph is framed with a blank margin before it is shrunk. Without one, a filled
    square fills its own box completely, its descriptor is constant, and once the mean
    is taken away there is nothing left to compare - no two squares would ever match.
    """
    h, w = bitmap.shape
    margin = 2
    side = max(h, w) + 2 * margin
    canvas = np.zeros((side, side), np.float32)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = bitmap
    small = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA).ravel()
    small = small - small.mean()
    norm = float(np.linalg.norm(small))
    return small / norm if norm > 1e-9 else small


def _fill_holes(bitmap: np.ndarray) -> np.ndarray:
    """The glyph with everything its outline encloses filled in."""
    h, w = bitmap.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:-1, 1:-1] = bitmap
    flood = canvas.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    outside = (flood == 1) & (canvas == 0)
    return (~outside[1:-1, 1:-1]).astype(np.uint8)


def _glyph_classes(mask: np.ndarray, thickness: float, settings: SeparationSettings,
                   hidden: np.ndarray | None = None) -> list[_MarkerClass]:
    """Shapes that recur often enough, at one size, to be markers.

    Line pieces are kept out first. A dot of a dotted line is barely wider than the
    stroke, and a dash or an arc between two markers is long and thin, so only
    components noticeably larger than the stroke and roughly as tall as they are wide
    are considered at all. Nor is anything touching ``hidden``, where another colour
    lies against this one: a marker with part of it covered is a shape of its own.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    covered = np.zeros(count, dtype=bool)
    if hidden is not None and hidden.any():
        covered[np.unique(labels[hidden & mask])] = True
    # Markers grow with resolution: 60 px is a large marker at 100 dpi and an ordinary
    # one at 600.
    largest = max(60, int(0.025 * max(mask.shape)))
    glyphs = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        big = max(w, h)
        if covered[i] or big < max(5, 2.4 * thickness) or big > largest or min(w, h) < 0.45 * big:
            continue
        own = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        points = np.column_stack(np.nonzero(own)).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(points)
        if min(rw, rh) < 0.45 * max(rw, rh, 1.0) or not _symmetric(own):
            continue
        glyphs.append((own, _descriptor(own), (int(w), int(h)),
                       (x + (w - 1) / 2.0, y + (h - 1) / 2.0)))
    glyphs = [g for g, dot in zip(glyphs, _in_dot_chain(glyphs)) if not dot]

    classes: list[_MarkerClass] = []
    for own, vector, (w, h), _ in glyphs:
        for cls in classes:
            cw, ch = cls.size
            if (abs(w - cw) <= max(2, 0.2 * cw) and abs(h - ch) <= max(2, 0.2 * ch)
                    and float(vector @ cls.vector) >= settings.glyph_similarity):
                cls.members += 1
                break
        else:
            classes.append(_MarkerClass(own, _fill_holes(own), vector, (w, h), 1))
    classes = [c for c in classes if c.members >= settings.min_markers]
    classes = _merge_lookalikes(classes)
    # One shape at several sizes - a size-coded scatter, a bubble chart - is still one
    # series: size is not one of the cues, and the sizes often run continuously.
    for index, cls in enumerate(classes):
        cls.family = next((other.family for other in classes[:index] if other.same_shape(cls)),
                          index)
    return _drop_clumps(classes)


def _symmetric(bitmap: np.ndarray, least: float = 0.7) -> bool:
    """Is a shape a mirror image of itself, left to right or top to bottom?

    Every marker is - circles, squares, triangles pointing any way, crosses, stars -
    and the pieces a line leaves where it turns a corner, an L or a bent dash, are not.
    """
    a = bitmap.astype(bool)
    for flipped in (a[:, ::-1], a[::-1, :]):
        union = float((a | flipped).sum())
        if union and float((a & flipped).sum()) / union >= least:
            return True
    return False


def _in_dot_chain(glyphs: list) -> list[bool]:
    """Which filled blobs are the dots of a heavy dotted line rather than markers.

    A thick dotted line is drawn as blobs the size of small markers. What gives them
    away is how they follow one another: each has a neighbour of its own size either
    side, a couple of its widths away and evenly spaced, as markers placed on a line or
    scattered never are.
    """
    if len(glyphs) < 6:
        return [False] * len(glyphs)
    centres = np.array([g[3] for g in glyphs], dtype=float)
    sizes = np.array([max(g[2]) for g in glyphs], dtype=float)
    filled = np.array([float(g[0].sum()) >= 0.8 * float(_fill_holes(g[0]).sum())
                       for g in glyphs])
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    np.fill_diagonal(d, np.inf)
    alike = np.abs(sizes[:, None] - sizes[None]) <= 0.3 * np.maximum(sizes[:, None], sizes[None])
    d[~alike] = np.inf
    dots = []
    for i in range(len(glyphs)):
        if not filled[i]:
            dots.append(False)
            continue
        near = np.argsort(d[i])[:2]
        a, b = d[i, near]
        if not (np.isfinite(b) and b <= 2.5 * sizes[i]):
            dots.append(False)
            continue
        va, vb = centres[near[0]] - centres[i], centres[near[1]] - centres[i]
        opposite = float(va @ vb) / max(1e-9, float(a * b)) <= -0.8
        dots.append(bool(opposite and abs(a - b) <= 0.25 * max(a, b)))
    # A lone chance alignment in a crowd is not a line: only runs of such blobs count.
    return dots if sum(dots) >= 6 else [False] * len(glyphs)


def _merge_lookalikes(classes: list[_MarkerClass]) -> list[_MarkerClass]:
    """Join classes a reader would call the same marker.

    Sub-pixel placement renders one marker in slightly different bitmaps, and now and
    then a variant falls short of the similarity bar and starts a class of its own.
    Two classes of the same shape and nearly the same size cannot be two series anyway
    - the same marker twice is exactly what cannot be told apart - so they are one.
    """
    classes = sorted(classes, key=lambda c: c.members, reverse=True)
    kept: list[_MarkerClass] = []
    for cls in classes:
        w, h = cls.size
        for home in kept:
            hw, hh = home.size
            if (home.describe() == cls.describe() and abs(w - hw) <= max(2, 0.25 * hw)
                    and abs(h - hh) <= max(2, 0.25 * hh)):
                home.members += cls.members
                break
        else:
            kept.append(cls)
    return kept


def _drop_clumps(classes: list[_MarkerClass]) -> list[_MarkerClass]:
    """Drop classes that are only a few copies of a commoner marker run together.

    In a dense scatter, two or three overlapping markers make a bitmap that recurs as
    often as a rare marker shape would. Such a clump is covered, almost pixel for
    pixel, by copies of a smaller marker that is found far more often; the copies are
    found one by one anyway when that marker is searched for.
    """
    classes = sorted(classes, key=lambda c: c.members, reverse=True)
    kept: list[_MarkerClass] = []
    for cls in classes:
        if not any(_covered_by(cls, small) for small in kept
                   if small.members >= 2 * cls.members
                   and small.template.sum() < cls.template.sum()
                   and small.family != cls.family):
            kept.append(cls)
    return kept


def _covered_by(big: _MarkerClass, small: _MarkerClass, recall: float = 0.85) -> bool:
    th, tw = small.template.shape
    image = np.pad(big.template, ((th, th), (tw, tw))).astype(np.float32)
    dilated = cv2.dilate(image, np.ones((3, 3), np.uint8))
    total = max(1.0, float(small.template.sum()))
    fit = cv2.matchTemplate(dilated, small.template.astype(np.float32), cv2.TM_CCORR) / total
    rows, cols = np.nonzero(fit >= recall)
    if rows.size == 0:
        return False
    covered = np.zeros(image.shape, dtype=bool)
    glyph = small.template.astype(bool)
    for r, c in zip(rows, cols):
        covered[r:r + th, c:c + tw] |= glyph
    covered = cv2.dilate(covered.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ink = image > 0
    return float((covered & ink).sum()) >= 0.9 * float(ink.sum())


def _match(image: np.ndarray, dilated: np.ndarray, fresh: np.ndarray,
           cls: _MarkerClass, settings: SeparationSettings) -> tuple[np.ndarray, np.ndarray]:
    """One pass of template matching; returns centres and scores."""
    template = cls.template.astype(np.float32)
    th, tw = template.shape
    # Correlate with a blank ring around the glyph. A filled square is constant on its
    # own, which leaves normalised correlation undefined and matching everywhere; the
    # ring is also genuinely informative, since a marker is surrounded by paper.
    # A hollow glyph varies across its own box and needs no ring - and a ring would
    # only penalise it where neighbours overlap, which is where help is needed most.
    ring = 1 if cls.filled else 0
    with np.errstate(invalid="ignore"):
        score = cv2.matchTemplate(image, np.pad(template, ring), cv2.TM_CCOEFF_NORMED)
    score = np.nan_to_num(score, nan=0.0)
    # Re-index so that score[r, c] refers to the glyph's own top-left at (r, c).
    score = np.pad(score, ring)
    total = max(1.0, float(template.sum()))
    # Recall against a one-pixel-dilated mask: of the marker's own pixels, how many
    # are inked here. A line crossing a hollow marker lowers the correlation but not
    # this, whereas a stretch of line that merely resembles part of it fails it.
    recall = cv2.matchTemplate(dilated, template, cv2.TM_CCORR) / total
    # And how much of it is ink nothing has explained yet, so that each pass finds
    # new markers rather than the same ones again.
    novelty = cv2.matchTemplate(fresh, template, cv2.TM_CCORR) / total

    radius = max(1, int(0.6 * min(th, tw)))
    peaks = score == cv2.dilate(score, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    keep = (peaks & (score >= settings.match_threshold)
            & (recall >= settings.match_recall) & (novelty >= 0.3))
    rows, cols = np.nonzero(keep)
    if cls.describe() in ("cross", "plus", "star") and rows.size:
        # Two lines crossing make an x or a + of their own. A marker drawn on a line has
        # ink leaving its box in two directions at most; a crossing, in four.
        clear = np.array([_leaving(image, r, c, th, tw) < 3 for r, c in zip(rows, cols)])
        rows, cols = rows[clear], cols[clear]
    oy, ox = cls.anchor
    centres = np.column_stack([cols + ox, rows + oy])
    return centres, score[rows, cols]


def _leaving(image: np.ndarray, top: int, left: int, height: int, width: int) -> int:
    """How many separate strokes leave a box: the arcs of ink in a ring just outside it."""
    pad = 2
    r0, c0 = top - pad, left - pad
    r1, c1 = top + height + pad, left + width + pad
    if r0 < 0 or c0 < 0 or r1 > image.shape[0] or c1 > image.shape[1]:
        return 0
    window = image[r0:r1, c0:c1] > 0
    ring = np.concatenate([window[0, :], window[1:, -1], window[-1, -2::-1], window[-2:0:-1, 0]])
    if not ring.any():
        return 0
    # Count runs of ink round the ring, joining the two ends.
    changes = np.flatnonzero(np.diff(np.r_[ring, ring[:1]].astype(np.int8)) == 1)
    return int(changes.size) if not ring.all() else 1


def _locate(mask: np.ndarray, cls: _MarkerClass, settings: SeparationSettings) -> None:
    """Every place this marker appears, including where it touches other ink.

    Where markers overlap, each spoils the correlation of its neighbours, so a single
    pass finds only the ones that stand clear. The search is therefore repeated: the
    ink explained by markers already found is set aside, and the next pass looks only
    for markers made mostly of ink that is still unexplained. Overlapping markers come
    away a layer at a time.
    """
    th, tw = cls.template.shape
    if mask.shape[0] < th or mask.shape[1] < tw:
        return
    image = mask.astype(np.float32)
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    found = np.empty((0, 2))
    scores = np.empty(0)
    explained = np.zeros(mask.shape, dtype=bool)
    for _ in range(4):
        fresh = (mask & ~explained).astype(np.float32)
        centres, score = _match(image, dilated, fresh, cls, settings)
        if found.size and centres.size:
            near = np.linalg.norm(centres[:, None] - found[None], axis=2).min(axis=1)
            new = near >= 0.35 * min(th, tw)
            centres, score = centres[new], score[new]
        if centres.size == 0:
            break
        found = np.vstack([found, centres])
        scores = np.concatenate([scores, score])
        explained |= _stamp(mask.shape, cls, centres, "template")
    cls.centres = found
    cls.scores = scores


def _resolve_overlaps(classes: list[_MarkerClass]) -> None:
    """Where two shapes matched the same marker, the better match keeps it."""
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            ca, cb = classes[a], classes[b]
            if ca.centres.size == 0 or cb.centres.size == 0:
                continue
            # Half the larger glyph: a small marker's template also fits inside a larger
            # one - a small square in each corner of a big square - and those fits are
            # the big marker, not four small ones.
            reach = 0.5 * max(max(ca.size), max(cb.size))
            d = np.linalg.norm(ca.centres[:, None] - cb.centres[None], axis=2)
            clash = d < reach
            if not clash.any():
                continue
            drop_a = np.zeros(len(ca.centres), bool)
            drop_b = np.zeros(len(cb.centres), bool)
            for i, j in zip(*np.nonzero(clash)):
                if ca.scores[i] >= cb.scores[j]:
                    drop_b[j] = True
                else:
                    drop_a[i] = True
            ca.centres, ca.scores = ca.centres[~drop_a], ca.scores[~drop_a]
            cb.centres, cb.scores = cb.centres[~drop_b], cb.scores[~drop_b]


def _stamp(shape: tuple[int, int], cls: _MarkerClass, centres: np.ndarray,
           which: str = "hull", grow: int = 0) -> np.ndarray:
    """A mask with the marker drawn at each centre."""
    out = np.zeros(shape, dtype=bool)
    glyph = (cls.hull if which == "hull" else cls.template).astype(bool)
    if grow:
        # Grown within the glyph's own box: that fills the gaps sub-pixel placement
        # leaves around the outline, without reaching out along a line that touches it.
        glyph = cv2.dilate(glyph.astype(np.uint8),
                           np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
        pad = grow
    else:
        pad = 0
    oy, ox = cls.anchor
    oy, ox = oy + pad, ox + pad
    gh, gw = glyph.shape
    for cx, cy in centres:
        r0, c0 = int(round(cy - oy)), int(round(cx - ox))
        r1, c1 = r0 + gh, c0 + gw
        sr0, sc0 = max(0, -r0), max(0, -c0)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(shape[0], r1), min(shape[1], c1)
        if r1 > r0 and c1 > c0:
            out[r0:r1, c0:c1] |= glyph[sr0:sr0 + (r1 - r0), sc0:sc0 + (c1 - c0)]
    return out


# -- stroke styles -----------------------------------------------------------------


def _style_masks(strokes: np.ndarray, thickness: float,
                 cut: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Sort stroke components into dots, dashes and continuous lines.

    Dashes are recognised by being *many pieces of one length*. A solid line cut short
    by markers, or clipped at the frame, also leaves short pieces - but of assorted
    lengths, so it does not produce the tight cluster a dash pattern does.

    ``cut`` marks where markers were lifted out. A short piece touching it was cut
    there, so its length says nothing about its style: it is left out of every style,
    to be claimed later by whichever line turns out to pass through it.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        strokes.astype(np.uint8), 8)
    if count <= 1:
        return {}
    kinds = np.zeros(count, dtype="<U11")
    lengths = np.zeros(count)
    dot_limit = 2.4 * thickness + 1.0
    touches = np.zeros(count, dtype=bool)
    if cut is not None and cut.any():
        touches[np.unique(labels[cut & (labels > 0)])] = True
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        # A piece that touches where a marker was lifted out may be a scrap of that
        # marker as easily as a piece of line, dot or not: its style is left open.
        if touches[i] and max(w, h) <= 20 * thickness:
            kinds[i] = "cut"
            continue
        if max(w, h) <= 3:
            kinds[i] = "dotted"
            continue
        own = np.column_stack(np.nonzero(labels[y:y + h, x:x + w] == i)).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(own)
        extent = max(rw, rh) + 1.0
        lengths[i] = max(rw, rh, area / max(thickness, 1.0))
        # A dot is about as long as it is wide - its own width, not the colour's
        # typical stroke, since a thin dashed line can share a colour with a heavy one.
        across = max(1.0, area / extent)
        compact = extent <= 2.0 * across + 1.0
        dot = ((compact and max(w, h) <= 2 * dot_limit)
               or (max(w, h) <= dot_limit and not extent >= 2.2 * across + 1.0))
        kinds[i] = "dotted" if dot else "solid"

    short = [i for i in range(1, count) if kinds[i] == "solid" and lengths[i] <= 20 * thickness]
    # A short piece of a line points along the line: its long axis aims at the next
    # piece. The arc left over from a marker that was only partly matched points
    # anywhere. Pieces that do not line up with a neighbour are neither dash nor line.
    aligned = _aligned_pieces(labels, stats, centroids, short)
    for i in short:
        if i not in aligned:
            kinds[i] = "debris"
    short = [i for i in short if i in aligned]
    # Each dash pattern is many pieces of one length. A figure can hold two - a dashed
    # and a dash-dot line, or a dashed and a long-dashed one - so the patterns are taken
    # one at a time: the length most pieces share, then the next among what is left.
    remaining = list(short)
    patterns: list[tuple[float, list[int]]] = []
    while len(remaining) >= 6:
        values = lengths[remaining]
        centre = max(np.unique(values),
                     key=lambda c: int(np.sum((values >= 0.75 * c) & (values <= 1.33 * c))))
        members = [i for i in remaining if 0.75 * centre <= lengths[i] <= 1.33 * centre]
        if len(members) < 6:
            break
        typical = float(np.median(lengths[members]))
        members = [i for i in remaining if lengths[i] <= 1.6 * typical
                   and lengths[i] >= 0.6 * typical]
        patterns.append((typical, members))
        remaining = [i for i in remaining if i not in members]
    # The shortest is "dashed"; a longer one is a style of its own, so that the tracker
    # never confuses its dashes with the first one's where the two lines cross.
    for rank, (_, members) in enumerate(sorted(patterns)):
        for i in members:
            kinds[i] = "dashed" if rank == 0 else "long-dashed"

    out: dict[str, np.ndarray] = {}
    for style in ("solid", "dashed", "long-dashed", "dotted"):
        chosen = np.flatnonzero(kinds == style)
        if chosen.size:
            out[style] = np.isin(labels, chosen)
    return out


def _aligned_pieces(labels: np.ndarray, stats: np.ndarray, centroids: np.ndarray,
                    pieces: list[int]) -> set[int]:
    """The pieces that line up with a neighbour the way the pieces of a line do.

    Two dashes of one line point along the line that joins them, both of them, and sit
    about a dash-length apart. Each test on its own is passed by chance surprisingly
    often - a randomly turned arc points at one of its two nearest neighbours more
    than half the time - so a pair has to pass all three.
    """
    if len(pieces) < 2:
        return set(pieces)
    axes, lengths = {}, {}
    for i in pieces:
        x, y, w, h, _ = stats[i]
        rows, cols = np.nonzero(labels[y:y + h, x:x + w] == i)
        points = np.column_stack([cols, rows]).astype(float)
        points -= points.mean(axis=0)
        # Principal direction of the piece's own pixels, and its extent along it.
        _, _, vt = np.linalg.svd(points, full_matrices=False)
        axes[i] = vt[0]
        along = points @ vt[0]
        lengths[i] = float(along.max() - along.min() + 1)
    centres = centroids[pieces]
    distance = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    aligned = set()
    for a, i in enumerate(pieces):
        # The nearest pieces are usually the next dash along; but where lines in one
        # style run close together, the nearest can be on the lines either side.
        for b in np.argsort(distance[a])[:4]:
            if not np.isfinite(distance[a, b]):
                continue
            j = pieces[b]
            towards = centres[b] - centres[a]
            towards /= max(1e-9, np.linalg.norm(towards))
            gap = distance[a, b] - 0.5 * (lengths[i] + lengths[j])
            # Within about 30 degrees, either way along each axis, and close.
            if (abs(float(towards @ axes[i])) >= 0.85 and abs(float(towards @ axes[j])) >= 0.85
                    and gap <= 1.5 * max(lengths[i], lengths[j])):
                aligned.update((i, j))
                break
    return aligned


# -- tracking ----------------------------------------------------------------------


@dataclass(eq=False)
class _Track:
    columns: list[int]
    rows: list[float]
    #: column -> (row, first ink row, last ink row, shared). ``shared`` is True where
    #: another track held the same ink, "wide" where the ink was clipped to the stroke.
    observed: dict[int, tuple[float, int, int, bool | str]] = field(default_factory=dict)
    misses: int = 0
    alive: bool = True
    #: The track's own ink - not shared with another curve - in the order it was walked,
    #: so the latest are the nearest.
    clean: list[tuple[int, float]] = field(default_factory=list)
    #: How many columns in a row the track has been on ink wider than its stroke.
    wide: int = 0

    def __post_init__(self) -> None:
        if not self.clean:
            self.clean = [(c, self.observed[c][0]) for c in self.columns
                          if c in self.observed and not self.observed[c][3]]

    def predict(self, column: int, window: int = 8, steady: int = 24) -> tuple[float, float]:
        """Where the curve goes next, and its slope, from the last few columns.

        A few columns follow a bend closely. But where the track is sharing merged ink
        with another curve, or coasting without any, those columns say little: at a
        crossing the merged ink is centred between the two curves, and a rasterised
        shallow line is a staircase whose steps are longer than a few columns. There
        the slope comes from a longer stretch of the curve's own ink before it.
        """
        cols = np.asarray(self.columns[-window:], dtype=float)
        rows = np.asarray(self.rows[-window:], dtype=float)
        if cols.size < 2 or np.ptp(cols) < 1e-9:
            return float(rows[-1]), 0.0
        last = self.observed.get(self.columns[-1])
        # Ink wider than the stroke is usually a marker the line runs through, and it is
        # crossed on the slope from before. Wide ink that goes on and on - a band of
        # crowded markers the line runs along - is the way the line now goes.
        brief = False
        if self.misses or (last is not None and last[3] is True) or brief:
            own = self.clean[-steady:]
            if len(own) >= 4:
                xs = np.fromiter((c for c, _ in own), dtype=float, count=len(own))
                ys = np.fromiter((r for _, r in own), dtype=float, count=len(own))
                if np.ptp(xs) >= 3:
                    slope = _fit(xs, ys)[0]
                    return float(rows[-1] + slope * (column - cols[-1])), slope
        slope = _fit(cols, rows)[0]
        return float(rows[-1] + slope * (column - cols[-1])), slope


def _fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Least-squares straight line through points: (slope, intercept).

    np.polyfit does the same through a general solver, which in the tracker's inner loop
    - hundreds of thousands of two-parameter fits - costs far more than the fit itself.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mx, my = xs.mean(), ys.mean()
    dx = xs - mx
    spread = float(dx @ dx)
    slope = float(dx @ (ys - my)) / spread if spread > 1e-12 else 0.0
    return slope, float(my - slope * mx)


def _gap_between_pieces(mask: np.ndarray, thickness: float) -> int:
    """Typical gap between the pieces of a broken line, to bridge when counting."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 3:
        return 1
    boxes = stats[1:, :4].astype(float)
    centres = centroids[1:]
    if centres.shape[0] > 3000:
        return 2
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    size = np.maximum(boxes[:, 2], boxes[:, 3])
    gaps = np.maximum(0.0, nearest - size)
    return int(np.clip(np.ceil(np.median(gaps)) + 1, 1, 12))


def _stable_stretch(counts: np.ndarray) -> tuple[int, int] | None:
    """Where to start tracking: a long run of columns with a constant stroke count.

    More strokes is better - every curve present at once - but only over a stretch
    long enough to trust. A dash fused onto a line, or a label, adds a stroke for a
    dozen columns; starting there would set a track on the stub. So the count only
    wins among stretches at least a twentieth of the inked width.
    """
    inked = np.flatnonzero(counts)
    if inked.size == 0:
        return None
    reliable = max(10, int(0.05 * (inked[-1] - inked[0] + 1)))
    best = None
    start = 0
    for column in range(1, counts.size + 1):
        if column == counts.size or counts[column] != counts[start]:
            length = column - start
            value = int(counts[start])
            if value > 0:
                trusted = length >= reliable
                key = (trusted, value if trusted else 0, length)
                if best is None or key > best[0]:
                    best = (key, start, column)
            start = column
    return None if best is None else (best[1], best[2])


def _walk(tracks: list[_Track], runs: list[np.ndarray], order: range,
          thickness: float, pool: int, settings: SeparationSettings,
          births: list[_Track] | None, holes: np.ndarray | None = None) -> None:
    """Advance every track through ``order``, one column at a time, together.

    ``holes`` marks small patches of paper that ink closes round: the inside of a hollow
    marker with a white face, which hides the line drawn through it.
    """
    base_gate = max(3.0, 1.5 * thickness) + pool
    # A track coasts across a hollow marker drawn over its line, and at high resolution
    # a marker is wider than the columns that would do at 100 dpi.
    coast = max(settings.max_gap, int(round(6 * thickness)))
    tentative: list[_Track] = []

    for column in order:
        spans = runs[column]
        live = [t for t in tracks if t.alive]
        predictions = [t.predict(column) for t in live]
        candidates = []
        enclosing: set[int] = set()
        for ti, (track, (row, _)) in enumerate(zip(live, predictions)):
            gate = base_gate + 0.5 * track.misses
            above = [(row - (end - 1), ri) for ri, (start, end) in enumerate(spans)
                     if end - 1 < row and row - (end - 1) <= gate]
            below = [(start - row, ri) for ri, (start, end) in enumerate(spans)
                     if start > row and start - row <= gate]
            inside = (holes is not None and 0 <= int(round(row)) < holes.shape[0]
                      and holes[int(round(row)), column])
            if inside:
                # The prediction is on paper that ink closes round: the inside of a
                # hollow marker with a white face, hiding the line drawn through it.
                # Neither side is the line, however far off centre the line crosses.
                ahead = [(row - (end - 1), ri) for ri, (start, end) in enumerate(spans)
                         if end - 1 < row]
                behind = [(start - row, ri) for ri, (start, end) in enumerate(spans)
                          if start > row]
                enclosing.update(ri for _, ri in (min(ahead, default=(0, None)),
                                                  min(behind, default=(0, None)))
                                 if ri is not None)
                continue
            if above and below and abs(min(above)[0] - min(below)[0]) <= thickness:
                # The prediction falls midway in a gap with ink on both sides: inside a
                # hollow marker drawn over the line, whose outline is equally far above
                # and below where the line would be. Neither side is the line, and
                # taking either walks the track off round the marker. Coast through.
                # (Off-centre, the nearer side is simply where the line has gone.)
                # Those two runs are the marker, accounted for: they must not start
                # tracks of their own, which would follow the outline round and take
                # the line over on the far side.
                enclosing.update((min(above)[1], min(below)[1]))
                continue
            for ri, (start, end) in enumerate(spans):
                lo, hi = start, end - 1
                distance = 0.0 if lo <= row <= hi else min(abs(row - lo), abs(row - hi))
                if distance <= gate:
                    # Tracks that had ink in the last column choose first.
                    candidates.append((track.misses > 1, distance, ti, ri))
        candidates.sort()

        owner: dict[int, int] = {}
        sharing: dict[int, list[int]] = {}
        for lost, _, ti, ri in candidates:
            if ti in owner:
                continue
            start, end = spans[ri]
            if ri in sharing and lost:
                # A track that has lost its own ink and is coasting on its prediction
                # may pick ink up again, but not ink another track already holds -
                # otherwise it drifts onto the nearest other curve and follows it as
                # if that were where its own had gone.
                continue
            if ri in sharing:
                # A second track may share ink only if the ink has room for it: either
                # it predicts itself inside, or the run is long enough to hold a second
                # stroke. That is two curves meeting, not two tracks fighting over one.
                row, slope = predictions[ti]
                expected = thickness + abs(slope) + 2 * pool + 1
                inside = start - 0.5 * thickness <= row <= end - 1 + 0.5 * thickness
                roomy = end - start >= 1.5 * expected
                if not (inside or roomy):
                    continue
            owner[ti] = ri
            sharing.setdefault(ri, []).append(ti)

        for ti, track in enumerate(live):
            row, slope = predictions[ti]
            if ti not in owner:
                track.misses += 1
                track.columns.append(column)
                track.rows.append(row)
                if track.misses > coast:
                    track.alive = False
                continue
            start, end = spans[owner[ti]]
            shared = len(sharing[owner[ti]]) > 1
            expected = thickness + abs(slope) + 2 * pool + 1
            if shared:
                # Two curves in one run of ink. Each track keeps its own prediction,
                # held only loosely to the ink. Pinning both to the run - which at a
                # crossing is barely wider than one stroke - would give them the same
                # history, hence the same slope, and they would leave the crossing
                # together on whichever curve happened to be nearer.
                half = 0.5 * expected
                centre = float(np.clip(row, start - thickness, end - 1 + thickness))
                lo, hi = int(np.floor(centre - half - 1)), int(np.ceil(centre + half + 1))
            elif end - start > thickness + abs(slope) + 2 * pool + 1.5:
                # More ink than one stroke makes: another curve has merged into this
                # one, or it runs into something upright. The midpoint of all that ink
                # belongs to neither - taking it bends the track towards the other
                # curve and it leaves the crossing on the wrong one. Keep to the part
                # of the run nearest the prediction instead. Even a pixel or two of
                # extra ink matters: a midpoint off by that much, column after column,
                # flattens the slope until the track follows the intruder. Where the
                # run is only just longer than the stroke the clip leaves the centre
                # almost where the midpoint would have been, so little is lost when
                # the extra ink was only rendering noise. The margin is the stroke's
                # own half-width: the stroke can lie right at the edge of the merged
                # ink, and a wider margin would hold the track off it and towards the
                # other stroke.
                half = 0.5 * expected
                edge = 0.5 * thickness
                centre = float(np.clip(row, start + edge - 0.5,
                                       max(start + edge - 0.5, end - 0.5 - edge)))
                lo, hi = int(np.floor(centre - half - 1)), int(np.ceil(centre + half + 1))
                # Not shared with another track, only clipped from wider ink: no reason
                # to stop following the curve's own bends, but no clean reading either.
                shared = "wide"
            else:
                lo, hi = start, end - 1
                centre = 0.5 * (start + end - 1)
            track.misses = 0
            track.columns.append(column)
            track.rows.append(centre)
            track.observed[column] = (centre, max(lo, start), min(hi, end - 1), shared)
            track.wide = track.wide + 1 if shared == "wide" else 0
            if not shared:
                track.clean.append((column, centre))

        if births is None:
            continue
        # Strokes nobody claimed may be a curve that starts here. Follow them for a
        # while before believing it; a speck of text or a legend swatch soon ends.
        claimed = set(owner.values()) | enclosing
        free = [ri for ri in range(len(spans)) if ri not in claimed
                and spans[ri][1] - spans[ri][0] <= 4 * thickness + 2 * pool]
        used = set()
        for cand in tentative:
            row, _ = cand.predict(column)
            best, best_d = None, base_gate
            for ri in free:
                if ri in used:
                    continue
                start, end = spans[ri]
                d = 0.0 if start <= row <= end - 1 else min(abs(row - start), abs(row - end + 1))
                if d <= best_d:
                    best, best_d = ri, d
            if best is None:
                cand.misses += 1
                if cand.misses > 3:
                    cand.alive = False
                continue
            used.add(best)
            start, end = spans[best]
            centre = 0.5 * (start + end - 1)
            cand.misses = 0
            cand.columns.append(column)
            cand.rows.append(centre)
            cand.observed[column] = (centre, int(start), int(end - 1), False)
            cand.clean.append((column, centre))
        for ri in free:
            if ri not in used:
                start, end = spans[ri]
                centre = 0.5 * (start + end - 1)
                tentative.append(_Track([column], [centre],
                                        {column: (centre, int(start), int(end - 1), False)}))
        promoted = [c for c in tentative if c.alive and len(c.observed) >= settings.birth_length]
        for cand in promoted:
            tracks.append(cand)
            births.append(cand)
        tentative = [c for c in tentative if c.alive and c not in promoted]
        # A few curves start a few tracks. Ink that keeps starting them - hundreds at a
        # time - is a cloud of marks, and following it further only costs time.
        if len(births) > 60 or len(tentative) > 200:
            raise _Cloud


def _small_holes(mask: np.ndarray, largest: int) -> np.ndarray:
    """Patches of paper, no wider or taller than ``largest``, that ink closes round."""
    closed = _fill_holes(mask.astype(np.uint8)).astype(bool) & ~mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), 8)
    small = (stats[:, cv2.CC_STAT_WIDTH] <= largest) & (stats[:, cv2.CC_STAT_HEIGHT] <= largest)
    small[0] = False
    return small[labels]


class _Cloud(Exception):
    """Ink followed as lines keeps starting new ones: a dense cloud, not a few curves."""


def _track_style(mask: np.ndarray, thickness: float, settings: SeparationSettings,
                 min_length: int, broken: bool = True) -> list[_Track]:
    """Follow every curve drawn in one style.

    ``broken`` says the style is made of separate pieces - dashes or dots - whose gaps
    need bridging. A solid line has none; bridging it anyway only fuses it with
    whatever happens to lie close by.
    """
    gap = min(_gap_between_pieces(mask, thickness), 8) if broken and mask.any() else 1
    # Bridge the gaps of a dashed or dotted line so that each curve is present in
    # (nearly) every column; the raw ink is still what gets recorded. The bridge has to
    # be round. Bridging sideways only works where the line is flat: on a steep stretch
    # the gap between two dashes is mostly vertical, so a column would see the end of
    # one dash and the start of the next as two separate strokes - two curves where
    # there is one. (A closing would not do either: the gap between two dots is open
    # to the paper on both sides, and a closing only fills gaps a disk cannot get into.)
    pool = 0 if gap <= 1 else int(np.ceil(gap / 2))
    if pool:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pool + 1, 2 * pool + 1))
        pooled = cv2.dilate(mask.astype(np.uint8), disk).astype(bool)
    else:
        pooled = mask
    runs = _column_runs(pooled)
    holes = _small_holes(pooled, max(12, int(round(6 * thickness))))
    counts = np.array([len(r) for r in runs])
    # Each curve crosses a column once. Ink crossing the typical column a dozen and more
    # times over is a cloud of marks, not a few lines to follow.
    if counts.any() and np.median(counts[counts > 0]) > 12:
        raise _Cloud
    stretch = _stable_stretch(counts)
    if stretch is None:
        return []
    start, end = stretch

    # Inside a stretch where the count never changes nothing crosses, so the i-th stroke
    # from the top is the same curve in every column - a safe place to start.
    k = int(counts[start])
    tracks = [_Track([], []) for _ in range(k)]
    for column in range(start, end):
        for i, (s, e) in enumerate(runs[column][:k]):
            centre = 0.5 * (s + e - 1)
            tracks[i].columns.append(column)
            tracks[i].rows.append(centre)
            tracks[i].observed[column] = (centre, int(s), int(e - 1), False)

    right = [_Track(list(t.columns), list(t.rows), t.observed) for t in tracks]
    born: list[_Track] = []
    _walk(right, runs, range(end, mask.shape[1]), thickness, pool, settings, born, holes)
    if len(born) > 60:
        raise _Cloud
    left = [_Track(list(reversed(t.columns)), list(reversed(t.rows)), t.observed)
            for t in right[:k]]
    born_left: list[_Track] = []
    _walk(left, runs, range(start - 1, -1, -1), thickness, pool, settings, born_left, holes)
    if len(born) + len(born_left) > 60:
        raise _Cloud

    # A curve that first appeared on the way right may extend further left than where
    # it was noticed, and vice versa; follow each newcomer back the other way.
    for track in born + born_left:
        columns = sorted(track.observed)
        noticed = set(columns)
        if track in born:
            back = _Track(columns[:8][::-1], [track.observed[c][0] for c in columns[:8][::-1]],
                          track.observed)
            _walk([back], runs, range(columns[0] - 1, -1, -1), thickness, pool, settings, None,
                  holes)
        else:
            ahead = _Track(columns[-8:], [track.observed[c][0] for c in columns[-8:]],
                           track.observed)
            _walk([ahead], runs, range(columns[-1] + 1, mask.shape[1]), thickness, pool,
                  settings, None, holes)
        # Walking alone, it will follow any ink - including another curve's, all the
        # way back. Crossing another curve is fine and brief; running along one for a
        # long stretch means it has been captured, so it is cut off where that began.
        others = [t for t in right[:k] + born + born_left if t is not track]
        _release_capture(track, noticed, others, max(2.0, thickness), backwards=track in born)
    everything = right[:k] + born + born_left
    # Keep only what was actually seen in the raw ink, and only curves long enough to
    # be data rather than a label or a legend swatch. Length is the span covered, not
    # the columns inked: a dotted line is mostly gaps, and is no shorter for it.
    kept = []
    for track in everything:
        seen = {c: v for c, v in track.observed.items()
                if mask[max(0, v[1]):v[2] + 1, c].any()}
        if len(seen) >= 6 and max(seen) - min(seen) + 1 >= min_length:
            track.observed = seen
            kept.append(track)
    return _drop_duplicates(kept, thickness)


def _release_capture(track: _Track, noticed: set[int], others: list[_Track],
                     tolerance: float, backwards: bool, stretch: int = 20) -> None:
    """Cut an extension where it starts running along another track."""
    extension = sorted((c for c in track.observed if c not in noticed), reverse=backwards)
    run = 0
    for index, column in enumerate(extension):
        row = track.observed[column][0]
        on_other = any(column in o.observed and abs(o.observed[column][0] - row) <= tolerance
                       for o in others)
        run = run + 1 if on_other else 0
        if run >= stretch:
            for dropped in extension[index - run + 1:]:
                del track.observed[dropped]
            return


def _drop_duplicates(tracks: list[_Track], thickness: float) -> list[_Track]:
    """Two tracks following the same stroke are one; keep the longer."""
    tracks = sorted(tracks, key=lambda t: len(t.observed), reverse=True)
    kept: list[_Track] = []
    for track in tracks:
        duplicate = False
        for other in kept:
            common = set(track.observed) & set(other.observed)
            if not common:
                continue
            columns = np.array(sorted(common), dtype=float)
            mine = np.array([track.observed[int(c)][0] for c in columns])
            theirs = np.array([other.observed[int(c)][0] for c in columns])
            close = int(np.sum(_across(mine - theirs, _centreline(other), columns)
                               <= max(2.0, thickness)))
            if close >= 0.6 * len(track.observed):
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
    return kept


def _across(rows: np.ndarray, line: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Vertical offsets from a centreline turned into distances across the stroke.

    On a steep stretch two tracks on one stroke can be several rows apart in the same
    column while barely a pixel apart across it.
    """
    if line.shape[0] < 2:
        return np.abs(rows)
    slope = np.gradient(line[:, 1], line[:, 0]) if np.all(np.diff(line[:, 0]) > 0) \
        else np.zeros(line.shape[0])
    local = np.interp(columns, line[:, 0], slope)
    return np.abs(rows) / np.sqrt(1.0 + local ** 2)


def _centreline(track: _Track) -> np.ndarray:
    columns = np.array(sorted(track.observed), dtype=float)
    rows = np.array([track.observed[int(c)][0] for c in columns])
    return np.column_stack([columns, rows])


def _track_mask(shape: tuple[int, int], raw: np.ndarray, track: _Track) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for column, (_, lo, hi, _) in track.observed.items():
        lo, hi = max(0, lo), min(shape[0] - 1, hi)
        if hi >= lo:
            out[lo:hi + 1, column] = raw[lo:hi + 1, column]
    return out


def _row_at(line: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """A centreline's row at arbitrary columns, NaN outside the columns it spans."""
    rows = np.interp(columns, line[:, 0], line[:, 1])
    rows[(columns < line[0, 0]) | (columns > line[-1, 0])] = np.nan
    return rows


def _colocated(a: np.ndarray, b: np.ndarray, tolerance: float) -> float:
    """Fraction of their common span over which two centrelines coincide."""
    lo, hi = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
    if hi - lo < 5:
        return 0.0
    columns = np.arange(np.ceil(lo), np.floor(hi) + 1)
    return float(np.mean(np.abs(_row_at(a, columns) - _row_at(b, columns)) <= tolerance))


# -- entry point -------------------------------------------------------------------


def separate_series(mask: np.ndarray, plot_width: int | None = None,
                    settings: SeparationSettings | None = None,
                    foreign: np.ndarray | None = None,
                    ink: np.ndarray | None = None) -> list[Part] | None:
    """Split one colour's ink into its series, or ``None`` if it is only one.

    ``foreign`` is the ink of the figure's other colours. Where it lies against this
    colour it can hide part of a marker or leave its own anti-aliased edge in this
    colour, so ink right next to it is not trusted to say what shape a marker is, nor
    followed as a line.

    ``ink`` is all the ink in the plot, whatever its colour. A thin upright stroke - the
    drop of a sawtooth - can be too pale to be clustered with its own colour at all,
    and it is only there that the join it makes can be seen.
    """
    settings = settings or SeparationSettings()
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 30:
        return None
    if foreign is not None and foreign.any():
        beside = cv2.dilate(np.asarray(foreign, dtype=np.uint8),
                            np.ones((5, 5), np.uint8)).astype(bool)
    else:
        beside = np.zeros(mask.shape, dtype=bool)
    shape = mask.shape
    columns = np.flatnonzero(mask.any(axis=0))
    width = plot_width or int(columns[-1] - columns[0] + 1)
    min_length = max(12, int(settings.min_track_fraction * width))
    # Pieces are kept a good deal shorter than a series has to be, because a line
    # broken where it crosses markers or another curve is only reassembled from
    # its pieces afterwards; judging each piece on its own would discard them.
    min_piece = max(12, int(0.05 * width))
    # 0. Lettering in the plot is neither markers nor lines; it takes no further part.
    mask = mask & ~_text(mask)
    if mask.sum() < 30:
        return None
    thickness = stroke_thickness(mask)
    # Error bars are set aside too, so that a marker is matched by its own shape; they
    # go back to their markers at the end.
    bars = _error_bars(mask, thickness)
    glyphs = mask & ~bars

    # 1. Markers, found wherever they are and lifted out of the ink.
    classes = _glyph_classes(glyphs, thickness, settings, beside)
    for cls in classes:
        _locate(glyphs, cls, settings)
    _resolve_overlaps(classes)
    classes = [c for c in classes if len(c.centres) >= settings.min_markers]
    band_points: dict[int, list[np.ndarray]] = {}
    band_ink = np.zeros(shape, dtype=bool)
    for index, cls in enumerate(classes):
        # Only hollow markers: a band of filled ones is indistinguishable from a thick
        # line, and a thick line is far the commoner of the two.
        if cls.filled:
            continue
        ink, points = _marker_band(glyphs, cls)
        if points.size:
            band_ink |= ink
            band_points[index] = [points]
    removed = band_ink.copy()
    for cls in classes:
        # Two pixels of margin: sub-pixel placement makes some copies of a glyph a
        # pixel larger than the template.
        removed |= _stamp(shape, cls, cls.centres, "hull", grow=2)
    strokes = glyphs & ~removed & ~beside
    cut = cv2.dilate(removed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~removed

    # 2. The remaining strokes, by style, each style followed curve by curve.
    curves: list[Part] = []
    try:
        tracked = [(style, style_mask, track)
                   for style, style_mask in _style_masks(strokes, thickness, cut).items()
                   for track in _track_style(style_mask, thickness, settings, min_piece,
                                             broken=style != "solid")]
    except _Cloud:
        log.info("not splitting: the strokes keep starting new tracks, a cloud not lines")
        return None
    for style, style_mask, track in tracked:
        part_mask = _track_mask(shape, style_mask, track)
        pieces = cv2.connectedComponents(part_mask.astype(np.uint8), connectivity=8)[0] - 1
        curves.append(Part("curve", style, part_mask, centreline=_centreline(track),
                           tally={style: int(part_mask.sum())},
                           pieces={style: int(pieces)}))

    # A handful of series leaves a handful of pieces, even broken at every crossing. A
    # dense cloud followed as if it were lines leaves far more, none of them a series.
    if len(curves) > 80:
        log.info("not splitting: %d stroke pieces is a cloud, not lines", len(curves))
        return None

    # 3. Put lines back together: pieces of one style broken where another line
    #    crossed, and fragments that lie along a longer line for most of their length.
    joins = mask if ink is None else (np.asarray(ink, dtype=bool) | mask)
    curves = _stitch(curves, width, thickness, joins)
    curves = _absorb(curves, thickness, settings)
    curves = _join_overlaps(curves, thickness)
    curves = _stitch(curves, width, thickness, joins)
    curves = _fold_fragments(curves, min_length, thickness)
    curves = _join_turns(curves, mask, thickness, width)
    curves = _join_overlaps(curves, thickness)
    # A dotted or dashed line is its pieces at a steady spacing along a smooth path.
    # Specks of noise, the edge of a shaded band or a scatter of small dots can be
    # strung together too, but not evenly and not smoothly.
    curves = [c for c in curves if c.folded or _evenly_broken(c) is not False]
    # Only now, with the pieces joined up, is a line judged on its length.
    curves = [c for c in curves if np.ptp(c.centreline[:, 0]) + 1 >= min_length]
    # A colour made of dead level lines across the plot is a grid - dashed, or broken
    # up by compression, so the frame detection did not recognise it - not data.
    level = [c for c in curves if np.ptp(c.centreline[:, 0]) >= 0.8 * width
             and np.subtract(*np.percentile(c.centreline[:, 1], [95, 5])) <= max(1.5, thickness)]
    if len(level) >= 2 and len(level) >= 0.5 * len(curves):
        return None

    # 4. Markers so crowded that they merged into a band never matched one at a time,
    #    and the band was followed as if it were a line. Give it back to its markers.
    if classes:
        near = np.zeros(shape, dtype=bool)
        for cls in classes:
            near |= _stamp(shape, cls, cls.centres, "hull", grow=max(2, int(round(thickness))))
        near |= band_ink
        # Scraps left around markers that were found - the corner of one poking out
        # from under another - line up well enough to be followed, but they are those
        # markers, already counted.
        curves = [c for c in curves
                  if float((c.mask & near).sum()) < 0.5 * float(c.mask.sum())]

    # 5. Close each line's gaps with ink nobody else claimed that lies on its path -
    #    where a marker sat on it, or a piece fused with a different style - and read
    #    its points off what it finally owns.
    marker_ink = band_ink.copy()
    for cls in classes:
        marker_ink |= _stamp(shape, cls, cls.centres, "template", grow=1)
    claimed = np.zeros(shape, dtype=bool)
    for curve in curves:
        claimed |= curve.mask
    free = glyphs & ~claimed & ~marker_ink
    for curve in curves:
        if curve.folded:
            continue
        _fill_gaps(curve, free, thickness)
        curve.points = _curve_points(curve, thickness)

    # 6. Markers drawn on a line belong to it; markers on no line are their own series.
    #    Whether a marker shape is a line's is decided for the shape as a whole - a
    #    scatter about a fit line has some of its points on the line by chance - and
    #    then each marker goes to the line it sits on.
    parts: list[Part] = list(curves)
    families: dict[int, list[int]] = {}
    for index, cls in enumerate(classes):
        families.setdefault(cls.family, []).append(index)
    owned = [dict() for _ in curves]
    for family, members in families.items():
        banded = any(index in band_points for index in members)
        owners = {index: np.array([_owner(curves, centre, classes[index], thickness, settings)
                                   for centre in classes[index].centres], dtype=int)
                  for index in members}
        total = sum(len(o) for o in owners.values())
        on = sum(int((o >= 0).sum()) for o in owners.values())
        free_centres, glyph = [], np.zeros(shape, dtype=bool)
        line_markers = bool(curves) and not banded and total and on >= settings.on_line_fraction * total
        for index in members:
            cls = classes[index]
            here = owners[index].copy()
            if not line_markers:
                # A scatter's template can also fit a different marker drawn on a line,
                # well enough to pass but clearly worse than it fits its own. Those
                # stay with their lines rather than join the scatter.
                here[cls.scores >= 0.7] = -1
            for j in set(here[here >= 0].tolist()):
                curve = curves[j]
                curve.mask |= _stamp(shape, cls, cls.centres[here == j], "template") & mask
                # Only a close fit names the line's marker: the template of one shape
                # matches other filled shapes loosely.
                good = int(((here == j) & (cls.scores >= 0.75)).sum())
                owned[j][family] = owned[j].get(family, 0) + good
            free = here < 0
            free_centres.append(cls.centres[free])
            glyph |= _stamp(shape, cls, cls.centres[free], "template") & mask
            glyph |= _bars_of(bars, _stamp(shape, cls, cls.centres[free], "hull", grow=2))
        points = np.vstack(free_centres) if free_centres else np.empty((0, 2))
        if line_markers and len(points) < settings.min_markers:
            # A few of a line's markers just off its track: not a series of their own.
            continue
        for index in members:
            if index in band_points and len(band_points[index]):
                cls = classes[index]
                band = np.vstack(band_points[index])
                if len(points):
                    # Where the band's markers were matched one by one, those matches
                    # are the answer; the band's own points only fill where they were not.
                    near = np.abs(band[:, None, 0] - points[None, :, 0]).min(axis=1)
                    band = band[near > 0.5 * cls.template.shape[1]]
                points = np.vstack([points, band])
        if len(points):
            points = points[np.argsort(points[:, 0], kind="stable")]
            parts.append(Part("markers", "markers", glyph,
                              marker=classes[members[0]].describe(), points=points))
    for curve, counts in zip(curves, owned):
        if counts:
            family, count = max(counts.items(), key=lambda item: item[1])
            if count >= 2:
                curve.marker = classes[families[family][0]].describe()

    parts = [p for p in parts if p.mask.sum() >= 12 or (p.points is not None and len(p.points))]
    if len(parts) <= 1:
        return None

    # 7. Account for the ink. Whatever the parts do not explain was never classified -
    #    a series the split could not see, or the stretch of one it lost. When that is
    #    much of the colour, the split is not to be trusted and the colour stays whole;
    #    when it is a little, the caller is told where.
    explained = bars | band_ink | marker_ink
    for part in parts:
        explained |= part.mask
    explained = cv2.dilate(explained.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    leftover = mask & ~explained
    count, labels, stats, _ = cv2.connectedComponentsWithStats(leftover.astype(np.uint8), 8)
    sizeable = stats[:, cv2.CC_STAT_AREA] >= max(8.0, 2.0 * thickness ** 2)
    sizeable[0] = False
    leftover = sizeable[labels]
    share = float(leftover.sum()) / max(1.0, float(mask.sum()))
    if share > settings.max_unexplained:
        log.info("not splitting: %.0f%% of the ink fits none of %d series", 100 * share,
                 len(parts))
        return None
    result = Separation(parts)
    if share > 0.1:
        columns = np.flatnonzero(leftover.any(axis=0))
        result.unassigned = (int(leftover.sum()), int(columns[0]), int(columns[-1]))
    log.info("split one colour into %d series: %s", len(parts),
             ", ".join(p.description for p in parts))
    return result


def _owner(curves: list[Part], centre: np.ndarray, cls: _MarkerClass, thickness: float,
           settings: SeparationSettings) -> int:
    """The line a marker is drawn on, or -1.

    The line has no ink of its own under the marker, so where it would be is judged from
    the stretches either side. Data drawn with markers usually changes direction at each
    marker, so each side is extended on its own and the nearer counts; a chord straight
    across would cut the corner. The end markers have a line on one side only.
    """
    cx, cy = float(centre[0]), float(centre[1])
    size = float(max(cls.size))
    tolerance = max(settings.on_line_px, 0.3 * float(min(cls.size)), 1.5 * thickness)
    best, owner = tolerance, -1
    for j, curve in enumerate(curves):
        if curve.folded:
            continue
        line = curve.centreline
        guesses = []
        for side in (-1, 1):
            offset = (line[:, 0] - cx) * side
            near = line[(offset >= 0.5 * size) & (offset <= 0.5 * size + max(8.0, 2.0 * size))]
            if near.shape[0] >= 3 and np.ptp(near[:, 0]) >= 2:
                slope, intercept = _fit(near[:, 0], near[:, 1])
                guesses.append(abs(slope * cx + intercept - cy) / np.sqrt(1 + slope ** 2))
        if not guesses:
            continue
        distance = min(guesses)
        if distance <= best:
            best, owner = distance, j
    return owner


# -- reassembly --------------------------------------------------------------------


def _combine(home: Part, other: Part) -> None:
    home.mask |= other.mask
    for style, count in other.tally.items():
        home.tally[style] = home.tally.get(style, 0) + count
    for style, count in other.pieces.items():
        home.pieces[style] = home.pieces.get(style, 0) + count
    home.centreline = _one_row_per_column(np.vstack([home.centreline, other.centreline]))
    home.style = _style_of(home)


def _style_of(part: Part) -> str:
    """Name a line assembled from pieces of several styles by what it is mostly made of.

    Dashes with dots between them are a dash-dot line: a good share of the ink in each,
    and about as many dots as dashes. A dashed line whose dashes broke into a few specks
    at a crossing has dotted ink too, but only a handful of pieces of it.
    """
    total = sum(part.tally.values()) or 1
    shares = {k: v / total for k, v in part.tally.items()}
    dashes = part.pieces.get("dashed", 0) + part.pieces.get("long-dashed", 0)
    dashed = shares.get("dashed", 0) + shares.get("long-dashed", 0)
    if dashed >= 0.15 and shares.get("dotted", 0) >= 0.1 \
            and part.pieces.get("dotted", 0) >= 0.6 * max(1, dashes):
        return "dash-dot"
    return max(part.tally, key=part.tally.get)


def _one_row_per_column(line: np.ndarray) -> np.ndarray:
    """Sort a centreline by column, averaging where two pieces overlap."""
    columns, inverse = np.unique(np.round(line[:, 0]), return_inverse=True)
    rows = np.bincount(inverse, weights=line[:, 1]) / np.bincount(inverse)
    return np.column_stack([columns, rows])


def _extrapolate(line: np.ndarray, column: float, from_end: bool, span: int = 30) -> float:
    """Where a centreline would be at ``column`` if it carried on straight."""
    slope, intercept = _end_fit(line, from_end, span)
    return float(slope * column + intercept)


def _end_fit(line: np.ndarray, from_end: bool, span: int = 30) -> tuple[float, float]:
    """The straight line a centreline follows over its last (or first) few columns."""
    if from_end:
        piece = line[line[:, 0] >= line[-1, 0] - span]
    else:
        piece = line[line[:, 0] <= line[0, 0] + span]
    if piece.shape[0] < 2 or np.ptp(piece[:, 0]) < 1:
        return 0.0, float(piece[-1 if from_end else 0, 1])
    return _fit(piece[:, 0], piece[:, 1])


def _stitch(curves: list[Part], width: int, thickness: float,
            mask: np.ndarray | None = None) -> list[Part]:
    """Join pieces of one style that follow one another along the x axis.

    Where a dashed line crosses a solid one, its dashes fuse into the solid line's ink
    for a while; the dashed track sees a gap longer than it will coast across, and
    resumes as a new track on the far side. A marker or a smear of compression noise
    can break a line the same way.

    Two pieces of one style that barely share any columns are taken to be one series if
    their ends roughly meet. The test is deliberately generous: two *different* curves
    in the same style would run side by side over a good part of their length, and
    those are left alone. Two pieces that follow one another are, on the evidence of
    the figure, one series with a gap - which is also what treating them as one would
    have done before any separating was attempted.

    Where the ends do not meet but are joined by an upright stroke of ink - the drop of
    a sawtooth, the riser of a step plot - the curve jumped there, and it is joined too.
    """
    curves = list(curves)
    limit = max(60, int(0.25 * width))
    ends: dict[tuple[int, bool], tuple[float, float]] = {}

    def extend(curve: Part, column: float, from_end: bool) -> float:
        key = (id(curve), from_end)
        if key not in ends:
            ends[key] = _end_fit(curve.centreline, from_end)
        slope, intercept = ends[key]
        return slope * column + intercept

    while True:
        best = None
        for a in curves:
            for b in curves:
                if a is b or a.style != b.style or b.centreline[0, 0] < a.centreline[0, 0]:
                    continue
                span = min(np.ptp(a.centreline[:, 0]), np.ptp(b.centreline[:, 0]))
                overlap = a.centreline[-1, 0] - b.centreline[0, 0]
                if overlap > max(10.0, 0.1 * span):
                    continue
                gap = max(0.0, -overlap)
                if gap > limit:
                    continue
                meet = 0.5 * (a.centreline[-1, 0] + b.centreline[0, 0])
                miss = abs(extend(a, meet, True) - extend(b, meet, False))
                riser = (mask is not None and gap <= max(3.0, 2.0 * thickness)
                         and _upright_link(mask, a.centreline[-1], b.centreline[0], thickness))
                if riser:
                    miss = 0.0
                if miss <= max(6.0, 3.0 * thickness) + 0.1 * gap:
                    key = (gap, miss)
                    if best is None or key < best[0]:
                        best = (key, a, b)
        if best is None:
            return curves
        _, a, b = best
        _combine(a, b)
        ends.pop((id(a), True), None)
        ends.pop((id(a), False), None)
        curves = [c for c in curves if c is not b]


def _evenly_broken(curve: Part) -> bool | None:
    """Are a broken line's pieces evenly spaced along a smooth path? None if not broken."""
    broken = sum(n for style, n in curve.tally.items()
                 if style in ("dotted", "dashed", "long-dashed"))
    if broken < 0.5 * max(1, sum(curve.tally.values())):
        return None
    count, _, _, centres = cv2.connectedComponentsWithStats(curve.mask.astype(np.uint8), 8)
    if count < 8:
        # Too few pieces to show a pattern, over a length a line must have.
        return False
    centres = centres[1:][np.argsort(centres[1:, 0])]
    spacing = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    typical = float(np.median(spacing))
    # Where the line ran under another, whole pieces are missing: the spacing there is
    # a multiple of the usual one, which is still the pattern.
    ratio = spacing / max(typical, 1e-9)
    steps = np.maximum(1.0, np.round(ratio))
    regular = float(np.mean(np.abs(ratio - steps) <= 0.25 * steps ** 0.5))
    single = float(np.mean(np.abs(ratio - 1.0) <= 0.25))
    rows = centres[:, 1]
    wander = np.median([abs(rows[i] - np.median(rows[max(0, i - 2):i + 3]))
                        for i in range(rows.size)])
    return regular >= 0.8 and single >= 0.5 and wander <= 1.0


def _upright_link(mask: np.ndarray, end: np.ndarray, start: np.ndarray,
                  thickness: float) -> bool:
    """Is there an upright stroke of ink from one piece's end to the next one's start?"""
    lo, hi = sorted((end[1], start[1]))
    if hi - lo < 2 * thickness:
        return False
    c0 = int(np.floor(min(end[0], start[0]))) - 1
    c1 = int(np.ceil(max(end[0], start[0]))) + 2
    r0, r1 = int(np.floor(lo)), int(np.ceil(hi)) + 1
    band = mask[max(0, r0):r1, max(0, c0):c1]
    if band.size == 0:
        return False
    # A thin upright line on a half pixel is two pale columns, much of it too pale to
    # be this colour: it can be broken, but it runs the whole way.
    present = band.any(axis=1)
    quarters = np.array_split(present, 4)
    return float(present.mean()) >= 0.3 and all(q.size and q.mean() >= 0.15 for q in quarters)


def _join_overlaps(curves: list[Part], thickness: float) -> list[Part]:
    """Join two pieces of one line that overlap a little and carry on either way.

    The tracker sometimes follows a stroke in two pieces that overlap for a stretch -
    on a thick or steep line whose runs it lost for a moment, or on a line that has
    just come out of a crossing. Where two pieces of one style lie on the same stroke
    over all of their overlap, and one carries on to the left and the other to the
    right, they are one line.
    """
    curves = list(curves)
    tolerance = max(2.0, thickness)
    changed = True
    while changed:
        changed = False
        for a in curves:
            for b in curves:
                if a is b or a.style != b.style:
                    continue
                if a.folded or b.folded:
                    # A curve that turns back on itself is one stroke: a piece of the
                    # same style that follows it, or runs into it, is more of it.
                    if (_masks_overlap(a.mask, b.mask, 1.5 * tolerance)
                            or _touch(a.mask, b.mask)):
                        _merge_branches(a, b)
                        curves = [c for c in curves if c is not b]
                        changed = True
                        break
                    continue
                la, lb = a.centreline, b.centreline
                if not (la[0, 0] < lb[0, 0] < la[-1, 0] < lb[-1, 0]):
                    continue
                columns = np.arange(np.ceil(lb[0, 0]), np.floor(la[-1, 0]) + 1)
                if columns.size < 3:
                    continue
                with np.errstate(invalid="ignore"):
                    apart = _across(_row_at(la, columns) - _row_at(lb, columns), la, columns)
                # Their tracks may disagree where one strayed onto a marker's outline,
                # a few pixels off the line; over a long overlap that is still one line.
                close = float(np.mean(apart <= 1.5 * tolerance)) >= 0.7
                near = (columns.size >= 20
                        and float(np.mean(apart <= max(6.0, 3.0 * thickness))) >= 0.8)
                if not (close or near or _masks_overlap(a.mask, b.mask, 1.5 * tolerance)):
                    continue
                _combine(a, b)
                curves = [c for c in curves if c is not b]
                changed = True
                break
            if changed:
                break
    return curves


def _touch(a: np.ndarray, b: np.ndarray) -> bool:
    grown = cv2.dilate(a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return int((grown & b).sum()) >= 3


def _masks_overlap(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    """Do two pieces follow the same strokes wherever both have ink, over a real overlap?

    For curves that turn back on themselves, which have no single row per column to
    compare, each column's ink is compared instead.
    """
    ca, cb = np.flatnonzero(a.any(axis=0)), np.flatnonzero(b.any(axis=0))
    if ca.size == 0 or cb.size == 0:
        return False
    lo, hi = max(ca[0], cb[0]), min(ca[-1], cb[-1])
    if hi - lo < 3 or (ca[0] <= cb[0] and ca[-1] >= cb[-1]) or (cb[0] <= ca[0] and cb[-1] >= ca[-1]):
        return False
    good = total = 0
    for column in range(lo, hi + 1):
        ra, rb = np.flatnonzero(a[:, column]), np.flatnonzero(b[:, column])
        if ra.size == 0 or rb.size == 0:
            continue
        total += 1
        d = np.abs(ra[:, None] - rb[None, :])
        if max(d.min(axis=1).mean(), d.min(axis=0).mean()) <= tolerance:
            good += 1
    return total >= 3 and good >= 0.7 * total


def _join_turns(curves: list[Part], mask: np.ndarray, thickness: float,
                width: int) -> list[Part]:
    """Join the branches of a curve that turns back on itself.

    Followed column by column, a loop comes apart into an upper and a lower branch and
    an S-shaped fold into three; each branch looks like a line of its own. Branches are
    one curve where an end of one is joined by ink to an end of the other, on the same
    side, and they meet the way a curve turns:

      * at both ends, for a closed loop;
      * head on, each arriving at the turn from the opposite side, for a fold;
      * or running back alongside each other the whole way, for a loop so thin its
        two branches almost touch.

    Two curves that merely start from a common point arrive side by side, at the angle
    between them, and then draw apart; they are left alone.
    """
    reach = max(4.0, 3.0 * thickness, 0.04 * width)
    alongside = max(6.0, 3.0 * thickness)
    curves = list(curves)
    for curve in curves:
        if not curve.branches:
            curve.branches = [curve.centreline]
            curve.ends = [(0, 0), (0, -1)]
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(curves):
            for b in curves[i + 1:]:
                if a.style != b.style:
                    continue
                joints = []
                for ka, side in a.ends:
                    for kb, other_side in b.ends:
                        if other_side != side:
                            continue
                        pa, pb = a.branches[ka][side], b.branches[kb][side]
                        if (abs(pa[0] - pb[0]) <= reach
                                and _ink_joins(mask, pa, pb, thickness)
                                and not _carries_on(mask, pa, pb, side, thickness)):
                            joints.append(((ka, side), (kb, side)))
                if not joints:
                    continue
                closed = len({j[0] for j in joints}) >= 2 and len({j[1] for j in joints}) >= 2
                turn = hairpin = False
                for (ka, side), (kb, _) in joints:
                    la, lb = a.branches[ka], b.branches[kb]
                    turn |= _arrival_angle(la, lb, side, thickness) >= 100.0
                    hairpin |= _alongside(la, lb, alongside) >= 0.8
                if closed or turn or hairpin:
                    used_a = {j[0] for j in joints}
                    used_b = {j[1] for j in joints}
                    offset = len(a.branches)
                    a.ends = ([e for e in a.ends if e not in used_a]
                              + [(k + offset, side) for k, side in b.ends
                                 if (k, side) not in used_b])
                    a.branches = a.branches + b.branches
                    _merge_branches(a, b)
                    curves = [c for c in curves if c is not b]
                    changed = True
                    break
            if changed:
                break
    return curves


def _alongside(a: np.ndarray, b: np.ndarray, distance: float) -> float:
    """Fraction of the shorter centreline that runs within ``distance`` of the other."""
    short, long_ = (a, b) if np.ptp(a[:, 0]) <= np.ptp(b[:, 0]) else (b, a)
    columns = np.unique(np.round(short[:, 0]))
    with np.errstate(invalid="ignore"):
        apart = _across(_row_at(short, columns) - _row_at(long_, columns), long_, columns)
    return float(np.mean(apart <= distance))


def _carries_on(mask: np.ndarray, pa: np.ndarray, pb: np.ndarray, side: int,
                thickness: float) -> bool:
    """Does ink continue past where two branch ends meet?

    At a turn the curve stops: past the two ends there is at most the turn's own cap.
    Where two tracks were only lost at a crossing, the curves carry on beyond it.
    """
    pad = int(max(3, 2 * thickness))
    length = int(max(8, 4 * thickness))
    r0 = max(0, int(min(pa[1], pb[1])) - pad)
    r1 = min(mask.shape[0], int(max(pa[1], pb[1])) + pad + 1)
    if side == 0:
        edge = int(min(pa[0], pb[0])) - 2 * pad
        c0, c1 = max(0, edge - length), max(0, edge)
    else:
        edge = int(np.ceil(max(pa[0], pb[0]))) + 2 * pad + 1
        c0, c1 = min(mask.shape[1], edge), min(mask.shape[1], edge + length)
    if c1 - c0 < 2:
        return False
    return float(mask[r0:r1, c0:c1].any(axis=0).mean()) >= 0.6


def _ink_joins(mask: np.ndarray, pa: np.ndarray, pb: np.ndarray, thickness: float) -> bool:
    """Are two points on one connected stretch of ink, close around them?"""
    pad = int(max(3, 2 * thickness))
    r0 = max(0, int(min(pa[1], pb[1])) - pad)
    r1 = min(mask.shape[0], int(max(pa[1], pb[1])) + pad + 1)
    c0 = max(0, int(min(pa[0], pb[0])) - pad)
    c1 = min(mask.shape[1], int(max(pa[0], pb[0])) + pad + 1)
    window = mask[r0:r1, c0:c1].astype(np.uint8)
    if not window.any():
        return False
    _, labels = cv2.connectedComponents(window, connectivity=8)

    def label_near(point) -> set[int]:
        r, c = int(round(point[1])) - r0, int(round(point[0])) - c0
        near = labels[max(0, r - pad):r + pad + 1, max(0, c - pad):c + pad + 1]
        return set(np.unique(near[near > 0]).tolist())

    return bool(label_near(pa) & label_near(pb))


def _arrival_angle(a: np.ndarray, b: np.ndarray, side: int, thickness: float) -> float:
    """Angle, in degrees, between the directions two centrelines arrive at one end."""
    length = max(8.0, 4.0 * thickness)

    def arriving(line: np.ndarray) -> np.ndarray:
        end = line[side]
        if side == 0:
            back = line[line[:, 0] >= end[0] + length]
            back = back[0] if back.size else line[-1]
        else:
            back = line[line[:, 0] <= end[0] - length]
            back = back[-1] if back.size else line[0]
        v = end - back
        return v / max(1e-9, float(np.linalg.norm(v)))

    cosine = float(np.clip(arriving(a) @ arriving(b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _merge_branches(home: Part, other: Part) -> None:
    home.mask |= other.mask
    for style, count in other.tally.items():
        home.tally[style] = home.tally.get(style, 0) + count
    for style, count in other.pieces.items():
        home.pieces[style] = home.pieces.get(style, 0) + count
    line = np.vstack([home.centreline, other.centreline])
    home.centreline = line[np.argsort(line[:, 0], kind="stable")]
    home.folded = True
    home.points = None


def _fold_fragments(curves: list[Part], min_length: int, thickness: float = 2.0) -> list[Part]:
    """Join the pieces of a style that never run side by side.

    Where a line runs through a crowd of markers, or tangles with another line, it is
    tracked in pieces with gaps too long and too wiggly to bridge by extrapolating. A
    reader joins them on style alone: if the dotted line never appears twice in the
    same place, every stretch of dotted line is the same line. So a style's pieces are
    folded into its longest one for as long as each covers columns the others do not.
    A piece that runs alongside is a second line in that style, and is left alone -
    which also means that where a style has several full-length lines, as in a figure
    of solid curves, nothing is folded at all.

    The longest piece must be long enough to be a series by itself, so the scraps of a
    scatter's markers, which have no real line to join, are not strung together into
    one.
    """
    out = list(curves)
    for style in {c.style for c in curves}:
        group = sorted((c for c in out if c.style == style),
                       key=lambda c: np.ptp(c.centreline[:, 0]), reverse=True)
        home = group[0]
        if np.ptp(home.centreline[:, 0]) + 1 < min_length:
            continue
        covered = set(np.round(home.centreline[:, 0]).astype(int).tolist())
        for piece in group[1:]:
            columns = set(np.round(piece.centreline[:, 0]).astype(int).tolist())
            if (len(columns & covered) <= 0.1 * len(columns)
                    and _continues(home.centreline, piece.centreline, thickness)):
                _combine(home, piece)
                covered |= columns
                out = [c for c in out if c is not piece]
    return out


def _continues(line: np.ndarray, piece: np.ndarray, thickness: float) -> bool:
    """Could a piece be more of a line, given where the line is on either side of it?

    Across a gap a curve can go anywhere its own slope allows, and a good deal more
    where it was lost among markers - but not leap as if it were a different curve.
    Two curves drawn over different stretches of x in the same style meet here as a
    jump far steeper than either of them.
    """
    for side in (0, -1):
        end = piece[side]
        if side == 0:
            before = line[line[:, 0] < end[0]]
            if before.size == 0:
                continue
            near = before[-1]
        else:
            after = line[line[:, 0] > end[0]]
            if after.size == 0:
                continue
            near = after[0]
        gap = abs(end[0] - near[0])
        steep = max(_end_slope(line, near[0]), _end_slope(piece, end[0]), 0.5)
        if abs(end[1] - near[1]) > 3.0 * thickness + 2.0 * steep * gap:
            return False
    return True


def _end_slope(line: np.ndarray, column: float, span: float = 20.0) -> float:
    near = line[np.abs(line[:, 0] - column) <= span]
    if near.shape[0] < 2 or np.ptp(near[:, 0]) < 1:
        return 0.0
    return abs(_fit(near[:, 0], near[:, 1])[0])


def _absorb(curves: list[Part], thickness: float, settings: SeparationSettings) -> list[Part]:
    """Fold a track into a longer one it lies on for most of its own length.

    This is what reassembles a dash-dot line, whose dashes and dots were followed as
    two styles, and what collects the odd stretch of a dashed line that fused with
    something and was taken for solid. It is asymmetric on purpose: the question is
    whether the *shorter* one is part of the longer, so two long curves that merely
    converge for part of their length are left alone.
    """
    order = sorted(curves, key=lambda c: np.ptp(c.centreline[:, 0]), reverse=True)
    kept: list[Part] = []
    tolerance = max(2.0, thickness)
    for curve in order:
        columns = np.unique(np.round(curve.centreline[:, 0]))
        rows = _row_at(curve.centreline, columns)
        home = None
        best = settings.colocated_fraction
        for other in kept:
            theirs = _row_at(other.centreline, columns)
            with np.errstate(invalid="ignore"):
                apart = _across(theirs - rows, other.centreline, columns)
            fraction = float(np.mean(apart <= tolerance))
            if fraction >= best:
                home, best = other, fraction
        if home is None:
            kept.append(curve)
        else:
            _combine(home, curve)
    return kept


def _reach(line: np.ndarray, columns: np.ndarray, thickness: float) -> np.ndarray:
    """How far from the centreline a column's ink may lie and still be this stroke."""
    slope = np.gradient(line[:, 1], line[:, 0]) if line.shape[0] > 1 else np.zeros(1)
    local = np.interp(columns, line[:, 0], np.abs(slope))
    return max(2.0, thickness) + local


def _fill_gaps(curve: Part, free: np.ndarray, thickness: float) -> None:
    """Claim unowned ink on the curve's own path in the columns it has none."""
    line = curve.centreline
    first, last = int(np.ceil(line[0, 0])), int(np.floor(line[-1, 0]))
    has_ink = curve.mask[:, first:last + 1].any(axis=0)
    missing = first + np.flatnonzero(~has_ink)
    if missing.size == 0:
        return
    # Across a long gap the straight line between its ends says little about where the
    # curve went, so only short gaps are filled.
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    missing = np.concatenate([r for r in runs if r.size <= 60] or [np.empty(0, int)])
    if missing.size == 0:
        return
    rows = _row_at(line, missing.astype(float))
    reach = _reach(line, missing.astype(float), thickness)
    height = free.shape[0]
    for column, row, r in zip(missing, rows, reach):
        if not np.isfinite(row):
            continue
        lo, hi = max(0, int(np.floor(row - r))), min(height, int(np.ceil(row + r)) + 1)
        if hi > lo and free[lo:hi, column].any():
            curve.mask[lo:hi, column] |= free[lo:hi, column]
            free[lo:hi, column] = False


def _curve_points(curve: Part, thickness: float) -> np.ndarray:
    """One point per column: the centre of the curve's ink near its centreline.

    Taking all of a column's ink would let a marker drawn on the line, or the ink of
    another curve at a crossing, pull the point off the line; only the ink within a
    stroke's reach of where the track says the line is counts.
    """
    line = curve.centreline
    columns = np.flatnonzero(curve.mask.any(axis=0))
    columns = columns[(columns >= line[0, 0] - 1) & (columns <= line[-1, 0] + 1)]
    if columns.size == 0:
        return np.empty((0, 2))
    rows = np.interp(columns, line[:, 0], line[:, 1])
    reach = _reach(line, columns.astype(float), thickness)
    points = []
    for column, row, r in zip(columns, rows, reach):
        ink = np.flatnonzero(curve.mask[:, column])
        near = ink[np.abs(ink - row) <= r]
        # Only where there is ink: a column whose ink all lies away from the track has
        # no point to give, and the track's own row there would be one on bare paper.
        if near.size:
            points.append((float(column), float(near.mean())))
    return np.asarray(points) if points else np.empty((0, 2))


def _marker_band(mask: np.ndarray, cls: _MarkerClass,
                 recall: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Markers packed so tightly they merge into a band, and where the band runs.

    Matching one marker at a time fails here - each copy is half hidden by the next.
    What still works is asking, at every position, whether a whole marker *could* be
    there: whether the ink covers nearly all of its outline. Beside an isolated
    marker that is true at one spot. Along a band of overlapping markers it is true
    all the way along, as a ridge; along a line it is never true, since a straight
    stroke covers only a small part of any glyph. A ridge much longer than the marker
    itself is a band.

    Returns the band's ink, to be kept out of the line tracking, and one point per
    column along the ridge for the marker series.
    """
    template = cls.template.astype(np.float32)
    th, tw = template.shape
    if mask.shape[0] < th or mask.shape[1] < tw:
        return np.zeros(mask.shape, dtype=bool), np.empty((0, 2))
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    fit = cv2.matchTemplate(dilated, template, cv2.TM_CCORR) / max(1.0, float(template.sum()))
    fits = (fit >= recall).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fits, 8)
    oy, ox = cls.anchor
    ink = np.zeros(mask.shape, dtype=bool)
    points = []
    # A few markers overlapping in a cluster leave a short ridge too; a band is several
    # markers long, and nothing shorter needs anything but matching.
    bands = np.flatnonzero(stats[:, cv2.CC_STAT_WIDTH] >= max(4 * tw, 40))
    bands = bands[bands > 0]
    if bands.size == 0:
        return ink, np.empty((0, 2))
    ridge = np.isin(labels, bands)
    # Stamping the glyph at every ridge position is a dilation of the ridge by the glyph,
    # anchored where the template's top-left corner sits.
    glyph = cv2.dilate(cls.hull.astype(np.uint8), np.ones((3, 3), np.uint8))
    placed = np.zeros(mask.shape, dtype=np.uint8)
    placed[:ridge.shape[0], :ridge.shape[1]] = ridge
    ink = cv2.dilate(placed, glyph[::-1, ::-1], anchor=(glyph.shape[1] - 2, glyph.shape[0] - 2),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    for band in bands:
        rows, cols = np.nonzero(labels == band)
        sums = np.bincount(cols, weights=rows)
        counts = np.bincount(cols)
        for column in np.flatnonzero(counts):
            points.append((float(column + ox), float(sums[column] / counts[column] + oy)))
    return ink & mask, np.asarray(points)
