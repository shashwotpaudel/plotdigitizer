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

__all__ = ["Part", "SeparationSettings", "separate_series", "stroke_thickness"]


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


def stroke_thickness(mask: np.ndarray) -> float:
    """Typical stroke width of the ink's lines, in pixels.

    Each pixel is measured across the narrower of its vertical and horizontal runs,
    which is the stroke's width whether the stroke is flat, upright or diagonal. Only
    elongated ink is measured - lines and dashes. Filled markers would otherwise report
    their own width, and on a scatter of filled circles every circle would then look
    no bigger than a stroke. With no elongated ink at all there is no stroke width to
    measure, and a typical two pixels stands in.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 2.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    elongated = np.zeros(count, dtype=bool)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if max(w, h) < 6:
            continue
        rows, cols = np.nonzero(labels[y:y + h, x:x + w] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32))
        long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
        # Long and thin, or long outright - a line curving through the plot.
        elongated[i] = long_side >= 3 * short_side or long_side >= 40
    lines = elongated[labels] & mask
    if lines.sum() < 20:
        return 2.0
    vertical = _run_lengths(mask)[lines]
    horizontal = _run_lengths(mask.T).T[lines]
    return float(max(1.0, np.percentile(np.minimum(vertical, horizontal), 30)))


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
        box_fill = area / float(w * h)
        if box_fill < 0.45 and solid > 0.9:
            # Nothing enclosed and little of its box inked: strokes, not an outline.
            return "cross"
        # Counting corners does not work at marker sizes, so two coarser measures. A
        # triangle's weight sits towards its base, which nothing symmetric does; after
        # that, how much of its box the outline covers - a square all of it, a circle
        # about pi/4, a diamond half.
        rows = np.nonzero(hull)[0]
        offset = (rows.mean() - (h - 1) / 2) / max(1.0, h)
        if abs(offset) > 0.06:
            shape = "triangle" if offset > 0 else "down triangle"
        elif box_fill > 0.88:
            shape = "square"
        elif box_fill > 0.64:
            shape = "circle"
        else:
            shape = "diamond"
        return f"{'filled' if solid >= 0.8 else 'hollow'} {shape}"


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


def _glyph_classes(mask: np.ndarray, thickness: float,
                   settings: SeparationSettings) -> list[_MarkerClass]:
    """Shapes that recur often enough, at one size, to be markers.

    Line pieces are kept out first. A dot of a dotted line is barely wider than the
    stroke, and a dash or an arc between two markers is long and thin, so only
    components noticeably larger than the stroke and roughly as tall as they are wide
    are considered at all.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    glyphs = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        big = max(w, h)
        if big < max(5, 2.4 * thickness) or big > 60 or min(w, h) < 0.45 * big:
            continue
        own = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        points = np.column_stack(np.nonzero(own)).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(points)
        if min(rw, rh) < 0.45 * max(rw, rh, 1.0):
            continue
        glyphs.append((own, _descriptor(own), (int(w), int(h))))

    classes: list[_MarkerClass] = []
    for own, vector, (w, h) in glyphs:
        for cls in classes:
            cw, ch = cls.size
            if (abs(w - cw) <= max(2, 0.2 * cw) and abs(h - ch) <= max(2, 0.2 * ch)
                    and float(vector @ cls.vector) >= settings.glyph_similarity):
                cls.members += 1
                break
        else:
            classes.append(_MarkerClass(own, _fill_holes(own), vector, (w, h), 1))
    return [c for c in classes if c.members >= settings.min_markers]


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
    ty, tx = np.nonzero(cls.template)
    centres = np.column_stack([cols + tx.mean(), rows + ty.mean()])
    return centres, score[rows, cols]


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
            reach = 0.5 * min(max(ca.size), max(cb.size))
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
        glyph = cv2.dilate(glyph.astype(np.uint8),
                           np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
        pad = grow
    else:
        pad = 0
    ty, tx = np.nonzero(cls.template)
    oy, ox = ty.mean() + pad, tx.mean() + pad
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
    kinds = np.zeros(count, dtype="<U7")
    lengths = np.zeros(count)
    dot_limit = 2.4 * thickness + 1.0
    touches = np.zeros(count, dtype=bool)
    if cut is not None and cut.any():
        touches[np.unique(labels[cut & (labels > 0)])] = True
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if touches[i] and max(w, h) <= 20 * thickness:
            kinds[i] = "cut"
            continue
        if max(w, h) <= dot_limit:
            kinds[i] = "dotted"
            continue
        if max(w, h) <= 3:
            kinds[i] = "dotted"
            continue
        own = np.column_stack(np.nonzero(labels[y:y + h, x:x + w] == i)).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(own)
        lengths[i] = max(rw, rh, area / max(thickness, 1.0))
        kinds[i] = "solid"

    short = [i for i in range(1, count) if kinds[i] == "solid" and lengths[i] <= 20 * thickness]
    # A short piece of a line points along the line: its long axis aims at the next
    # piece. The arc left over from a marker that was only partly matched points
    # anywhere. Pieces that do not line up with a neighbour are neither dash nor line.
    aligned = _aligned_pieces(labels, stats, centroids, short)
    for i in short:
        if i not in aligned:
            kinds[i] = "debris"
    short = [i for i in short if i in aligned]
    if len(short) >= 6:
        typical = float(np.median(lengths[short]))
        similar = [i for i in short if 0.7 * typical <= lengths[i] <= 1.35 * typical]
        if len(similar) >= 6 and len(similar) >= 0.5 * len(short):
            for i in short:
                if lengths[i] <= 1.6 * typical:
                    kinds[i] = "dashed"

    out: dict[str, np.ndarray] = {}
    for style in ("solid", "dashed", "dotted"):
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
        for b in np.argsort(distance[a])[:2]:
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
    observed: dict[int, tuple[float, int, int, bool]] = field(default_factory=dict)
    misses: int = 0
    alive: bool = True

    def predict(self, column: int, window: int = 8) -> tuple[float, float]:
        cols = np.asarray(self.columns[-window:], dtype=float)
        rows = np.asarray(self.rows[-window:], dtype=float)
        if cols.size < 2 or np.ptp(cols) < 1e-9:
            return float(rows[-1]), 0.0
        slope = float(np.polyfit(cols, rows, 1)[0])
        return float(rows[-1] + slope * (column - cols[-1])), slope


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
          births: list[_Track] | None) -> None:
    """Advance every track through ``order``, one column at a time, together."""
    base_gate = max(3.0, 1.5 * thickness) + pool
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
                if track.misses > settings.max_gap:
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
                shared = True
            else:
                lo, hi = start, end - 1
                centre = 0.5 * (start + end - 1)
            track.misses = 0
            track.columns.append(column)
            track.rows.append(centre)
            track.observed[column] = (centre, max(lo, start), min(hi, end - 1), shared)

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
    counts = np.array([len(r) for r in runs])
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
    _walk(right, runs, range(end, mask.shape[1]), thickness, pool, settings, born)
    left = [_Track(list(reversed(t.columns)), list(reversed(t.rows)), t.observed)
            for t in right[:k]]
    born_left: list[_Track] = []
    _walk(left, runs, range(start - 1, -1, -1), thickness, pool, settings, born_left)

    # A curve that first appeared on the way right may extend further left than where
    # it was noticed, and vice versa; follow each newcomer back the other way.
    for track in born + born_left:
        columns = sorted(track.observed)
        noticed = set(columns)
        if track in born:
            back = _Track(columns[:8][::-1], [track.observed[c][0] for c in columns[:8][::-1]],
                          track.observed)
            _walk([back], runs, range(columns[0] - 1, -1, -1), thickness, pool, settings, None)
        else:
            ahead = _Track(columns[-8:], [track.observed[c][0] for c in columns[-8:]],
                           track.observed)
            _walk([ahead], runs, range(columns[-1] + 1, mask.shape[1]), thickness, pool,
                  settings, None)
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
            close = sum(abs(track.observed[c][0] - other.observed[c][0]) <= max(2.0, thickness)
                        for c in common)
            if close >= 0.6 * len(track.observed):
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
    return kept


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
                    settings: SeparationSettings | None = None) -> list[Part] | None:
    """Split one colour's ink into its series, or ``None`` if it is only one."""
    settings = settings or SeparationSettings()
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 30:
        return None
    shape = mask.shape
    columns = np.flatnonzero(mask.any(axis=0))
    width = plot_width or int(columns[-1] - columns[0] + 1)
    min_length = max(12, int(settings.min_track_fraction * width))
    # Pieces are kept a good deal shorter than a series has to be, because a line
    # broken where it crosses markers or another curve is only reassembled from
    # its pieces afterwards; judging each piece on its own would discard them.
    min_piece = max(12, int(0.05 * width))
    thickness = stroke_thickness(mask)

    # 1. Markers, found wherever they are and lifted out of the ink.
    classes = _glyph_classes(mask, thickness, settings)
    for cls in classes:
        _locate(mask, cls, settings)
    _resolve_overlaps(classes)
    classes = [c for c in classes if len(c.centres) >= settings.min_markers]
    band_points: dict[int, list[np.ndarray]] = {}
    band_ink = np.zeros(shape, dtype=bool)
    for index, cls in enumerate(classes):
        ink, points = _marker_band(mask, cls)
        if points.size:
            band_ink |= ink
            band_points[index] = [points]
    removed = band_ink.copy()
    for cls in classes:
        # Two pixels of margin: sub-pixel placement makes some copies of a glyph a
        # pixel larger than the template.
        removed |= _stamp(shape, cls, cls.centres, "hull", grow=2)
    strokes = mask & ~removed
    cut = cv2.dilate(removed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~removed

    # 2. The remaining strokes, by style, each style followed curve by curve.
    curves: list[Part] = []
    for style, style_mask in _style_masks(strokes, thickness, cut).items():
        for track in _track_style(style_mask, thickness, settings, min_piece,
                                  broken=style != "solid"):
            part_mask = _track_mask(shape, style_mask, track)
            curves.append(Part("curve", style, part_mask, centreline=_centreline(track),
                               tally={style: int(part_mask.sum())}))

    # 3. Put lines back together: pieces of one style broken where another line
    #    crossed, and fragments that lie along a longer line for most of their length.
    curves = _stitch(curves, width, thickness)
    curves = _absorb(curves, thickness, settings)
    curves = _stitch(curves, width, thickness)
    curves = _fold_fragments(curves, min_length)
    # Only now, with the pieces joined up, is a line judged on its length.
    curves = [c for c in curves if np.ptp(c.centreline[:, 0]) + 1 >= min_length]

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
    free = mask & ~claimed & ~marker_ink
    for curve in curves:
        _fill_gaps(curve, free, thickness)
        curve.points = _curve_points(curve, thickness)

    # 6. Markers drawn on a line belong to it; markers on no line are their own series.
    parts: list[Part] = list(curves)
    for index, cls in enumerate(classes):
        owners = np.full(len(cls.centres), -1)
        for i, (cx, cy) in enumerate(cls.centres):
            best = settings.on_line_px
            for j, curve in enumerate(curves):
                row = _row_at(curve.centreline, np.array([cx]))[0]
                if np.isfinite(row) and abs(row - cy) <= best:
                    best, owners[i] = abs(row - cy), j
        on_lines = owners >= 0
        if curves and index not in band_points and on_lines.mean() >= settings.on_line_fraction:
            for j in set(owners[on_lines].tolist()):
                curve = curves[j]
                curve.mask |= _stamp(shape, cls, cls.centres[owners == j], "template") & mask
                curve.marker = cls.describe()
            continue
        glyph = _stamp(shape, cls, cls.centres, "template") & mask
        points = cls.centres
        if index in band_points:
            band = np.vstack(band_points[index])
            if len(cls.centres) and len(band):
                # Where the band's markers were matched one by one, those matches are
                # the answer; the band's own points only fill where they were not.
                near = np.abs(band[:, None, 0] - cls.centres[None, :, 0]).min(axis=1)
                band = band[near > 0.5 * cls.template.shape[1]]
            points = np.vstack([points, band])
            points = points[np.argsort(points[:, 0], kind="stable")]
        parts.append(Part("markers", "markers", glyph, marker=cls.describe(), points=points))

    parts = [p for p in parts if p.mask.sum() >= 12 or (p.points is not None and len(p.points))]
    if len(parts) <= 1:
        return None
    log.info("split one colour into %d series: %s", len(parts),
             ", ".join(p.description for p in parts))
    return parts


# -- reassembly --------------------------------------------------------------------


def _combine(home: Part, other: Part) -> None:
    home.mask |= other.mask
    for style, count in other.tally.items():
        home.tally[style] = home.tally.get(style, 0) + count
    home.centreline = _one_row_per_column(np.vstack([home.centreline, other.centreline]))
    total = sum(home.tally.values()) or 1
    shares = {k: v / total for k, v in home.tally.items()}
    if shares.get("dashed", 0) >= 0.15 and shares.get("dotted", 0) >= 0.1:
        home.style = "dash-dot"
    else:
        home.style = max(home.tally, key=home.tally.get)


def _one_row_per_column(line: np.ndarray) -> np.ndarray:
    """Sort a centreline by column, averaging where two pieces overlap."""
    columns, inverse = np.unique(np.round(line[:, 0]), return_inverse=True)
    rows = np.bincount(inverse, weights=line[:, 1]) / np.bincount(inverse)
    return np.column_stack([columns, rows])


def _extrapolate(line: np.ndarray, column: float, from_end: bool, span: int = 30) -> float:
    """Where a centreline would be at ``column`` if it carried on straight."""
    if from_end:
        piece = line[line[:, 0] >= line[-1, 0] - span]
    else:
        piece = line[line[:, 0] <= line[0, 0] + span]
    if piece.shape[0] < 2 or np.ptp(piece[:, 0]) < 1:
        return float(piece[-1 if from_end else 0, 1])
    slope, intercept = np.polyfit(piece[:, 0], piece[:, 1], 1)
    return float(slope * column + intercept)


def _stitch(curves: list[Part], width: int, thickness: float) -> list[Part]:
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
    """
    curves = list(curves)
    limit = max(60, int(0.25 * width))
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
                miss = abs(_extrapolate(a.centreline, meet, from_end=True)
                           - _extrapolate(b.centreline, meet, from_end=False))
                if miss <= max(6.0, 3.0 * thickness) + 0.1 * gap:
                    key = (gap, miss)
                    if best is None or key < best[0]:
                        best = (key, a, b)
        if best is None:
            return curves
        _, a, b = best
        _combine(a, b)
        curves = [c for c in curves if c is not b]


def _fold_fragments(curves: list[Part], min_length: int) -> list[Part]:
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
            if len(columns & covered) <= 0.1 * len(columns):
                _combine(home, piece)
                covered |= columns
                out = [c for c in out if c is not piece]
    return out


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
            fraction = float(np.mean(np.abs(theirs - rows) <= tolerance))
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
        points.append((float(column), float(near.mean()) if near.size else float(row)))
    return np.asarray(points)


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
    ty, tx = np.nonzero(cls.template)
    oy, ox = ty.mean(), tx.mean()
    ink = np.zeros(mask.shape, dtype=bool)
    points = []
    for i in range(1, count):
        # A few markers overlapping in a cluster leave a short ridge too; a band is
        # several markers long, and nothing shorter needs anything but matching.
        if stats[i, cv2.CC_STAT_WIDTH] < max(4 * tw, 40):
            continue
        rows, cols = np.nonzero(labels == i)
        where = np.column_stack([cols + ox, rows + oy])
        ink |= _stamp(mask.shape, cls, where, "hull", grow=1)
        for column in np.unique(cols):
            points.append((float(column + ox), float(rows[cols == column].mean() + oy)))
    return ink & mask, (np.asarray(points) if points else np.empty((0, 2)))
