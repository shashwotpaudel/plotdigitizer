"""Follow one stroke from a seed point, even when it shares its colour with others.

Colour separation cannot help on a black-and-white figure: three curves drawn in the
same ink are one cluster, and no threshold will split them. What *does* distinguish
them is that each one is continuous - it goes somewhere, smoothly - while the others
merely pass nearby.

So this is a tracker, not a flood fill. Starting from a seeded column it steps outwards
one column at a time, and in each new column picks the run of ink whose centre best
continues the trajectory it has been following. Where two curves cross, both runs are
present and the one matching the current slope wins; a flood fill would have swallowed
both. Where a dotted line has a gap, no run is present at all, and the tracker coasts
on its predicted slope until ink reappears.

The limits are honest ones: where curves run genuinely on top of each other for a long
stretch there is no information to separate them, and the tracker will follow whichever
is closer to its prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TraceSettings", "trace_stroke", "runs_in_column", "stroke_mask", "sweep_along"]


@dataclass
class TraceSettings:
    """Knobs for :func:`trace_stroke`, all in pixels unless noted."""

    #: How far the centre of a run may sit from the prediction and still be accepted.
    max_jump: float = 6.0
    #: How many consecutive columns without usable ink before the trace stops.
    max_gap: int = 30
    #: Columns between recorded points. 1 records every column.
    step: int = 1
    #: How many recent centres feed the slope estimate.
    slope_window: int = 8
    #: Runs longer than this are ignored - a vertical rule or a filled block, not a stroke.
    max_run: float = 40.0
    #: How many columns either side of the seed to search for something to start on.
    seed_search: int = 10
    #: Two candidate runs closer together than this count as an ambiguous crossing.
    ambiguous_margin: float = 2.5


def stroke_mask(rgb: np.ndarray, ink, frame=None, column: float = 0.0, row: float = 0.0,
                colour_tolerance: float = 45.0, radius: int = 3,
                legend_rect: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The ink a trace is allowed to follow, restricted around a seed.

    The tracer follows geometry, so it will happily walk onto anything inked - the
    axis frame, a tick label, a curve of a completely different colour that happens to
    cross. Narrowing the mask first is what keeps it honest:

      * to the plot interior, so the frame and labels are out of reach; and
      * to the seed's *ink*, which removes other series when the figure is in colour
        and changes nothing when it is monochrome - in that case continuity is the only
        separator available, which is exactly the situation this is for.

    "The seed's ink" has to mean the whole anti-aliased family of that colour, not a
    narrow band around one sampled pixel. A dotted stroke is mostly partial coverage,
    so matching sampled colours within a fixed distance keeps only its darkest cores
    and throws away most of the line - which then reads as a series of unbridgeable
    gaps. Testing instead whether a pixel is that ink blended with the background keeps
    every level of coverage while still excluding a genuinely different colour.
    """
    mask = np.asarray(ink.mask).copy()

    if frame is not None:
        interior = np.zeros_like(mask)
        r0, r1, c0, c1 = frame.interior_bounds(inset=2)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(mask.shape[0], r1), min(mask.shape[1], c1)
        if r1 > r0 and c1 > c0:
            interior[r0:r1, c0:c1] = True
        mask &= interior

    # The legend's sample lines are drawn in the series' own styles and colours, so a
    # trace seeded near one would happily follow a swatch instead of the data.
    if legend_rect is not None:
        x0, y0, x1, y1 = legend_rect
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False

    height, width = mask.shape
    col, row_i = int(round(column)), int(round(row))
    r0, r1 = max(0, row_i - radius), min(height, row_i + radius + 1)
    c0, c1 = max(0, col - radius), min(width, col + radius + 1)
    patch = mask[r0:r1, c0:c1]
    if not patch.any():
        return mask

    # Take the purest sample near the seed - the pixel furthest from the background -
    # as the ink colour. A median would average in half-covered edge pixels and give a
    # washed-out colour that no fully-inked pixel then matches.
    colours = np.asarray(rgb)[r0:r1, c0:c1][patch].reshape(-1, 3).astype(np.float32)
    strength = np.asarray(ink.distance)[r0:r1, c0:c1][patch].reshape(-1)
    seed_colour = colours[int(np.argmax(strength))]

    background = np.asarray(ink.background, dtype=np.float32)
    axis = seed_colour - background
    if float(axis @ axis) < 1e-6:
        return mask

    image = np.asarray(rgb, dtype=np.float32) - background
    alpha = (image @ axis) / float(axis @ axis)
    residual = np.linalg.norm(image - alpha[..., None] * axis, axis=2)
    same_ink = (residual <= colour_tolerance) & (alpha > 0.1)

    kept = mask & same_ink
    # If the test somehow excludes nearly everything, the restriction is doing more
    # harm than good; fall back to the unrestricted interior.
    return kept if kept.sum() >= 0.2 * mask.sum() else mask


def runs_in_column(mask: np.ndarray, column: int) -> list[tuple[int, int]]:
    """Contiguous inked spans in one column, as (start, end_exclusive) pairs."""
    if not (0 <= column < mask.shape[1]):
        return []
    values = mask[:, column]
    if not values.any():
        return []
    edges = np.diff(values.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if values[0]:
        starts.insert(0, 0)
    if values[-1]:
        ends.append(values.size)
    return list(zip(starts, ends))


def sweep_along(mask: np.ndarray, path, corridor: float = 12.0, step: int = 1,
                max_run: float = 40.0) -> np.ndarray:
    """Read the stroke the user dragged along, one point per column swept.

    Where several curves overlap, no amount of analysis can say which one was meant -
    but the person dragging the mouse knows. Their path is taken as the answer: in each
    column only ink within ``corridor`` pixels of where they dragged is eligible, and
    the run nearest the path wins. That is the one piece of information the automatic
    tracer does not have.

    The cursor path is *interpolated* across the columns it spans rather than sampled
    where the mouse events happened to land. Mouse events arrive at a rate that depends
    on how fast the hand moves, so sampling them directly would make a quick sweep
    sparser than a slow one over the same curve - the data would record the gesture
    instead of the figure.

    Columns with no ink in the corridor simply produce no point, so a sweep that strays
    off the curve leaves a gap rather than inventing a value.
    """
    mask = np.asarray(mask, dtype=bool)
    samples = np.asarray(path, dtype=float).reshape(-1, 2)
    if mask.ndim != 2 or samples.shape[0] < 2 or not mask.any():
        return np.empty((0, 2), dtype=float)

    height, width = mask.shape

    # Sort by column and average duplicates so the guide is a function of x that
    # np.interp can evaluate; a drag that doubles back is flattened rather than refused.
    order = np.argsort(samples[:, 0], kind="stable")
    xs, ys = samples[order, 0], samples[order, 1]
    unique_x, inverse = np.unique(xs, return_inverse=True)
    guide_y = np.bincount(inverse, weights=ys) / np.bincount(inverse)

    first = max(0, int(np.floor(unique_x.min())))
    last = min(width - 1, int(np.ceil(unique_x.max())))
    if last < first:
        return np.empty((0, 2), dtype=float)

    columns = np.arange(first, last + 1, max(1, int(step)))
    guides = np.interp(columns, unique_x, guide_y)

    points: list[tuple[float, float]] = []
    for column, guide in zip(columns, guides):
        if not (0 <= guide < height):
            continue
        best: tuple[float, float] | None = None
        for start, end in runs_in_column(mask, int(column)):
            if end - start > max_run:
                continue
            centre = 0.5 * (start + end - 1)
            # Inside the run counts as no distance at all: on a steep stretch the run
            # is tall and its centre can be further from the path than the corridor
            # allows, even though the path is running right through it.
            distance = 0.0 if start <= guide <= end - 1 else min(abs(start - guide),
                                                                 abs(end - 1 - guide))
            if distance > corridor:
                continue
            if best is None or distance < best[0]:
                best = (distance, centre)
        if best is not None:
            points.append((float(column), best[1]))

    return np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float)


def _predict(centres: list[float], columns: list[int], next_column: int,
             window: int) -> float:
    """Where the stroke should be in ``next_column``, from its recent slope.

    Extrapolating rather than reusing the last value is what keeps the tracker on a
    steep curve: on a slope of two rows per column, a flat prediction falls behind by
    more than the jump limit within a few steps and the trace dies.
    """
    if not centres:
        return 0.0
    if len(centres) == 1:
        return centres[-1]
    recent_y = np.array(centres[-window:], dtype=float)
    recent_x = np.array(columns[-window:], dtype=float)
    if recent_x.size < 2 or np.ptp(recent_x) < 1e-9:
        return float(recent_y[-1])
    slope = np.polyfit(recent_x, recent_y, 1)[0]
    return float(recent_y[-1] + slope * (next_column - recent_x[-1]))


def _walk(mask: np.ndarray, seed_column: int, seed_centre: float,
          direction: int, settings: TraceSettings,
          stats: dict | None = None) -> list[tuple[int, float]]:
    """March one way from the seed, returning (column, centre) pairs."""
    height, width = mask.shape
    centres: list[float] = [seed_centre]
    columns: list[int] = [seed_column]
    found: list[tuple[int, float]] = []

    column = seed_column + direction * settings.step
    gap = 0
    while 0 <= column < width:
        predicted = _predict(centres, columns, column, settings.slope_window)
        candidates = [
            (start, end) for start, end in runs_in_column(mask, column)
            if end - start <= settings.max_run
        ]
        accepted: list[tuple[float, float]] = []
        for start, end in candidates:
            centre = 0.5 * (start + end - 1)
            # Accept a run either because its centre is close to the prediction, or
            # because the prediction falls inside it - which is what happens where a
            # steep stroke spans many rows in a single column.
            distance = abs(centre - predicted)
            if start - settings.max_jump <= predicted <= end - 1 + settings.max_jump:
                distance = min(distance, 0.0 if start <= predicted <= end - 1 else distance)
            elif distance > settings.max_jump:
                continue
            accepted.append((distance, centre))

        accepted.sort()
        best = accepted[0] if accepted else None

        if stats is not None and best is not None:
            # Two plausible continuations means two strokes run together here, and
            # picking the nearer one is a guess. It is still the best guess available,
            # but it is counted so the caller can say so rather than present a trace
            # that silently swapped curves as if it were certain.
            if len(accepted) > 1 and accepted[1][0] - accepted[0][0] < settings.ambiguous_margin:
                stats["ambiguous"] = stats.get("ambiguous", 0) + 1

            # Curves that overlap outright fuse into a single run, so no second
            # candidate exists to notice and this count stays silent through exactly
            # the stretch where the trace can no longer tell the curves apart. Run
            # thickness looked like the missing tell and is not: a marker drawn on the
            # curve thickens it just as much, so that test flags correct traces as
            # often as wrong ones. Nothing here detects the overlap case - it is a
            # documented limitation rather than a silently-handled one.

        if best is None:
            gap += settings.step
            if gap > settings.max_gap:
                break
            # Coast: keep the prediction alive so a dashed stroke can be rejoined.
            if 0 <= predicted < height:
                centres.append(predicted)
                columns.append(column)
        else:
            gap = 0
            centres.append(best[1])
            columns.append(column)
            found.append((column, best[1]))
        column += direction * settings.step

    return found


def trace_stroke(mask: np.ndarray, column: float, row: float,
                 settings: TraceSettings | None = None,
                 stats: dict | None = None) -> np.ndarray:
    """Trace the stroke passing through (column, row).

    ``mask`` is a boolean ink mask. Returns an (N, 2) array of (column, row) points
    ordered by column, empty when the seed is not on any usable ink.

    Pass a dict as ``stats`` to get back an ``ambiguous`` count: columns where another
    stroke was an equally good continuation, so the choice made there was a guess. It
    is worth surfacing, because a trace that changed curves looks exactly as convincing
    as one that did not.
    """
    settings = settings or TraceSettings()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return np.empty((0, 2), dtype=float)

    height, width = mask.shape
    seed_column = int(round(column))
    seed_row = int(round(row))
    if not (0 <= seed_column < width and 0 <= seed_row < height):
        return np.empty((0, 2), dtype=float)

    # Find something to start on, tolerating a click a few pixels off the stroke and,
    # importantly, a click that lands in one of the gaps of a dotted line - which is
    # most of a dotted line's length, so demanding ink exactly under the cursor would
    # make the tool feel broken on precisely the curves it is most needed for.
    seed = None
    for offset in range(0, settings.seed_search + 1):
        for candidate_column in ({seed_column} if offset == 0
                                 else (seed_column - offset, seed_column + offset)):
            if not (0 <= candidate_column < width):
                continue
            runs = [(s, e) for s, e in runs_in_column(mask, candidate_column)
                    if e - s <= settings.max_run
                    and s - settings.max_jump <= seed_row <= e - 1 + settings.max_jump]
            if runs:
                start, end = min(runs, key=lambda run: min(abs(run[0] - seed_row),
                                                           abs(run[1] - 1 - seed_row)))
                seed = (candidate_column, 0.5 * (start + end - 1))
                break
        if seed is not None:
            break

    if seed is None:
        return np.empty((0, 2), dtype=float)
    seed_column, seed_centre = seed

    left = _walk(mask, seed_column, seed_centre, -1, settings, stats)
    right = _walk(mask, seed_column, seed_centre, +1, settings, stats)

    points = [*reversed(left), (seed_column, seed_centre), *right]
    array = np.array(points, dtype=float)
    return array[np.argsort(array[:, 0])]
