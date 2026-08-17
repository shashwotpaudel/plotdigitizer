"""Pull a hand-placed point onto the ink it was aimed at.

Clicking a curve by hand lands wherever the cursor happened to be - usually a pixel or
two off the stroke, often on its edge rather than its middle. Snapping removes that
error, and it removes the difference between a point placed by hand and one produced by
the automatic extractor, which measures a stroke by the centre of its ink in each
column (see ``_column_centres`` in :mod:`plotdigitizer.detect.extract`). The same
measure is used here so the two agree.

Snapping to a *run centre* rather than to the nearest inked pixel is the important
detail: the nearest pixel is on the near edge of the stroke, which biases every point
towards whichever side the user approached from.
"""

from __future__ import annotations

import numpy as np

from .frame import InkImage

__all__ = ["snap_to_ink", "SnapResult"]


class SnapResult(tuple):
    """(column, row, snapped) - a plain tuple with a readable ``snapped`` flag."""

    __slots__ = ()

    def __new__(cls, column: float, row: float, snapped: bool):
        return super().__new__(cls, (float(column), float(row), bool(snapped)))

    @property
    def column(self) -> float:
        return self[0]

    @property
    def row(self) -> float:
        return self[1]

    @property
    def snapped(self) -> bool:
        return self[2]


def _run_containing(mask_column: np.ndarray, row: int, radius: int) -> tuple[int, int] | None:
    """The contiguous inked run in one column nearest ``row``, as (start, end_exclusive)."""
    height = mask_column.size
    if height == 0:
        return None

    lo = max(0, row - radius)
    hi = min(height, row + radius + 1)
    window = np.flatnonzero(mask_column[lo:hi])
    if window.size == 0:
        return None

    # Nearest inked pixel in the window, then grow to the whole run it belongs to.
    seed = int(window[np.argmin(np.abs(window + lo - row))]) + lo
    start = seed
    while start - 1 >= 0 and mask_column[start - 1]:
        start -= 1
    end = seed + 1
    while end < height and mask_column[end]:
        end += 1
    return start, end


def snap_to_ink(
    ink: InkImage,
    column: float,
    row: float,
    radius: float = 6.0,
    max_run: float = 40.0,
) -> SnapResult:
    """Snap (column, row) onto the centre of the nearest stroke.

    ``radius`` is how far to look, in pixels. ``max_run`` guards against snapping to
    the middle of a large filled region - a legend box or a shaded band - where the
    "centre of the ink" is not a meaningful place for a data point; there the nearest
    inked pixel is used instead.

    Returns the original position unchanged when there is no ink within reach, so a
    click on empty space still places a point exactly where the user asked.
    """
    height, width = ink.shape
    col = int(round(column))
    row_i = int(round(row))
    reach = max(1, int(round(radius)))

    if not (0 <= col < width and 0 <= row_i < height):
        return SnapResult(column, row, False)

    run = _run_containing(ink.mask[:, col], row_i, reach)
    if run is not None:
        start, end = run
        if end - start <= max_run:
            return SnapResult(column, 0.5 * (start + end - 1), True)
        # Too thick to have a meaningful centre: sit on the near edge instead.
        nearest = start if abs(start - row) <= abs(end - 1 - row) else end - 1
        return SnapResult(column, float(nearest), True)

    # Nothing in this column - look around for the nearest ink at all.
    c0, c1 = max(0, col - reach), min(width, col + reach + 1)
    r0, r1 = max(0, row_i - reach), min(height, row_i + reach + 1)
    patch = ink.mask[r0:r1, c0:c1]
    if not patch.any():
        return SnapResult(column, row, False)

    rows, cols = np.nonzero(patch)
    distances = (rows + r0 - row) ** 2 + (cols + c0 - column) ** 2
    best = int(np.argmin(distances))
    best_col = int(cols[best]) + c0

    # Re-centre on the run in whichever column we landed in.
    run = _run_containing(ink.mask[:, best_col], int(rows[best]) + r0, reach)
    if run is not None and run[1] - run[0] <= max_run:
        return SnapResult(float(best_col), 0.5 * (run[0] + run[1] - 1), True)
    return SnapResult(float(best_col), float(rows[best] + r0), True)
