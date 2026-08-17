"""Finding the points that do not belong to the curve they are sitting in.

When a trace crosses onto a neighbouring curve it does not drift - it steps, holds the
wrong line for a stretch, and steps back. Against the local trend those points stand out
sharply, which makes them findable without knowing anything about the figure.

The scale is a median absolute deviation rather than a standard deviation, for a reason
that decides whether this works at all: the spikes being hunted are themselves large
residuals, and they would inflate a standard deviation enough to hide inside it. A
median-based scale is unmoved by them, so the threshold stays where the *clean* data
puts it.

Nothing here modifies anything. It returns indices, so the caller can show a selection
and let the user look before deciding.
"""

from __future__ import annotations

import numpy as np

__all__ = ["select_outliers", "residuals_from_trend", "displaced_runs"]

#: Scales the MAD to be comparable with a standard deviation for normal data.
_MAD_TO_SIGMA = 1.4826


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median over a centred window, with the ends handled by edge padding."""
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    padded = np.pad(values, window // 2, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strided, axis=1)


def residuals_from_trend(points: np.ndarray, window: int = 9) -> np.ndarray:
    """How far each point sits from the local trend of its neighbours, in y.

    Points are ordered by x first, because "local" means neighbouring along the curve,
    not neighbouring in whatever order the points happen to be stored.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=float)

    order = np.argsort(points[:, 0], kind="stable")
    ys = points[order, 1]
    trend = _rolling_median(ys, min(window, ys.size if ys.size % 2 else ys.size - 1))

    residual = np.empty(points.shape[0], dtype=float)
    residual[order] = ys - trend
    return residual


#: Below this many points there is no "local trend" to deviate from - the window would
#: span most of the series, so the comparison is against the series as a whole.
MIN_POINTS = 15
#: Floor on the robust scale, in pixels. On a trace that follows its stroke exactly the
#: MAD is zero, and without a floor the scale would have to be taken from the deviations
#: themselves - which means scaling the threshold by the very outliers being hunted, and
#: reliably flagging the second-worst points instead of the worst.
_SCALE_FLOOR_PX = 0.5
#: However cleanly a curve fits its trend, a point has to be at least this far off it to
#: be worth selecting. A rolling median cannot follow curvature exactly and drifts by a
#: pixel or so either side of a bend - without this floor those artefacts get flagged,
#: and the points on the shoulders of a real problem get selected along with it.
_MIN_DEVIATION_PX = 3.0


def _robust_scale(values: np.ndarray) -> float:
    """MAD-based spread, floored so a perfectly clean trace still has a usable scale."""
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(mad * _MAD_TO_SIGMA, _SCALE_FLOOR_PX)


def displaced_runs(points: np.ndarray, sensitivity: float = 3.0,
                   max_fraction: float = 0.4) -> np.ndarray:
    """Indices of stretches that stepped away from the curve and later stepped back.

    This is the shape a wandering trace actually makes. It does not drift one point at
    a time - it jumps onto a neighbouring curve, follows *that* faithfully for a while,
    and jumps back. Deviation from a local trend cannot see the middle of such a run: a
    rolling median with enough displaced points inside its window simply follows them,
    so the run becomes the trend and only its two ends look unusual.

    What stays visible is the pair of jumps. Anomalously large steps between adjacent
    points are found first, then an up-step matched with a later down-step of similar
    size marks everything between them as displaced. A steadily steep curve is not
    caught, because there every step is large and none of them is anomalous.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    n = points.shape[0]
    if n < MIN_POINTS:
        return np.empty(0, dtype=int)

    order = np.argsort(points[:, 0], kind="stable")
    ys = points[order, 1]
    steps = np.diff(ys)
    scale = _robust_scale(steps)
    threshold = sensitivity * scale
    jumps = [(i, float(steps[i])) for i in np.flatnonzero(np.abs(steps) > threshold)]
    if not jumps:
        return np.empty(0, dtype=int)

    flagged: set[int] = set()
    used: set[int] = set()
    for a, (start, rise) in enumerate(jumps):
        if a in used:
            continue
        for b in range(a + 1, len(jumps)):
            if b in used:
                continue
            end, fall = jumps[b]
            if np.sign(fall) == np.sign(rise):
                continue
            # The excursion has to come back to roughly where it left, or it is a
            # genuine feature of the curve rather than a detour off it.
            if abs(abs(fall) - abs(rise)) > 0.6 * max(abs(rise), abs(fall)):
                continue
            if end - start > max_fraction * n:
                continue
            flagged.update(range(start + 1, end + 1))
            used.update({a, b})
            break

    if not flagged:
        return np.empty(0, dtype=int)
    return np.sort(order[np.array(sorted(flagged), dtype=int)])


def select_outliers(points: np.ndarray, sensitivity: float = 3.0,
                    window: int = 9) -> np.ndarray:
    """Indices of points that do not belong to the curve they sit in.

    Combines two views, because a wandering trace produces both: isolated points far
    from the local trend, and whole stretches that stepped onto another curve and back.
    Neither test finds the other's case.

    ``sensitivity`` is the number of robust standard deviations beyond which something
    is flagged: lower catches more. A clean curve returns an empty array rather than
    always surrendering its worst few points.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.shape[0] < MIN_POINTS:
        # A sparse scatter has no local trend to stand out from: the window would span
        # most of the series, so "deviates from its neighbours" would just mean "is not
        # near the middle". Declining is better than confidently selecting the wrong
        # points on data this tool cannot reason about.
        return np.empty(0, dtype=int)

    residual = residuals_from_trend(points, window)
    scale = _robust_scale(residual)
    threshold = max(sensitivity * scale, _MIN_DEVIATION_PX)
    spikes = np.flatnonzero(np.abs(residual) > threshold)
    runs = displaced_runs(points, sensitivity)
    return np.unique(np.concatenate([spikes, runs])).astype(int)
