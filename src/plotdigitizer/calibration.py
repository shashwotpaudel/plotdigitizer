"""Mapping between pixel coordinates and data coordinates.

An axis is stored the way the user sees it in the UI: two reference points, each a
pixel position paired with the data value at that position, plus a scale type. That
is exactly what the two draggable handles per axis represent, so dragging a handle
is a direct edit of the calibration rather than a re-fit.

Every scale works by linearising the value: ``t(value)`` is affine in pixel position.
For a linear axis ``t`` is the identity, for a log axis it is the logarithm, and so on.
All the fitting code therefore only ever has to solve a straight-line problem.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = [
    "AxisScale",
    "AxisCalibration",
    "AxisFit",
    "Calibration",
    "fit_axis",
]


class AxisScale(str, Enum):
    """Supported axis types, matching the options offered by plotdigitizer.com."""

    LINEAR = "linear"
    LOG10 = "log10"
    LOGE = "loge"
    RECIPROCAL = "reciprocal"
    DATE = "date"

    @property
    def label(self) -> str:
        return {
            AxisScale.LINEAR: "Linear",
            AxisScale.LOG10: "Log10",
            AxisScale.LOGE: "Loge (ln)",
            AxisScale.RECIPROCAL: "Reciprocal (1/x)",
            AxisScale.DATE: "Date / Time",
        }[self]

    @property
    def requires_positive(self) -> bool:
        """True when a value of zero or less cannot be represented on this scale."""
        return self in (AxisScale.LOG10, AxisScale.LOGE, AxisScale.RECIPROCAL)


# Date values travel through the linear machinery as seconds since the epoch.
_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def _to_seconds(value) -> float:
    if isinstance(value, _dt.datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
        return (aware - _EPOCH).total_seconds()
    return float(value)


def linearise(values, scale: AxisScale) -> np.ndarray:
    """Map data values into the space where they are affine in pixel position."""
    if scale is AxisScale.DATE:
        arr = np.asarray([_to_seconds(v) for v in np.atleast_1d(values)], dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        if scale is AxisScale.LINEAR or scale is AxisScale.DATE:
            out = arr.astype(float)
        elif scale is AxisScale.LOG10:
            out = np.log10(arr)
        elif scale is AxisScale.LOGE:
            out = np.log(arr)
        elif scale is AxisScale.RECIPROCAL:
            out = 1.0 / arr
        else:  # pragma: no cover - exhaustive over the enum
            raise ValueError(f"unsupported scale {scale!r}")

    out = np.where(np.isfinite(out), out, np.nan)
    return out


def delinearise(values, scale: AxisScale) -> np.ndarray:
    """Inverse of :func:`linearise`."""
    arr = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if scale is AxisScale.LINEAR or scale is AxisScale.DATE:
            out = arr
        elif scale is AxisScale.LOG10:
            out = np.power(10.0, arr)
        elif scale is AxisScale.LOGE:
            out = np.exp(arr)
        elif scale is AxisScale.RECIPROCAL:
            out = 1.0 / arr
        else:  # pragma: no cover
            raise ValueError(f"unsupported scale {scale!r}")
    return np.where(np.isfinite(out), out, np.nan)


@dataclass
class AxisCalibration:
    """One axis: two (pixel, value) reference points and a scale.

    ``p1``/``p2`` are pixel coordinates along the axis - column for an x axis, row for
    a y axis. Rows increase downwards, which is why a normal y axis ends up with a
    negative slope; nothing here needs to care, the arithmetic handles it.
    """

    p1: float
    v1: float
    p2: float
    v2: float
    scale: AxisScale = AxisScale.LINEAR
    # Diagnostics from automatic fitting; None when the user set the axis by hand.
    fit: "AxisFit | None" = None

    def __post_init__(self) -> None:
        if isinstance(self.scale, str):
            self.scale = AxisScale(self.scale)

    @property
    def is_valid(self) -> bool:
        """False when the two reference points cannot define a mapping."""
        if abs(self.p2 - self.p1) < 1e-9:
            return False
        t1, t2 = linearise([self.v1, self.v2], self.scale)
        if not (math.isfinite(t1) and math.isfinite(t2)):
            return False
        return abs(t2 - t1) > 1e-12

    def _coefficients(self) -> tuple[float, float]:
        """Return (a, b) such that ``t(value) = a * pixel + b``."""
        t1, t2 = linearise([self.v1, self.v2], self.scale)
        a = (t2 - t1) / (self.p2 - self.p1)
        b = t1 - a * self.p1
        return float(a), float(b)

    def to_data(self, pixels):
        """Pixel coordinate(s) -> data value(s)."""
        a, b = self._coefficients()
        t = a * np.asarray(pixels, dtype=float) + b
        out = delinearise(t, self.scale)
        return float(out) if np.ndim(pixels) == 0 else out

    def to_pixel(self, values):
        """Data value(s) -> pixel coordinate(s)."""
        a, b = self._coefficients()
        t = linearise(values, self.scale)
        out = (t - b) / a
        if np.ndim(values) == 0:
            return float(np.atleast_1d(out)[0])
        return out

    def to_dict(self) -> dict:
        return {
            "p1": self.p1, "v1": self.v1, "p2": self.p2, "v2": self.v2,
            "scale": self.scale.value,
            "fit": self.fit.to_dict() if self.fit else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AxisCalibration":
        return cls(
            p1=float(d["p1"]), v1=float(d["v1"]),
            p2=float(d["p2"]), v2=float(d["v2"]),
            scale=AxisScale(d.get("scale", "linear")),
            fit=AxisFit.from_dict(d["fit"]) if d.get("fit") else None,
        )


@dataclass
class AxisFit:
    """Diagnostics describing how well an automatic calibration matched the ticks."""

    scale: AxisScale
    n_ticks: int
    n_inliers: int
    rms_pixel_error: float
    r_squared: float
    #: Ticks the fit rejected, as (pixel, value) - shown to the user as suspicious labels.
    outliers: list[tuple[float, float]] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """0-1 score combining fit quality with how many ticks agreed."""
        if self.n_inliers < 2:
            return 0.0
        # A sub-pixel RMS error is a perfect fit; 5 px is worthless.
        accuracy = max(0.0, 1.0 - self.rms_pixel_error / 5.0)
        support = min(1.0, self.n_inliers / 4.0)
        agreement = self.n_inliers / max(1, self.n_ticks)
        return float(accuracy * 0.5 + support * 0.25 + agreement * 0.25)

    def to_dict(self) -> dict:
        return {
            "scale": self.scale.value, "n_ticks": self.n_ticks, "n_inliers": self.n_inliers,
            "rms_pixel_error": self.rms_pixel_error, "r_squared": self.r_squared,
            "outliers": [list(o) for o in self.outliers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AxisFit":
        return cls(
            scale=AxisScale(d["scale"]), n_ticks=int(d["n_ticks"]), n_inliers=int(d["n_inliers"]),
            rms_pixel_error=float(d["rms_pixel_error"]), r_squared=float(d["r_squared"]),
            outliers=[tuple(o) for o in d.get("outliers", [])],
        )


@dataclass
class Calibration:
    """The pair of axes needed to turn a pixel into a data point."""

    x: AxisCalibration
    y: AxisCalibration

    @property
    def is_valid(self) -> bool:
        return self.x.is_valid and self.y.is_valid

    def to_data(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) array of (column, row) pixels -> (N, 2) array of (x, y) values."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return np.column_stack([
            np.atleast_1d(self.x.to_data(pts[:, 0])),
            np.atleast_1d(self.y.to_data(pts[:, 1])),
        ])

    def to_pixel(self, values: np.ndarray) -> np.ndarray:
        vals = np.atleast_2d(np.asarray(values, dtype=float))
        return np.column_stack([
            np.atleast_1d(self.x.to_pixel(vals[:, 0])),
            np.atleast_1d(self.y.to_pixel(vals[:, 1])),
        ])

    def to_dict(self) -> dict:
        return {"x": self.x.to_dict(), "y": self.y.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(x=AxisCalibration.from_dict(d["x"]), y=AxisCalibration.from_dict(d["y"]))


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------


def _fit_line_ransac(
    pixels: np.ndarray,
    targets: np.ndarray,
    tolerance_px: float,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Fit ``targets = a * pixels + b``, rejecting outliers.

    The tolerance is expressed in pixels rather than in target units so that it means
    the same thing on a linear and a log axis. Returns (a, b, inlier_mask).
    """
    n = pixels.size
    best_mask = np.ones(n, dtype=bool)
    if n < 3:
        a, b = np.polyfit(pixels, targets, 1)
        return float(a), float(b), best_mask

    best_score = -1.0
    # Exhaustive over pairs while that is cheap; ticks are few, so this is deterministic.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > 400:
        idx = rng.choice(len(pairs), size=400, replace=False)
        pairs = [pairs[k] for k in idx]

    for i, j in pairs:
        dp = pixels[j] - pixels[i]
        dt = targets[j] - targets[i]
        if abs(dp) < 1e-9 or abs(dt) < 1e-12:
            continue
        a = dt / dp
        b = targets[i] - a * pixels[i]
        # Residual measured back in pixel space.
        predicted_px = (targets - b) / a
        residual = np.abs(predicted_px - pixels)
        mask = residual <= tolerance_px
        # Prefer more inliers, break ties on tighter residuals.
        score = mask.sum() - float(residual[mask].mean()) / (tolerance_px * 100.0) if mask.any() else 0.0
        if score > best_score:
            best_score, best_mask = score, mask

    if best_mask.sum() >= 2:
        a, b = np.polyfit(pixels[best_mask], targets[best_mask], 1)
    else:
        a, b = np.polyfit(pixels, targets, 1)
        best_mask = np.ones(n, dtype=bool)
    return float(a), float(b), best_mask


def _evaluate(pixels: np.ndarray, values: np.ndarray, scale: AxisScale,
              tolerance_px: float, rng: np.random.Generator) -> tuple[AxisCalibration, AxisFit] | None:
    """Try one scale; return the calibration and its diagnostics, or None if unusable."""
    targets = linearise(values, scale)
    ok = np.isfinite(targets)
    if ok.sum() < 2:
        return None
    px, tv, vv = pixels[ok], targets[ok], values[ok]

    a, b, mask = _fit_line_ransac(px, tv, tolerance_px, rng)
    if abs(a) < 1e-15 or not np.isfinite(a):
        return None

    predicted_px = (tv - b) / a
    residual = predicted_px - px
    rms = float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else float("inf")

    # R^2 in the linearised space, over the inliers.
    t_in = tv[mask]
    fitted = a * px[mask] + b
    ss_res = float(np.sum((t_in - fitted) ** 2))
    ss_tot = float(np.sum((t_in - t_in.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    # Materialise two reference points at the extreme inlier ticks, using the fitted
    # line rather than the raw labels so a slightly-off label cannot skew the mapping.
    inlier_px = px[mask]
    p1, p2 = float(inlier_px.min()), float(inlier_px.max())
    if abs(p2 - p1) < 1e-6:
        return None
    v1 = float(delinearise(a * p1 + b, scale))
    v2 = float(delinearise(a * p2 + b, scale))
    if not (math.isfinite(v1) and math.isfinite(v2)):
        return None

    fit = AxisFit(
        scale=scale,
        n_ticks=int(pixels.size),
        n_inliers=int(mask.sum()),
        rms_pixel_error=rms,
        r_squared=float(r2),
        outliers=[(float(p), float(v)) for p, v in zip(px[~mask], vv[~mask])],
    )
    return AxisCalibration(p1=p1, v1=v1, p2=p2, v2=v2, scale=scale, fit=fit), fit


def fit_axis(
    pixels,
    values,
    scales: "list[AxisScale] | None" = None,
    tolerance_px: float = 2.0,
    seed: int = 0,
) -> AxisCalibration | None:
    """Fit an axis to detected (pixel, value) tick pairs, choosing the scale automatically.

    Candidate scales are compared by how well each predicts the tick *pixel* positions,
    which is a fair common ground: comparing residuals in value space would always
    flatter whichever transform compressed the values most.

    Returns None when there is not enough usable information to define an axis.
    """
    pixels = np.asarray(pixels, dtype=float)
    values = np.asarray(values, dtype=float)
    if pixels.size != values.size:
        raise ValueError("pixels and values must have the same length")
    if pixels.size < 2:
        return None

    order = np.argsort(pixels)
    pixels, values = pixels[order], values[order]

    if scales is None:
        scales = [AxisScale.LINEAR]
        # Two points lie exactly on *every* candidate scale, so a curved one picked
        # from two ticks is not a finding, it is a coin toss - and getting it wrong
        # distorts every value between them. Only look past linear with three or more.
        if pixels.size >= 3:
            scales.append(AxisScale.LOG10)
            # A reciprocal axis is only worth testing when nothing crosses zero.
            if np.all(values > 0) or np.all(values < 0):
                scales.append(AxisScale.RECIPROCAL)

    rng = np.random.default_rng(seed)
    best: tuple[AxisCalibration, AxisFit] | None = None
    for scale in scales:
        if scale.requires_positive and not np.all(values[np.isfinite(values)] > 0):
            continue
        result = _evaluate(pixels, values, scale, tolerance_px, rng)
        if result is None:
            continue
        if best is None:
            best = result
            continue

        # More inliers wins. On a tie the residual decides, but a curved scale has to
        # beat linear by a visible margin rather than by floating-point noise: linear
        # is tried first and is the safe answer, so an immeasurably better log fit is
        # not evidence of a log axis.
        _, best_fit = best
        _, fit = result
        if fit.n_inliers > best_fit.n_inliers:
            best = result
        elif (fit.n_inliers == best_fit.n_inliers
              and fit.rms_pixel_error < best_fit.rms_pixel_error - 0.05):
            best = result

    return best[0] if best else None


def calibration_from_corners(
    x1_px: float, x1_val: float, x2_px: float, x2_val: float,
    y1_px: float, y1_val: float, y2_px: float, y2_val: float,
    x_scale: AxisScale = AxisScale.LINEAR,
    y_scale: AxisScale = AxisScale.LINEAR,
) -> Calibration:
    """Build a calibration from the four handles the user drags in the UI."""
    return Calibration(
        x=AxisCalibration(x1_px, x1_val, x2_px, x2_val, x_scale),
        y=AxisCalibration(y1_px, y1_val, y2_px, y2_val, y_scale),
    )
