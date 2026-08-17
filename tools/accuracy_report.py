"""Measure end-to-end digitization accuracy against the ground-truth corpus.

For every figure this runs the full pipeline and compares the recovered numbers with
what was actually plotted. Errors are reported as a percentage of the axis range, which
is the honest unit: an absolute error of 0.5 means nothing without knowing whether the
axis spans 1 or 1000.

Series are matched to the truth by colour, then compared by interpolating the extracted
curve at the true x positions - a digitizer is not expected to return the same *number*
of points as the original data, only to trace the same shape.

Usage:  python tools/accuracy_report.py [--device auto|cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotdigitizer.image_io import load_image          # noqa: E402
from plotdigitizer.pipeline import AutoDigitizer       # noqa: E402


def _hex_to_rgb(text: str) -> np.ndarray:
    text = text.lstrip("#")
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _match_series(detected, truth_series):
    """Pair each true series with the detected one closest to it in colour."""
    pairs = []
    available = list(detected)
    for truth in truth_series:
        if not available:
            pairs.append((truth, None))
            continue
        want = _hex_to_rgb(truth["color"])
        best = min(available, key=lambda s: np.linalg.norm(np.array(s.color, dtype=float) - want))
        available.remove(best)
        pairs.append((truth, best))
    return pairs


def _axis_space(values: np.ndarray, scale: str) -> np.ndarray:
    """Map values into the space the axis is linear in.

    Errors have to be measured here, not in raw data units. On a log axis spanning four
    decades, an absolute error of 0.5 is catastrophic near the bottom and invisible near
    the top; expressed as a fraction of the *plotted* extent it means the same thing
    everywhere, and matches what the eye would call "off by that much".
    """
    values = np.asarray(values, dtype=float)
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(values > 0, values, np.nan))
    return values


def _normalised(values: np.ndarray, limits, scale: str) -> np.ndarray:
    lo, hi = _axis_space(np.asarray(limits, dtype=float), scale)
    span = hi - lo
    return (_axis_space(values, scale) - lo) / (span if abs(span) > 1e-30 else 1.0)


def _curve_error(detected_xy, true_x, true_y, record) -> tuple[float, float]:
    """Mean vertical error as a percentage of the plotted y extent."""
    if detected_xy.shape[0] < 2:
        return float("nan"), float("nan")
    dx = _normalised(detected_xy[:, 0], record["xlim"], record["xscale"])
    dy = _normalised(detected_xy[:, 1], record["ylim"], record["yscale"])
    tx = _normalised(true_x, record["xlim"], record["xscale"])
    ty = _normalised(true_y, record["ylim"], record["yscale"])

    order = np.argsort(dx)
    dx, dy = dx[order], dy[order]
    inside = (tx >= dx.min()) & (tx <= dx.max()) & np.isfinite(tx) & np.isfinite(ty)
    if inside.sum() < 2:
        return float("nan"), float("nan")
    error = np.abs(np.interp(tx[inside], dx, dy) - ty[inside]) * 100.0
    return float(np.nanmean(error)), float(np.nanmax(error))


def _scatter_error(detected_xy, true_x, true_y, record) -> tuple[float, float]:
    """Nearest-neighbour distance from each true point to a detected one, in % of extent."""
    if detected_xy.shape[0] == 0:
        return float("nan"), float("nan")
    detected = np.column_stack([
        _normalised(detected_xy[:, 0], record["xlim"], record["xscale"]),
        _normalised(detected_xy[:, 1], record["ylim"], record["yscale"]),
    ])
    truth = np.column_stack([
        _normalised(true_x, record["xlim"], record["xscale"]),
        _normalised(true_y, record["ylim"], record["yscale"]),
    ])
    good = np.all(np.isfinite(detected), axis=1)
    detected = detected[good]
    if detected.shape[0] == 0:
        return float("nan"), float("nan")
    distances = np.linalg.norm(truth[:, None, :] - detected[None, :, :], axis=2)
    nearest = distances.min(axis=1) * 100.0
    return float(np.nanmean(nearest)), float(np.nanmax(nearest))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data", default=str(ROOT / "tests" / "data"))
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    digitizer = AutoDigitizer(device=args.device)
    print(f"device: {digitizer.backend.describe()}\n")
    header = (f"{'figure':22s} {'axes':>16s} {'series':>7s} {'pt err%':>8s} "
              f"{'max%':>7s} {'pts':>5s} {'ms':>6s}")
    print(header)
    print("-" * len(header))

    all_mean_errors: list[float] = []
    axis_failures: list[str] = []
    worst: list[tuple[float, str]] = []

    for record in manifest["figures"]:
        image = load_image(data_dir / record["file"])
        result = digitizer.run(image)

        # Check the calibration in pixel space: where does it think the true axis
        # limits are drawn? One pixel is the finest a click could ever be, so that is
        # the only fair tolerance, and it means the same on linear and log axes.
        box = record["axes_box"]
        x_error = max(abs(result.calibration.x.to_pixel(record["xlim"][0]) - box["left"]),
                      abs(result.calibration.x.to_pixel(record["xlim"][1]) - box["right"]))
        y_error = max(abs(result.calibration.y.to_pixel(record["ylim"][0]) - box["bottom"]),
                      abs(result.calibration.y.to_pixel(record["ylim"][1]) - box["top"]))
        x_ok, y_ok = x_error <= 1.5, y_error <= 1.5
        axes_note = ("x" if x_ok else "X") + ("y" if y_ok else "Y")
        if not (x_ok and y_ok):
            axis_failures.append(f"{record['name']} ({axes_note}: {x_error:.1f}/{y_error:.1f}px)")

        means, maxes = [], []
        for truth, detected in _match_series(result.series, record["series"]):
            if detected is None or detected.data_points.shape[0] == 0:
                means.append(float("nan"))
                continue
            true_x = np.asarray(truth["x"], dtype=float)
            true_y = np.asarray(truth["y"], dtype=float)
            if truth["linestyle"] == "none":
                mean, mx = _scatter_error(detected.data_points, true_x, true_y, record)
            else:
                mean, mx = _curve_error(detected.data_points, true_x, true_y, record)
            means.append(mean)
            maxes.append(mx)

        mean_error = float(np.nanmean(means)) if means else float("nan")
        max_error = float(np.nanmax(maxes)) if maxes else float("nan")
        if np.isfinite(mean_error):
            all_mean_errors.append(mean_error)
            worst.append((mean_error, record["name"]))

        print(f"{record['name']:22s} {axes_note:>16s} "
              f"{len(result.series):d}/{len(record['series']):d}".ljust(48)
              + f"{mean_error:8.3f} {max_error:7.3f} {result.total_points:5d} "
                f"{result.elapsed_seconds * 1000:6.0f}")

        for warning in result.warnings:
            print(f"    ! {warning}")

    print()
    if all_mean_errors:
        print(f"mean point error across corpus: {np.mean(all_mean_errors):.3f}% of axis range")
        worst.sort(reverse=True)
        print("worst figures: " + ", ".join(f"{n} ({e:.2f}%)" for e, n in worst[:3]))
    print(f"axis calibration: {len(manifest['figures']) - len(axis_failures)}"
          f"/{len(manifest['figures'])} figures exact")
    if axis_failures:
        print("  failed: " + ", ".join(axis_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
