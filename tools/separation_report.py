"""Measure how well series that share a colour are told apart.

The accuracy report pairs detected series with the truth by colour, which is exactly
what cannot work here - every series in these figures may be the same black. Series are
paired by shape instead: each true series is matched to the detected series whose
points lie on it most often.

Three numbers per true series, all measured in pixels against the curve as drawn:

``purity``    the fraction of the matched series' points that lie on *this* curve.
              Low purity means points from a neighbouring curve leaked in, which is
              the failure that matters: a series that silently mixes two curves looks
              just as convincing as a clean one.
``coverage``  the fraction of this curve that the matched series actually reaches.
              Low coverage is a visible, fixable gap rather than a wrong number.
``error``     the mean distance of the pure points from the curve.

A point counts as lying on a curve when it is within ``TOLERANCE_PX`` of it. Where two
curves genuinely coincide, a point on one lies on both, and it is not counted against
either - no method can say which one it belongs to.

Usage:  python tools/separation_report.py [--data tests/samecolour] [--device cpu]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: How far a point may sit from a curve and still be on it. Lines in these figures are
#: 1-2 px wide and markers 6-9 px across, so 3 px is "on the stroke", not "nearby".
TOLERANCE_PX = 3.0
#: Coverage is sampled along the curve at this spacing.
COVERAGE_STEP_PX = 2.0


def _polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest segment of a polyline."""
    if points.size == 0:
        return np.empty(0)
    if polyline.shape[0] == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    a = polyline[:-1][None, :, :]
    b = polyline[1:][None, :, :]
    p = points[:, None, :]
    ab = b - a
    denom = np.maximum((ab ** 2).sum(axis=2), 1e-12)
    t = np.clip(((p - a) * ab).sum(axis=2) / denom, 0.0, 1.0)
    nearest = a + t[..., None] * ab
    return np.linalg.norm(p - nearest, axis=2).min(axis=1)


def _point_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty(0)
    if targets.size == 0:
        return np.full(points.shape[0], np.inf)
    return np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=2).min(axis=1)


def _inside(points: np.ndarray, box: dict, margin: float = 1.0) -> np.ndarray:
    return ((points[:, 0] >= box["left"] - margin) & (points[:, 0] <= box["right"] + margin)
            & (points[:, 1] >= box["top"] - margin) & (points[:, 1] <= box["bottom"] + margin))


@dataclass
class TruthCurve:
    name: str
    scatter: bool
    pixels: np.ndarray          # (N, 2) column,row - clipped to the axes box
    samples: np.ndarray         # points along the curve used for coverage

    def distance(self, points: np.ndarray) -> np.ndarray:
        if self.scatter:
            return _point_distance(points, self.pixels)
        return _polyline_distance(points, self.pixels)


def _densify(polyline: np.ndarray, step: float) -> np.ndarray:
    if polyline.shape[0] < 2:
        return polyline
    out = [polyline[0]]
    for a, b in zip(polyline[:-1], polyline[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for t in np.arange(1, n + 1) / n:
            out.append(a + t * (b - a))
    return np.asarray(out)


def truth_curves(record: dict) -> list[TruthCurve]:
    curves = []
    box = record["axes_box"]
    for series in record["series"]:
        pixels = np.column_stack([series["px"], series["py"]]).astype(float)
        pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
        pixels = pixels[_inside(pixels, box)]
        scatter = series["linestyle"] == "none"
        samples = pixels if scatter else _densify(pixels, COVERAGE_STEP_PX)
        curves.append(TruthCurve(series["name"], scatter, pixels, samples))
    return curves


@dataclass
class SeriesScore:
    truth: str
    detected: str | None
    purity: float
    coverage: float
    error_px: float
    points: int


def _score_pair(curve: TruthCurve, points: np.ndarray) -> SeriesScore:
    if points.size == 0:
        return SeriesScore(curve.name, None, 0.0, 0.0, float("nan"), 0)
    distance = curve.distance(points)
    on = distance <= TOLERANCE_PX
    purity = float(on.mean())
    reach = _point_distance(curve.samples, points)
    coverage = float((reach <= TOLERANCE_PX).mean()) if curve.samples.size else 0.0
    error = float(distance[on].mean()) if on.any() else float("nan")
    return SeriesScore(curve.name, "", purity, coverage, error, int(points.shape[0]))


def score_figure(record: dict, detected: list) -> list[SeriesScore]:
    """Pair truth with detection to maximise total purity, then score each pair.

    ``detected`` is a list of (name, pixel_points) pairs.
    """
    curves = truth_curves(record)
    n_truth, n_det = len(curves), len(detected)
    table = np.zeros((n_truth, max(n_det, 1)))
    cache: dict[tuple[int, int], SeriesScore] = {}
    for i, curve in enumerate(curves):
        for j, (_, points) in enumerate(detected):
            score = _score_pair(curve, np.asarray(points, dtype=float).reshape(-1, 2))
            cache[(i, j)] = score
            # Purity and coverage together, so a tiny pure fragment does not beat the
            # series that actually carries the curve.
            table[i, j] = score.purity * score.coverage

    best_total, best_assignment = -1.0, None
    slots = list(range(n_det)) + [None] * n_truth
    if n_det <= 8:
        for perm in itertools.permutations(slots, n_truth):
            if len({p for p in perm if p is not None}) != sum(p is not None for p in perm):
                continue
            total = sum(table[i, j] for i, j in enumerate(perm) if j is not None)
            if total > best_total:
                best_total, best_assignment = total, perm
    else:  # pragma: no cover - greedy fallback for very over-split results
        taken: set[int] = set()
        best_assignment = []
        for i in range(n_truth):
            options = [j for j in range(n_det) if j not in taken]
            j = max(options, key=lambda k: table[i, k]) if options else None
            if j is not None:
                taken.add(j)
            best_assignment.append(j)

    scores = []
    for i, j in enumerate(best_assignment):
        if j is None:
            scores.append(SeriesScore(curves[i].name, None, 0.0, 0.0, float("nan"), 0))
            continue
        score = cache[(i, j)]
        score.detected = detected[j][0]
        scores.append(score)
    return scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data", default=str(ROOT / "tests" / "samecolour"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    from plotdigitizer.image_io import load_image
    from plotdigitizer.pipeline import AutoDigitizer

    data_dir = Path(args.data)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    digitizer = AutoDigitizer(device=args.device)

    header = f"{'figure':22s} {'series':>7s} {'purity':>7s} {'cover':>7s} {'err px':>7s}  worst"
    print(header)
    print("-" * (len(header) + 20))
    counts_ok = 0
    purities, coverages = [], []
    for record in manifest["figures"]:
        result = digitizer.run(load_image(data_dir / record["file"]))
        detected = [(s.name, s.pixel_points) for s in result.series]
        scores = score_figure(record, detected)
        n_ok = len(result.series) == len(record["series"])
        counts_ok += n_ok
        purity = min(s.purity for s in scores)
        coverage = min(s.coverage for s in scores)
        error = np.nanmean([s.error_px for s in scores])
        purities.extend(s.purity for s in scores)
        coverages.extend(s.coverage for s in scores)
        worst = min(scores, key=lambda s: s.purity * s.coverage)
        flag = "" if n_ok else " <-- count"
        print(f"{record['name']:22s} {len(result.series):>3d}/{len(record['series']):<3d} "
              f"{purity:7.2f} {coverage:7.2f} {error:7.2f}  {worst.truth}{flag}")
        if args.verbose:
            for s in scores:
                print(f"      {s.truth:14s} <- {str(s.detected):10s} purity {s.purity:.2f} "
                      f"cover {s.coverage:.2f} err {s.error_px:.2f}px  ({s.points} pts)")
    print()
    print(f"series count right: {counts_ok}/{len(manifest['figures'])} figures")
    print(f"mean purity {np.mean(purities):.3f}, mean coverage {np.mean(coverages):.3f} "
          f"over {len(purities)} true series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
