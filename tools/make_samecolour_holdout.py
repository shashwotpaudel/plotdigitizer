"""Held-out figures for same-colour separation, written after the method was fixed.

Every threshold in ``detect/separate.py`` was chosen while looking at the figures of
``tools/make_samecolour.py``. These were written afterwards and are never used for
tuning, so they measure whether the approach generalises rather than how well its
constants fit the development set. They deliberately differ from it:

  * four line styles at once, thicker lines, a serif font and a higher resolution,
  * lines carrying cross and plus markers, which have nothing to fill,
  * a dense black scatter in four shapes, including diamonds,
  * log-log power laws that cross,
  * JPEG compression,
  * a small, low-resolution figure,
  * dark grey and black together, one solid and one dashed in each.

Usage:  python tools/make_samecolour_holdout.py [outdir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_plots import _finish, _new_fig  # noqa: E402

RNG = np.random.default_rng(20260917)
INK = "#000000"


def _line(name, x, y, linestyle, color=INK, marker="none"):
    return dict(name=name, x=x, y=y, color=color, marker=marker, linestyle=linestyle)


def fig_four_styles(outdir):
    """Solid, dashed, dotted and dash-dot, crossing one another, in a serif face."""
    with plt.rc_context({"font.family": "serif"}):
        fig, ax = _new_fig(figsize=(6.0, 4.2), dpi=120)
        x = np.linspace(0, 6, 300)
        ys = [2 + 0.9 * x, 7 - 0.7 * x, 4 + 1.5 * np.sin(1.1 * x), 3 + 0.12 * x ** 2]
        styles = ["-", "--", ":", "-."]
        for y, ls in zip(ys, styles):
            ax.plot(x, y, color=INK, linestyle=ls, linewidth=2.0)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 9)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("amplitude")
        return _finish(fig, ax, "hs_four_styles", outdir,
                       [_line(ls, x, y, ls) for y, ls in zip(ys, styles)],
                       notes="four black styles crossing, serif, 120 dpi, lw 2")


def fig_cross_markers(outdir):
    """Lines carrying + and x markers: glyphs with no inside to fill."""
    fig, ax = _new_fig()
    x = np.linspace(0, 20, 120)
    ys = [10 + 0.8 * x, 30 - 0.6 * x]
    marks = ["+", "x"]
    for y, mk in zip(ys, marks):
        ax.plot(x, y, color=INK, linewidth=1.0, marker=mk, markevery=10, markersize=8)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 35)
    return _finish(fig, ax, "hs_cross_markers", outdir,
                   [_line(f"line {m}", x, y, "-", marker=m) for y, m in zip(ys, marks)],
                   notes="solid lines with + and x markers, crossing")


def fig_dense_shapes(outdir):
    """A dense black scatter in four shapes, clouds overlapping."""
    fig, ax = _new_fig()
    specs = [("o", "none", 2.0), ("s", "none", 3.2), ("^", INK, 4.4), ("D", "none", 5.6)]
    series = []
    for marker, face, level in specs:
        x = np.sort(RNG.uniform(0.3, 9.7, 25))
        y = level + 0.25 * np.sin(x) + RNG.normal(0, 0.35, x.size)
        ax.plot(x, y, linestyle="none", marker=marker, markersize=6,
                markerfacecolor=face, markeredgecolor=INK, markeredgewidth=1.0)
        series.append(dict(name=marker, x=x, y=y, color=INK, marker=marker, linestyle="none"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    return _finish(fig, ax, "hs_dense_shapes", outdir, series,
                   notes="four marker shapes, 25 each, overlapping clouds")


def fig_loglog_power(outdir):
    """Power laws on log-log axes, two crossing and one dashed."""
    fig, ax = _new_fig()
    x = np.geomspace(1, 1000, 300)
    ys = [2.0 * x ** 0.5, 0.2 * x ** 0.9, 50 * x ** -0.3]
    styles = ["-", "-", "--"]
    for y, ls in zip(ys, styles):
        ax.loglog(x, y, color=INK, linestyle=ls, linewidth=1.2)
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 1000)
    return _finish(fig, ax, "hs_loglog_power", outdir,
                   [_line(f"p{i}", x, y, ls) for i, (y, ls) in enumerate(zip(ys, styles))],
                   notes="log-log power laws, two solid crossing, one dashed")


def fig_small(outdir):
    """A small figure at low resolution: every stroke is one or two pixels."""
    fig, ax = _new_fig(figsize=(4.0, 3.0), dpi=80)
    x = np.linspace(0, 4, 200)
    ys = [1 + x, 5 - 0.8 * x, 2.5 + np.cos(1.5 * x)]
    styles = ["-", "--", ":"]
    for y, ls in zip(ys, styles):
        ax.plot(x, y, color=INK, linestyle=ls, linewidth=1.0)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    return _finish(fig, ax, "hs_small", outdir,
                   [_line(ls, x, y, ls) for y, ls in zip(ys, styles)],
                   notes="320x240, lw 1, three styles")


def fig_greys(outdir):
    """Black and dark grey, a solid and a dashed line in each."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    specs = [("#000000", "-", 20 + 5 * np.sin(0.5 * x)), ("#000000", "--", 12 + 0.8 * x),
             ("#555555", "-", 30 - 1.2 * x), ("#555555", "--", 5 + 2 * np.cos(0.8 * x))]
    for colour, ls, y in specs:
        ax.plot(x, y, color=colour, linestyle=ls, linewidth=1.6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 35)
    return _finish(fig, ax, "hs_greys", outdir,
                   [_line(f"{c} {ls}", x, y, ls, color=c) for c, ls, y in specs],
                   notes="two greys, a solid and a dashed line in each")


def fig_jpeg(outdir, base):
    """The four-style figure pushed through JPEG."""
    source = Image.open(outdir / base["file"])
    target = outdir / "hs_jpeg.png"
    jpeg = target.with_suffix(".jpg")
    source.convert("RGB").save(jpeg, quality=60)
    Image.open(jpeg).convert("RGB").save(target)
    jpeg.unlink(missing_ok=True)
    out = dict(base)
    out.update(name="hs_jpeg", file=target.name, notes="four styles, JPEG quality 60")
    return out


def build(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    records = [
        fig_four_styles(outdir),
        fig_cross_markers(outdir),
        fig_dense_shapes(outdir),
        fig_loglog_power(outdir),
        fig_small(outdir),
        fig_greys(outdir),
    ]
    records.append(fig_jpeg(outdir, records[0]))
    manifest = {"figures": records}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv):
    outdir = Path(argv[1]) if len(argv) > 1 else \
        Path(__file__).resolve().parents[1] / "tests" / "samecolour_holdout"
    manifest = build(outdir)
    for record in manifest["figures"]:
        print(f"{record['name']:22s} {record['width']}x{record['height']}  {record['notes']}")
    print(f"\n{len(manifest['figures'])} figures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
