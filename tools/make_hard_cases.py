"""Figures that broke the same-colour separation, each with its ground truth.

These are the failures an independent review found in the first version of
``detect/separate.py``, rebuilt from known data so that a fix can be measured rather
than eyeballed. Two kinds are mixed here on purpose:

  * figures whose series share a colour and should be told apart - curves crossing at
    ordinary angles, close parallel dashes, lines carrying filled markers, error bars,
    text in the plot, high resolutions;
  * figures whose colours already tell their series apart, or that hold only one
    series. Colour separation handled these before separation existed, and the only
    acceptable outcome for them is the same one: scatters in their own colour, pale
    bands and shading, families of related colours, closed and folded curves.

This set was used to fix those failures, so it is a development set like
``tests/samecolour``, not a held-out one.

Usage:  python tools/make_hard_cases.py [outdir]
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

INK = "#000000"
FIGURES = []


def figure(function):
    FIGURES.append(function)
    return function


def _s(name, x, y, color=INK, marker="none", linestyle="-"):
    return dict(name=name, x=np.asarray(x, float), y=np.asarray(y, float), color=color,
                marker=marker, linestyle=linestyle)


def _rng(seed):
    return np.random.default_rng(20260921 + seed)


# -- same-style curves crossing ---------------------------------------------------------


def _cross(outdir, name, style, d, lw=1.5):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    up, down = 10 + d * (x - 5), 10 - d * (x - 5)
    ax.plot(x, up, color=INK, linestyle=style, linewidth=lw)
    ax.plot(x, down, color=INK, linestyle=style, linewidth=lw)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    return _finish(fig, ax, name, outdir, [_s("up", x, up, linestyle=style),
                                           _s("down", x, down, linestyle=style)],
                   notes=f"two {style} lines crossing, slope {d}")


@figure
def x_dash_07(outdir):
    return _cross(outdir, "x_dash_07", "--", 0.7)


@figure
def x_dot_07(outdir):
    return _cross(outdir, "x_dot_07", ":", 0.7)


@figure
def x_dash_04(outdir):
    return _cross(outdir, "x_dash_04", "--", 0.4)


@figure
def x_dash_10(outdir):
    return _cross(outdir, "x_dash_10", "--", 1.0)


@figure
def x_solid_02(outdir):
    return _cross(outdir, "x_solid_02", "-", 0.2)


# -- a scatter alone in its colour ------------------------------------------------------


def _red_scatter(outdir, name, marker, n, seed, ms=6):
    rng = _rng(seed)
    fig, ax = _new_fig()
    x = np.sort(rng.uniform(0.2, 9.8, n))
    y = 5 + 0.3 * x + rng.normal(0, 1.0, n) - 1.5
    ax.plot(x, y, linestyle="none", marker=marker, markersize=ms, color="#d62728")
    xl = np.linspace(0, 10, 200)
    yl = 8.5 - 0.2 * xl
    ax.plot(xl, yl, color="#1f77b4", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, name, outdir,
                   [_s("scatter", x, y, "#d62728", marker, "none"), _s("line", xl, yl, "#1f77b4")],
                   notes=f"{n} red '{marker}' markers and a blue line")


@figure
def scatter_o_150a(outdir):
    return _red_scatter(outdir, "scatter_o_150a", "o", 150, 0)


@figure
def scatter_o_150b(outdir):
    return _red_scatter(outdir, "scatter_o_150b", "o", 150, 2)


@figure
def scatter_o_200(outdir):
    return _red_scatter(outdir, "scatter_o_200", "o", 200, 1)


@figure
def scatter_diamond_100(outdir):
    return _red_scatter(outdir, "scatter_diamond_100", "D", 100, 1)


@figure
def scatter_plus_300(outdir):
    return _red_scatter(outdir, "scatter_plus_300", "+", 300, 0, ms=4)


def _small_dots(outdir, name, n, ms, seed):
    rng = _rng(seed)
    fig, ax = _new_fig()
    x = rng.uniform(0, 10, n)
    y = 20 + 8 * np.sin(x) + rng.normal(0, 3, n)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ax.plot(x, y, ".", color="#1f77b4", markersize=ms)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    return _finish(fig, ax, name, outdir, [_s("dots", x, y, "#1f77b4", ".", "none")],
                   notes=f"{n} small dots, ms {ms}")


@figure
def small_dots_ms3(outdir):
    return _small_dots(outdir, "small_dots_ms3", 200, 3, 1)


@figure
def small_dots_ms2(outdir):
    return _small_dots(outdir, "small_dots_ms2", 300, 2, 0)


@figure
def data_and_fit(outdir):
    """Black filled squares scattered about a black fit line."""
    rng = _rng(3)
    fig, ax = _new_fig()
    x = np.linspace(0.3, 9.7, 40)
    y = 2 + 0.8 * x + rng.normal(0, 0.6, x.size)
    slope, intercept = np.polyfit(x, y, 1)
    xl = np.linspace(0, 10, 200)
    ax.plot(x, y, "s", color=INK, markersize=5)
    ax.plot(xl, slope * xl + intercept, color=INK, linewidth=1.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    return _finish(fig, ax, "data_and_fit", outdir,
                   [_s("data", x, y, marker="s", linestyle="none"),
                    _s("fit", xl, slope * xl + intercept)],
                   notes="40 filled squares about a fit line")


# -- noise ----------------------------------------------------------------------------


def _speckle(path, fraction, seed):
    rng = _rng(seed)
    image = np.array(Image.open(path).convert("RGB"))
    h, w = image.shape[:2]
    count = int(fraction * h * w)
    rows, cols = rng.integers(0, h, count), rng.integers(0, w, count)
    image[rows, cols] = rng.integers(0, 80, (count, 1))
    jpeg = path.with_suffix(".jpg")
    Image.fromarray(image).save(jpeg, quality=75)
    Image.open(jpeg).convert("RGB").save(path)
    jpeg.unlink(missing_ok=True)


@figure
def speckled_styles(outdir):
    """Solid, dashed and dotted black lines on a speckled, compressed scan."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ys = [2 * x + 5, 25 - 1.5 * x, 12 + 6 * np.sin(0.7 * x)]
    styles = ["-", "--", ":"]
    for y, ls in zip(ys, styles):
        ax.plot(x, y, color=INK, linestyle=ls, linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    record = _finish(fig, ax, "speckled_styles", outdir,
                     [_s(ls, x, y, linestyle=ls) for y, ls in zip(ys, styles)],
                     notes="three styles, 0.2% speckle, JPEG 75")
    _speckle(outdir / record["file"], 0.002, 0)
    return record


@figure
def speckled_colours(outdir):
    """Two coloured lines on a speckled scan."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 20 + 15 * np.sin(x) + 1.5 * x, 10 + 2 * x
    ax.plot(x, ya, color="#d62728", linewidth=2.0)
    ax.plot(x, yb, color="#1f77b4", linewidth=2.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 50)
    record = _finish(fig, ax, "speckled_colours", outdir,
                     [_s("red", x, ya, "#d62728"), _s("blue", x, yb, "#1f77b4")],
                     notes="red and blue lines, 0.2% speckle, JPEG 75")
    _speckle(outdir / record["file"], 0.002, 1)
    return record


# -- pieces of one style ----------------------------------------------------------------


def _partial(outdir, name, a_end, b_start, a, b):
    fig, ax = _new_fig()
    xa = np.linspace(0, a_end, 150)
    xb = np.linspace(b_start, 10, 150)
    x = np.linspace(0, 10, 300)
    ya, yb, yd = a(xa), b(xb), 16 + 3 * np.sin(0.9 * x)
    ax.plot(xa, ya, color=INK, linewidth=1.5)
    ax.plot(xb, yb, color=INK, linewidth=1.5)
    ax.plot(x, yd, color=INK, linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    return _finish(fig, ax, name, outdir, [_s("left", xa, ya), _s("right", xb, yb),
                                           _s("dashed", x, yd, linestyle="--")],
                   notes="two solid curves on different x ranges and a dashed one")


@figure
def partial_apart(outdir):
    return _partial(outdir, "partial_apart", 4.0, 6.0,
                    lambda x: 30 - 2 * x, lambda x: 5 + 1.5 * (x - 5))


@figure
def partial_touching(outdir):
    return _partial(outdir, "partial_touching", 4.85, 5.15,
                    lambda x: 30 - 2 * x, lambda x: 5 + 1.5 * (x - 5))


@figure
def partial_overlapping(outdir):
    return _partial(outdir, "partial_overlapping", 5.1, 4.9,
                    lambda x: 30 + 0.5 * x, lambda x: 6 + 0.5 * (x - 5))


def _three_dashed(outdir, name, spacing):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    ys = [8 + 0.6 * x + k * spacing for k in range(3)]
    for y in ys:
        ax.plot(x, y, color=INK, linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    return _finish(fig, ax, name, outdir,
                   [_s(f"k{k}", x, y, linestyle="--") for k, y in enumerate(ys)],
                   notes=f"three parallel dashed lines {spacing} apart")


@figure
def three_dashed_055(outdir):
    return _three_dashed(outdir, "three_dashed_055", 0.55)


@figure
def three_dashed_04(outdir):
    return _three_dashed(outdir, "three_dashed_04", 0.4)


# -- markers next to lines --------------------------------------------------------------


@figure
def thick_line_markers(outdir):
    """Three thin lines, one thick one, and a scatter of small filled circles."""
    rng = _rng(4)
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    lines = [3 + 0.5 * x, 18 - 0.4 * x, 10 + 2 * np.sin(0.6 * x)]
    for y in lines:
        ax.plot(x, y, color=INK, linewidth=1.0)
    thick = 14 + 0.3 * x
    ax.plot(x, thick, color=INK, linewidth=4.0)
    mx = np.linspace(0.7, 9.3, 14)
    my = 1.2 + rng.uniform(0, 0.8, mx.size)
    ax.plot(mx, my, "o", color=INK, markersize=4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    series = [_s(f"thin{i}", x, y) for i, y in enumerate(lines)]
    series += [_s("thick", x, thick), _s("markers", mx, my, marker="o", linestyle="none")]
    return _finish(fig, ax, "thick_line_markers", outdir, series,
                   notes="three lw 1 lines, a lw 4 line, 14 small filled circles")


@figure
def converging_markers(outdir):
    """Two thin lines a few pixels apart for a stretch, and a scatter."""
    rng = _rng(5)
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    ya = 5 + 0.8 * x
    yb = ya + 0.05 * 20 * np.clip((x - 4) / 6, 0, None) ** 1.2 * 6 + 0.3
    ax.plot(x, ya, color=INK, linewidth=1.0)
    ax.plot(x, yb, color=INK, linewidth=1.0)
    mx = np.linspace(0.5, 9.5, 12)
    my = 2 + rng.uniform(0, 1.0, mx.size)
    ax.plot(mx, my, "o", color=INK, markersize=5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    return _finish(fig, ax, "converging_markers", outdir,
                   [_s("a", x, ya), _s("b", x, yb),
                    _s("markers", mx, my, marker="o", linestyle="none")],
                   notes="two lines a few pixels apart at first, 12 filled circles")


@figure
def thin_and_round_dots(outdir):
    """A thin solid line beside a thick round-capped dotted one."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 10 + 0.5 * x, 4 + 0.9 * x
    ax.plot(x, ya, color=INK, linewidth=1.0)
    ax.plot(x, yb, color=INK, linestyle=":", linewidth=2.0, dash_capstyle="round")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    return _finish(fig, ax, "thin_and_round_dots", outdir,
                   [_s("solid", x, ya), _s("dotted", x, yb, linestyle=":")],
                   notes="lw 1 solid and lw 2 round-cap dotted")


@figure
def crossing_with_x_markers(outdir):
    """Two curves crossing at right angles, and a row of x markers above them."""
    rng = _rng(6)
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 400)
    ya, yb = 5 + 1.34 * np.sin(x), 5 - 1.34 * np.sin(x)
    ax.plot(x, ya, color=INK, linewidth=1.0)
    ax.plot(x, yb, color=INK, linewidth=1.0)
    mx = np.linspace(0.8, 9.2, 12)
    my = 8.3 + rng.uniform(0, 0.6, mx.size)
    ax.plot(mx, my, "x", color=INK, markersize=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "crossing_with_x_markers", outdir,
                   [_s("a", x, ya), _s("b", x, yb),
                    _s("x", mx, my, marker="x", linestyle="none")],
                   notes="sine curves crossing at right angles, 12 x markers")


def _marker_pair(outdir, name, first, second, fills=("none", "none"), ms=6):
    rng = _rng(7)
    fig, ax = _new_fig()
    series = []
    for i, (marker, face) in enumerate(zip((first, second), fills)):
        mx = np.linspace(0.6, 9.4, 12)
        my = 3 + 3 * i + rng.normal(0, 0.3, mx.size)
        ax.plot(mx, my, linestyle="none", marker=marker, markersize=ms,
                markerfacecolor=face, markeredgecolor=INK)
        series.append(_s(marker, mx, my, marker=marker, linestyle="none"))
    x = np.linspace(0, 10, 200)
    ax.plot(x, 9.5 - 0.1 * x, color=INK, linewidth=1.0)
    series.append(_s("line", x, 9.5 - 0.1 * x))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    return _finish(fig, ax, name, outdir, series,
                   notes=f"'{first}' and '{second}' markers and a line")


@figure
def markers_diamond_circle(outdir):
    return _marker_pair(outdir, "markers_diamond_circle", "D", "o")


@figure
def markers_left_right(outdir):
    return _marker_pair(outdir, "markers_left_right", "<", ">")


@figure
def markers_x_plus(outdir):
    return _marker_pair(outdir, "markers_x_plus", "x", "+")


# -- two dash patterns ----------------------------------------------------------------


def _two_patterns(outdir, name, first, second, lw=(1.5, 1.5)):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 12 + 5 * np.sin(0.5 * x), 4 + 1.2 * x
    ax.plot(x, ya, color=INK, linestyle=first, linewidth=lw[0])
    ax.plot(x, yb, color=INK, linestyle=second, linewidth=lw[1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    return _finish(fig, ax, name, outdir,
                   [_s("a", x, ya, linestyle=str(first)), _s("b", x, yb, linestyle=str(second))],
                   notes=f"{first} and {second}")


@figure
def dash_and_dashdot(outdir):
    return _two_patterns(outdir, "dash_and_dashdot", "--", "-.")


@figure
def dash_and_longdash(outdir):
    return _two_patterns(outdir, "dash_and_longdash", "--", (0, (10, 3)))


@figure
def dash_thin_and_thick(outdir):
    return _two_patterns(outdir, "dash_thin_and_thick", "--", "--", lw=(1.0, 2.0))


@figure
def thick_dashdot_alone(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 2 + 0.6 * x + np.sin(x), 9 - 0.4 * x
    ax.plot(x, ya, color="#d62728", linestyle="-.", linewidth=4)
    ax.plot(x, yb, color="#1f77b4", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "thick_dashdot_alone", outdir,
                   [_s("red", x, ya, "#d62728", linestyle="-."), _s("blue", x, yb, "#1f77b4")],
                   notes="a red lw 4 dash-dot line alone in its colour, a blue line")


# -- pale regions and related colours --------------------------------------------------


@figure
def grey_band(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y = 5 + np.sin(x)
    ax.fill_between(x, y - 1, y + 1, color="0.8")
    ax.plot(x, y, color=INK, linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "grey_band", outdir, [_s("line", x, y)],
                   notes="black line in a grey confidence band")


@figure
def blue_band(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y, z = 5 + np.sin(x), 8 - 0.3 * x
    ax.fill_between(x, y - 1, y + 1, color="tab:blue", alpha=0.3)
    ax.plot(x, y, color="tab:blue", linewidth=1.5)
    ax.plot(x, z, color="tab:red", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "blue_band", outdir,
                   [_s("blue", x, y, "#1f77b4"), _s("red", x, z, "#d62728")],
                   notes="blue line in a translucent blue band, a red line")


@figure
def two_bands(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y, z = 3 + np.sin(x), 7 + 0.5 * np.cos(x)
    ax.fill_between(x, y - 0.8, y + 0.8, color="tab:blue", alpha=0.3)
    ax.fill_between(x, z - 0.8, z + 0.8, color="tab:orange", alpha=0.3)
    ax.plot(x, y, color="tab:blue", linewidth=1.5)
    ax.plot(x, z, color="tab:orange", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "two_bands", outdir,
                   [_s("blue", x, y, "#1f77b4"), _s("orange", x, z, "#ff7f0e")],
                   notes="two lines, each in its own translucent band")


@figure
def shaded_span(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y, z = 5 + np.sin(x), 8 - 0.3 * x
    ax.axvspan(3, 5, color="tab:orange", alpha=0.3)
    ax.plot(x, y, color="tab:blue", linewidth=1.5)
    ax.plot(x, z, color="tab:red", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "shaded_span", outdir,
                   [_s("blue", x, y, "#1f77b4"), _s("red", x, z, "#d62728")],
                   notes="a shaded vertical span behind two lines")


@figure
def watermark(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y, z = 5 + np.sin(x), 2 + 0.6 * x
    ax.text(5, 5, "DRAFT", fontsize=60, color="0.85", ha="center", va="center", rotation=20)
    ax.plot(x, y, color="tab:blue", linewidth=1.5)
    ax.plot(x, z, color="tab:orange", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "watermark", outdir,
                   [_s("blue", x, y, "#1f77b4"), _s("orange", x, z, "#ff7f0e")],
                   notes="a pale grey watermark behind two lines")


@figure
def navy_and_light(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 1.5 + 0.3 * x, 5.5 + 0.3 * x
    ax.plot(x, ya, color="#08306b", linewidth=1.5)
    ax.plot(x, yb, color="#6baed6", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "navy_and_light", outdir,
                   [_s("navy", x, ya, "#08306b"), _s("light", x, yb, "#6baed6")],
                   notes="navy and light blue lines")


def _family(outdir, name, cmap):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    colours = plt.get_cmap(cmap)(np.linspace(0.35, 1, 5))
    series = []
    for i, colour in enumerate(colours):
        y = 1 + (2 + 1.6 * i) * np.exp(-x / 3)
        ax.plot(x, y, color=colour, linewidth=1.5)
        series.append(_s(f"c{i}", x, y, matplotlib.colors.to_hex(colour)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, name, outdir, series, notes=f"five curves from {cmap}")


@figure
def family_reds(outdir):
    return _family(outdir, "family_reds", "Reds")


@figure
def family_blues(outdir):
    return _family(outdir, "family_blues", "Blues")


@figure
def family_greys(outdir):
    return _family(outdir, "family_greys", "Greys")


@figure
def family_viridis(outdir):
    return _family(outdir, "family_viridis", "viridis")


@figure
def ggplot_three(outdir):
    fig, ax = _new_fig(style="ggplot")
    x = np.linspace(0, 10, 300)
    ys = [5 + 3 * np.sin(0.6 * x), 2 + 0.8 * x, 11 - 0.5 * x]
    colours = ["#E24A33", "#348ABD", "#988ED5"]
    for y, c in zip(ys, colours):
        ax.plot(x, y, color=c, linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    return _finish(fig, ax, "ggplot_three", outdir,
                   [_s(c, x, y, c) for y, c in zip(ys, colours)], notes="ggplot style")


@figure
def raw_and_smoothed(outdir):
    rng = _rng(8)
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    raw = 5 + 2 * np.sin(0.8 * x) + rng.normal(0, 0.12, x.size)
    smooth = np.convolve(np.pad(raw, 12, mode="edge"), np.ones(25) / 25, mode="valid")
    ax.plot(x, raw, color="#aec7e8", linewidth=1.5)
    ax.plot(x, smooth, color="#1f77b4", linewidth=2.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "raw_and_smoothed", outdir,
                   [_s("raw", x, raw, "#aec7e8"), _s("smooth", x, smooth, "#1f77b4")],
                   notes="pale raw data under its dark moving average")


@figure
def pale_points_dark_fit(outdir):
    rng = _rng(9)
    fig, ax = _new_fig()
    x = np.sort(rng.uniform(0.5, 9.5, 40))
    y = 2 + 0.6 * x + rng.normal(0, 0.1, x.size)
    xl = np.linspace(0, 10, 200)
    slope, intercept = np.polyfit(x, y, 1)
    ax.scatter(x, y, color="#aec7e8", s=25)
    ax.plot(xl, slope * xl + intercept, color="#1f77b4", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "pale_points_dark_fit", outdir,
                   [_s("points", x, y, "#aec7e8", "o", "none"),
                    _s("fit", xl, slope * xl + intercept, "#1f77b4")],
                   notes="pale points about a dark fit line of the same hue")


# -- high resolution -----------------------------------------------------------------


@figure
def hidpi_line(outdir):
    fig, ax = _new_fig(dpi=360)
    x = np.linspace(0, 10, 300)
    y = 5 + 2 * x - 0.1 * x ** 2
    ax.plot(x, y, color=INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "hidpi_line", outdir, [_s("line", x, y)],
                   notes="one black line at 360 dpi")


@figure
def hidpi_styles(outdir):
    rng = _rng(10)
    fig, ax = _new_fig(dpi=300)
    x = np.linspace(0, 10, 300)
    ys = [2 * x + 5, 28 - 1.8 * x, 15 + 6 * np.sin(0.8 * x)]
    styles = ["-", "--", ":"]
    for y, ls in zip(ys, styles):
        ax.plot(x, y, color=INK, linestyle=ls)
    mx = np.linspace(0.5, 9.5, 15)
    my = 3 + rng.uniform(0, 3, mx.size)
    ax.plot(mx, my, "^", color=INK, markerfacecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 32)
    series = [_s(ls, x, y, linestyle=ls) for y, ls in zip(ys, styles)]
    series.append(_s("triangles", mx, my, marker="^", linestyle="none"))
    return _finish(fig, ax, "hidpi_styles", outdir, series,
                   notes="solid, dashed, dotted and hollow triangles at 300 dpi")


@figure
def hidpi_line_markers(outdir):
    fig, ax = _new_fig(dpi=300)
    x = np.linspace(0, 10, 300)
    ya, yb, yc = 4 + 1.5 * x, 25 - 1.2 * x, 12 + 4 * np.sin(0.7 * x)
    ax.plot(x, ya, color=INK, marker="o", markevery=30, markerfacecolor="white")
    ax.plot(x, yb, color=INK, linestyle=":", marker="^", markevery=30, markerfacecolor="white")
    ax.plot(x, yc, color=INK, linestyle="--")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "hidpi_line_markers", outdir,
                   [_s("circles", x, ya, marker="o"), _s("triangles", x, yb, marker="^",
                                                        linestyle=":"),
                    _s("dashed", x, yc, linestyle="--")],
                   notes="lines carrying hollow markers, at 300 dpi")


@figure
def hidpi_dashdot(outdir):
    fig, ax = _new_fig(dpi=400)
    x = np.linspace(0, 10, 300)
    ya, yb = 3 + 1.8 * x, 20 - 1.5 * x
    ax.plot(x, ya, color=INK, linewidth=1.5)
    ax.plot(x, yb, color=INK, linestyle="-.", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    return _finish(fig, ax, "hidpi_dashdot", outdir,
                   [_s("solid", x, ya), _s("dashdot", x, yb, linestyle="-.")],
                   notes="solid and dash-dot crossing at 400 dpi")


@figure
def hidpi_markers(outdir):
    rng = _rng(11)
    fig, ax = _new_fig(dpi=500)
    series = []
    for i, (marker, face) in enumerate([("o", "none"), ("s", INK), ("^", "none")]):
        mx = np.sort(rng.uniform(0.5, 9.5, 14))
        my = 2 + 3 * i + rng.normal(0, 0.5, mx.size)
        ax.plot(mx, my, linestyle="none", marker=marker, markerfacecolor=face,
                markeredgecolor=INK)
        series.append(_s(marker, mx, my, marker=marker, linestyle="none"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "hidpi_markers", outdir, series,
                   notes="hollow circles, filled squares, hollow triangles at 500 dpi")


@figure
def hidpi_filled(outdir):
    rng = _rng(12)
    fig, ax = _new_fig(dpi=450)
    series = []
    for i, marker in enumerate(["o", "s", "^"]):
        mx = np.sort(rng.uniform(0.5, 9.5, 14))
        my = 2 + 3 * i + rng.normal(0, 0.5, mx.size)
        ax.plot(mx, my, linestyle="none", marker=marker, color=INK)
        series.append(_s(marker, mx, my, marker=marker, linestyle="none"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "hidpi_filled", outdir, series,
                   notes="filled circles, squares and triangles at 450 dpi")


# -- text in the plot ------------------------------------------------------------------


@figure
def legend_unframed_one(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    ax.plot(x, 2 * x + 5, color=INK, label="experiment (sample A)")
    ax.legend(frameon=False, loc="upper center", fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "legend_unframed_one", outdir, [_s("line", x, 2 * x + 5)],
                   notes="one line, legend without a frame")


@figure
def legend_unframed_two(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    ya, yb = 2 + np.sin(x), 0.5 * np.cos(x)
    ax.plot(x, ya, color=INK, label="measured response")
    ax.plot(x, yb, color=INK, linestyle="--", label="reference model")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 4.5)
    return _finish(fig, ax, "legend_unframed_two", outdir,
                   [_s("solid", x, ya), _s("dashed", x, yb, linestyle="--")],
                   notes="solid and dashed, legend without a frame")


@figure
def legend_unframed_four(outdir):
    fig, ax = _new_fig(dpi=150)
    x = np.linspace(0, 10, 300)
    ys = [2 + 0.5 * x, 9 - 0.6 * x, 5 + 1.5 * np.sin(0.8 * x), 1 + 0.05 * x ** 2]
    styles = ["-", "--", ":", "-."]
    for i, (y, ls) in enumerate(zip(ys, styles)):
        ax.plot(x, y, color=INK, linestyle=ls,
                label=f"sensor {['one', 'two', 'three', 'four'][i]}")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    return _finish(fig, ax, "legend_unframed_four", outdir,
                   [_s(ls, x, y, linestyle=ls) for y, ls in zip(ys, styles)],
                   notes="four styles, legend without a frame, 150 dpi")


@figure
def annotated(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    y = 25 * (1 - np.exp(-x / 2.5))
    ax.plot(x, y, color=INK)
    ax.text(4, 27, "onset of saturation")
    ax.text(6.2, 8, "slope = 2.00 ± 0.05")
    ax.text(6.2, 5, "R² = 0.998")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 32)
    return _finish(fig, ax, "annotated", outdir, [_s("line", x, y)],
                   notes="one line and three annotations")


# -- filled markers on lines -----------------------------------------------------------


def _markevery(outdir, name, every, dpi, legend):
    fig, ax = _new_fig(dpi=dpi)
    x = np.linspace(0, 10, 200)
    ya, yb = 2 * x + 5, 30 - 2 * x
    ax.plot(x, ya, "k-o", markevery=every, label="heating run")
    ax.plot(x, yb, "k--s", markevery=every, label="cooling run")
    if legend:
        ax.legend(frameon=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 32)
    return _finish(fig, ax, name, outdir,
                   [_s("heating", x, ya, marker="o"), _s("cooling", x, yb, marker="s",
                                                         linestyle="--")],
                   notes=f"lines with filled markers every {every}, {dpi} dpi")


@figure
def markevery_legend(outdir):
    return _markevery(outdir, "markevery_legend", 20, 100, True)


@figure
def markevery_25(outdir):
    return _markevery(outdir, "markevery_25", 25, 100, False)


@figure
def markevery_dense(outdir):
    return _markevery(outdir, "markevery_dense", 10, 150, False)


# -- error bars ------------------------------------------------------------------------


@figure
def errorbars_and_line(outdir):
    fig, ax = _new_fig()
    xs = np.linspace(0.5, 9.5, 13)
    x = np.linspace(0, 10, 200)
    ax.errorbar(xs, 2 * xs + 3, yerr=1.5, fmt="ko", mfc="none", capsize=3)
    ax.plot(x, 28 - 2 * x, "k-")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "errorbars_and_line", outdir,
                   [_s("points", xs, 2 * xs + 3, marker="o", linestyle="none"),
                    _s("line", x, 28 - 2 * x)],
                   notes="hollow circles with y error bars, and a line")


@figure
def errorbars_two_shapes(outdir):
    fig, ax = _new_fig()
    xs = np.linspace(0.5, 9.5, 13)
    ax.errorbar(xs, 2 * xs + 3, yerr=1.2, fmt="ks", capsize=3)
    ax.errorbar(xs + 0.2, 25 - xs, yerr=1.2, fmt="k^", mfc="none", capsize=3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "errorbars_two_shapes", outdir,
                   [_s("squares", xs, 2 * xs + 3, marker="s", linestyle="none"),
                    _s("triangles", xs + 0.2, 25 - xs, marker="^", linestyle="none")],
                   notes="filled squares and hollow triangles, both with error bars")


@figure
def errorbars_horizontal(outdir):
    fig, ax = _new_fig()
    xs = np.linspace(0.5, 9.5, 13)
    x = np.linspace(0, 10, 200)
    ax.errorbar(xs, 2 * xs + 3, xerr=0.2, fmt="ko", mfc="none", capsize=3)
    ax.plot(x, 28 - 2 * x, "k-")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "errorbars_horizontal", outdir,
                   [_s("points", xs, 2 * xs + 3, marker="o", linestyle="none"),
                    _s("line", x, 28 - 2 * x)],
                   notes="hollow circles with capped x error bars, and a line")


@figure
def errorbars_dashed(outdir):
    fig, ax = _new_fig()
    xs = np.linspace(0.5, 9.5, 13)
    x = np.linspace(0, 10, 200)
    ax.errorbar(xs, 2 * xs + 3, yerr=1.5, fmt="ko--", mfc="none", capsize=3)
    ax.plot(x, 28 - 2 * x, "k-")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "errorbars_dashed", outdir,
                   [_s("points", xs, 2 * xs + 3, marker="o", linestyle="--"),
                    _s("line", x, 28 - 2 * x)],
                   notes="a dashed line through error-bar markers, and a solid line")


@figure
def errorbars_journal(outdir):
    fig, ax = _new_fig(figsize=(3.3, 2.5), dpi=300)
    xs = np.linspace(0.5, 9.5, 13)
    x = np.linspace(0, 10, 200)
    ax.errorbar(xs, 2 * xs + 3, yerr=1.0, fmt="ks", ms=4, capsize=2)
    ax.errorbar(xs, 25 - xs, yerr=1.0, fmt="k^", mfc="none", ms=4, capsize=2)
    ax.plot(x, 28 - 2 * x, "k-", lw=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    return _finish(fig, ax, "errorbars_journal", outdir,
                   [_s("squares", xs, 2 * xs + 3, marker="s", linestyle="none"),
                    _s("triangles", xs, 25 - xs, marker="^", linestyle="none"),
                    _s("line", x, 28 - 2 * x)],
                   notes="single-column figure at 300 dpi, two error-bar series and a line")


# -- curves that are not a function of x -----------------------------------------------


@figure
def ellipse(outdir):
    fig, ax = _new_fig()
    t = np.linspace(0, 2 * np.pi, 400)
    x, y = 5 + 4 * np.cos(t), 5 + 3 * np.sin(t)
    ax.plot(x, y, color=INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "ellipse", outdir, [_s("ellipse", x, y)], notes="one ellipse")


@figure
def voltammogram(outdir):
    fig, ax = _new_fig()
    t = np.linspace(0, 2 * np.pi, 600)
    e = 0.5 * np.cos(t)
    i = 2 * np.sin(t) * np.exp(-((e - 0.1) ** 2) / 0.05) + 0.3 * np.sin(t) + e
    ax.plot(e, i, color="tab:green")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-3, 3)
    return _finish(fig, ax, "voltammogram", outdir, [_s("cv", e, i, "#2ca02c")],
                   notes="one cyclic voltammogram in green")


@figure
def thin_loop(outdir):
    fig, ax = _new_fig()
    t = np.linspace(0, 2 * np.pi, 600)
    h = 3 * np.cos(t)
    m = np.tanh(1.2 * h) + 0.02 * np.sin(t)
    ax.plot(h, m, color=INK)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1.5, 1.5)
    return _finish(fig, ax, "thin_loop", outdir, [_s("loop", h, m)],
                   notes="a hysteresis loop whose branches almost coincide")


@figure
def z_fold(outdir):
    fig, ax = _new_fig()
    s = np.linspace(-1.6, 1.6, 400)
    x, y = s ** 3 - 1.5 * s, s
    ax.plot(x, y, color=INK)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return _finish(fig, ax, "z_fold", outdir, [_s("fold", x, y)], notes="an S-shaped fold")


@figure
def sawtooth(outdir):
    fig, ax = _new_fig(figsize=(6, 4))
    t = np.linspace(0, 10, 4000)
    y = (t % 2.0) / 2.0
    ax.plot(t, y, color="tab:red", linewidth=1.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 1.1)
    return _finish(fig, ax, "sawtooth", outdir, [_s("saw", t, y, "#d62728")],
                   notes="a steep sawtooth in red")


@figure
def sawtooth_thin(outdir):
    fig, ax = _new_fig(figsize=(6, 4))
    t = np.linspace(0, 10, 4000)
    y = (t % 2.0) / 2.0
    ax.plot(t, y, color="tab:green", linewidth=1.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 1.1)
    return _finish(fig, ax, "sawtooth_thin", outdir, [_s("saw", t, y, "#2ca02c")],
                   notes="a steep sawtooth in green, lw 1")


# -- markers drawn on lines ------------------------------------------------------------


@figure
def white_markers_line(outdir):
    """A line with white-faced circles close together: the face hides the line."""
    fig, ax = _new_fig()
    x = np.linspace(0.5, 9.5, 20)
    ax.plot(x, 1 + 0.8 * x, "o-", color=INK, mfc="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "white_markers_line", outdir, [_s("line", x, 1 + 0.8 * x, marker="o")],
                   notes="one line, 20 white-faced circles")


@figure
def white_markers_sine(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0.3, 9.7, 30)
    y = 5 + 3 * np.sin(x / 1.3)
    ax.plot(x, y, "o-", color="tab:blue", mfc="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "white_markers_sine", outdir,
                   [_s("sine", x, y, "#1f77b4", "o")], notes="a sine through 30 white-faced circles")


@figure
def markers_on_dotted(outdir):
    """Few markers on a dotted line, and squares on a dashed sine, each alone in a colour."""
    fig, ax = _new_fig()
    x = np.linspace(0.5, 9.5, 5)
    xs = np.linspace(0.3, 9.7, 10)
    ys = 5 + 3 * np.sin(xs / 1.3)
    ax.plot(x, 1 + 0.8 * x, "o:", color=INK)
    ax.plot(xs, ys, "s--", color="tab:blue", ms=6, lw=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "markers_on_dotted", outdir,
                   [_s("dotted", x, 1 + 0.8 * x, marker="o", linestyle=":"),
                    _s("dashed", xs, ys, "#1f77b4", "s", "--")],
                   notes="markers on a dotted line, squares on a dashed sine")


@figure
def dashed_with_markers(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0.5, 9.5, 20)
    ax.plot(x, 1 + 0.8 * x, "o--", color=INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "dashed_with_markers", outdir,
                   [_s("line", x, 1 + 0.8 * x, marker="o", linestyle="--")],
                   notes="one dashed line with 20 filled circles")


@figure
def journal_three(outdir):
    """Three black lines, each with its own style and filled marker."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    ys = [2 + 0.6 * x, 8 - 0.5 * x, 5 + 2 * np.sin(x / 1.5)]
    for y, fmt in zip(ys, ["k-o", "k--s", "k:^"]):
        ax.plot(x, y, fmt, markevery=25, ms=6, lw=1.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "journal_three", outdir,
                   [_s("circles", x, ys[0], marker="o"),
                    _s("squares", x, ys[1], marker="s", linestyle="--"),
                    _s("triangles", x, ys[2], marker="^", linestyle=":")],
                   notes="-o, --s and :^ in black, markevery 25")


@figure
def dashdot_white_markers(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    ya, yb = 8 - 0.3 * x, 2 + 0.6 * x
    ax.plot(x, ya, "k-")
    ax.plot(x, yb, "k--", marker="o", markevery=15, mfc="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "dashdot_white_markers", outdir,
                   [_s("solid", x, ya), _s("dashed", x, yb, marker="o", linestyle="--")],
                   notes="a solid line and a dashed one with white-faced circles")


# -- short curves and dense data ---------------------------------------------------------


@figure
def short_curve(outdir):
    """Two full-width curves and one that leaves the top early."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 400)
    ys = [np.exp(0.1 * x) + 1, np.exp(0.2 * x), np.exp(1.8 * x)]
    for y in ys:
        ax.plot(x, y, color=INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "short_curve", outdir,
                   [_s(f"c{i}", x[y <= 10.5], y[y <= 10.5]) for i, y in enumerate(ys)],
                   notes="three solid curves, one leaving the plot at x=1.3")


@figure
def tangent_and_data(outdir):
    rng = _rng(13)
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    model = 8 * (1 - np.exp(-x / 3))
    xd = np.linspace(0.5, 9.5, 12)
    yd = 8 * (1 - np.exp(-xd / 3)) + rng.normal(0, 0.3, xd.size)
    xt = np.linspace(0, 1.3, 30)
    ax.plot(x, model, "k-")
    ax.plot(xd, yd, "k^", mfc="none")
    ax.plot(xt, 8 / 3 * xt, "k--")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "tangent_and_data", outdir,
                   [_s("model", x, model), _s("data", xd, yd, marker="^", linestyle="none"),
                    _s("tangent", xt, 8 / 3 * xt, linestyle="--")],
                   notes="model line, triangle data and a short dashed tangent")


@figure
def noisy_and_reference(outdir):
    rng = _rng(14)
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 1000)
    y = 5 + 1.5 * np.sin(x) + rng.normal(0, 0.3, x.size)
    ax.plot(x, y, "k-", lw=1)
    ax.plot(x, 8.5 + 0.3 * np.sin(x), "k--")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "noisy_and_reference", outdir,
                   [_s("noisy", x, y), _s("reference", x, 8.5 + 0.3 * np.sin(x), linestyle="--")],
                   notes="1000 noisy samples and a dashed reference")


@figure
def jagged_and_triangles(outdir):
    rng = _rng(15)
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 1000)
    y = 5 + 1.5 * np.sin(x) + rng.normal(0, 0.1, x.size)
    x2 = np.linspace(0.3, 9.7, 20)
    y2 = 8.5 + 0.3 * np.sin(x2)
    ax.plot(x, y, "k-", lw=1)
    ax.plot(x2, y2, "k^", ms=6, mfc="none")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "jagged_and_triangles", outdir,
                   [_s("jagged", x, y), _s("triangles", x2, y2, marker="^", linestyle="none")],
                   notes="a slightly noisy line and 20 hollow triangles")


@figure
def thick_dotted_alone(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 2 + 0.6 * x + np.sin(x), 9 - 0.4 * x
    ax.plot(x, ya, color="#d62728", linestyle=":", linewidth=3)
    ax.plot(x, yb, color="#1f77b4", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "thick_dotted_alone", outdir,
                   [_s("red", x, ya, "#d62728", linestyle=":"), _s("blue", x, yb, "#1f77b4")],
                   notes="a red lw 3 dotted line alone in its colour")


# -- step curves ------------------------------------------------------------------------


@figure
def survival_steps(outdir):
    rng = _rng(16)
    fig, ax = _new_fig()
    series = []
    for scale, ls in ((10, "-"), (20, "--")):
        t = np.sort(rng.exponential(scale, 40))
        s = 1 - np.arange(1, t.size + 1) / t.size
        tt, ss = np.r_[0, t], np.r_[1, s]
        ax.step(tt, ss, where="post", color=INK, linestyle=ls)
        # The truth of a step curve is its corners.
        cx = np.repeat(tt, 2)[1:]
        cy = np.repeat(ss, 2)[:-1]
        keep = cx <= 60
        series.append(_s(f"km{scale}", cx[keep], cy[keep], linestyle=ls))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.05)
    return _finish(fig, ax, "survival_steps", outdir, series,
                   notes="two survival curves, solid and dashed")


# -- sizes ------------------------------------------------------------------------------


@figure
def bubble_sizes(outdir):
    rng = _rng(17)
    fig, ax = _new_fig()
    x = np.sort(rng.uniform(0.5, 9.5, 25))
    y = 2 + 0.6 * x + rng.normal(0, 0.5, x.size)
    ax.scatter(x, y, s=rng.choice([15, 40, 90], x.size), c="tab:blue")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "bubble_sizes", outdir, [_s("bubbles", x, y, "#1f77b4", "o", "none")],
                   notes="one scatter in three marker sizes")


@figure
def two_square_sizes(outdir):
    rng = _rng(18)
    fig, ax = _new_fig()
    x = np.sort(rng.uniform(0.5, 9.5, 25))
    y = 2 + 0.6 * x + rng.normal(0, 0.8, x.size)
    ax.scatter(x, y, s=rng.choice([20, 70], x.size), c="k", marker="s")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "two_square_sizes", outdir, [_s("squares", x, y, marker="s", linestyle="none")],
                   notes="one scatter of squares in two sizes")


# -- compression and gridlines ---------------------------------------------------------


def _jpeg(path, quality):
    jpeg = path.with_suffix(".jpg")
    Image.open(path).convert("RGB").save(jpeg, quality=quality)
    Image.open(jpeg).convert("RGB").save(path)
    jpeg.unlink(missing_ok=True)


@figure
def jpeg_legend(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 5 + 2 * np.sin(x), 2 + 0.5 * x
    ax.plot(x, ya, label="first")
    ax.plot(x, yb, label="second")
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    record = _finish(fig, ax, "jpeg_legend", outdir,
                     [_s("first", x, ya, "#1f77b4"), _s("second", x, yb, "#ff7f0e")],
                     notes="two coloured lines and a legend, JPEG 85")
    _jpeg(outdir / record["file"], 85)
    return record


@figure
def jpeg_grid(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y = 5 + 3 * np.sin(0.8 * x)
    ax.grid(True, color="#cccccc")
    ax.plot(x, y, color="#2ca02c", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    record = _finish(fig, ax, "jpeg_grid", outdir, [_s("curve", x, y, "#2ca02c")],
                     notes="a curve over light gridlines, JPEG 80")
    _jpeg(outdir / record["file"], 80)
    return record


@figure
def jpeg_scatter(outdir):
    rng = _rng(19)
    fig, ax = _new_fig()
    x = np.linspace(0.5, 9.5, 18)
    y = 3.0 * x + 8.0 + rng.normal(0, 2.0, x.size)
    ax.scatter(x, y, c="#1f77b4", marker="o", s=42)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 50)
    record = _finish(fig, ax, "jpeg_scatter", outdir, [_s("scatter", x, y, "#1f77b4", "o", "none")],
                     notes="one scatter, JPEG 85")
    _jpeg(outdir / record["file"], 85)
    return record


@figure
def dashed_grid(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ya, yb = 5 + 2 * np.sin(x), 2 + 0.5 * x
    ax.grid(True, linestyle="--")
    ax.plot(x, ya, color="tab:red")
    ax.plot(x, yb, color="tab:blue")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return _finish(fig, ax, "dashed_grid", outdir,
                   [_s("red", x, ya, "#d62728"), _s("blue", x, yb, "#1f77b4")],
                   notes="two lines over dashed gridlines")


def build(outdir: Path, only: list[str] | None = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    records = [f(outdir) for f in FIGURES if not only or f.__name__ in only]
    manifest = {"figures": records}
    if not only:
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv):
    outdir = Path(argv[1]) if len(argv) > 1 else \
        Path(__file__).resolve().parents[1] / "tests" / "hard"
    manifest = build(outdir)
    for record in manifest["figures"]:
        print(f"{record['name']:24s} {record['width']}x{record['height']}  {record['notes']}")
    print(f"\n{len(manifest['figures'])} figures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
