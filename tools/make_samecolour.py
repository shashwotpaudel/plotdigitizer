"""Figures whose series share a colour, for developing same-colour separation.

Colour separation cannot split these: every series is drawn in the same ink. What does
tell them apart is what a reader uses - the line style (solid, dashed, dotted), the
marker glyph, and the fact that each curve continues smoothly while the others merely
pass nearby. Each figure isolates one of those cues, or the case where they have to be
combined.

This is the *development* set: the separation thresholds were chosen while looking at
it. ``tools/make_samecolour_holdout.py`` holds figures written after the method was
fixed, and is the one that says whether it generalises.

Usage:  python tools/make_samecolour.py [outdir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_plots import _finish, _new_fig  # noqa: E402

RNG = np.random.default_rng(1110917)
INK = "#000000"


def _line(name, x, y, linestyle, color=INK, marker="none"):
    return dict(name=name, x=x, y=y, color=color, marker=marker, linestyle=linestyle)


def fig_styles(outdir):
    """Solid, dashed and dotted, apart from one another: style is the only cue needed."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y1 = 32 + 7 * np.sin(0.6 * x)
    y2 = 20 + 5 * np.sin(0.6 * x + 1.0)
    y3 = 6 + 0.9 * x
    ax.plot(x, y1, color=INK, linestyle="-", linewidth=1.5, label="solid")
    ax.plot(x, y2, color=INK, linestyle="--", linewidth=1.5, label="dashed")
    ax.plot(x, y3, color=INK, linestyle=":", linewidth=1.5, label="dotted")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 45)
    ax.legend(loc="upper right")
    return _finish(fig, ax, "sc_styles", outdir, [
        _line("solid", x, y1, "-"), _line("dashed", x, y2, "--"), _line("dotted", x, y3, ":"),
    ], notes="black solid/dashed/dotted, separated")


def fig_solid_three(outdir):
    """Three solid black curves that never touch: only their continuity separates them."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y1 = 5 + 0.35 * x ** 2
    y2 = 12 + 2.0 * x
    y3 = 30 + 6 * np.sqrt(x)
    for y in (y1, y2, y3):
        ax.plot(x, y, color=INK, linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 60)
    return _finish(fig, ax, "sc_solid_three", outdir, [
        _line("a", x, y1, "-"), _line("b", x, y2, "-"), _line("c", x, y3, "-"),
    ], notes="three solid black curves, never touching")


def fig_crossing(outdir):
    """Two solid curves that cross twice: at each crossing, slope decides who is who."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 400)
    y1 = 20 + 10 * np.sin(0.8 * x)
    y2 = 20 + 10 * np.cos(0.8 * x)
    ax.plot(x, y1, color=INK, linewidth=1.5)
    ax.plot(x, y2, color=INK, linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    return _finish(fig, ax, "sc_crossing", outdir, [
        _line("sin", x, y1, "-"), _line("cos", x, y2, "-"),
    ], notes="two solid black curves crossing three times")


def fig_markers(outdir):
    """A monochrome scatter: hollow circles, filled squares, hollow triangles."""
    fig, ax = _new_fig()
    specs = [
        ("circles", "o", "none", 8.0, 1.2),
        ("squares", "s", INK, 20.0, 0.9),
        ("triangles", "^", "none", 32.0, 0.6),
    ]
    series = []
    for name, marker, face, offset, slope in specs:
        x = np.sort(RNG.uniform(0.5, 9.5, 16))
        y = offset + slope * x + RNG.normal(0, 1.2, x.size)
        ax.plot(x, y, linestyle="none", marker=marker, markersize=7,
                markerfacecolor=face, markeredgecolor=INK, markeredgewidth=1.2)
        series.append(dict(name=name, x=x, y=y, color=INK, marker=marker, linestyle="none"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 45)
    return _finish(fig, ax, "sc_markers", outdir, series,
                   notes="black scatter told apart only by marker shape")


def fig_line_markers(outdir):
    """Each series a line with its own marker - the markers belong to their line."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 200)
    ys = [8 + 2.2 * x, 30 - 1.5 * x + 3 * np.sin(x), 18 + 4 * np.cos(0.7 * x)]
    styles = [("-", "o"), ("--", "s"), (":", "^")]
    series = []
    for i, (y, (ls, mk)) in enumerate(zip(ys, styles)):
        ax.plot(x, y, color=INK, linestyle=ls, linewidth=1.3, marker=mk, markevery=20,
                markersize=6, markerfacecolor="white", markeredgecolor=INK)
        series.append(_line(f"s{i}", x, y, ls, marker=mk))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    return _finish(fig, ax, "sc_line_markers", outdir, series,
                   notes="lines with markers every 20 points, crossing")


def fig_mic_like(outdir):
    """The case that prompted all this: a solid line, a dotted line and hollow
    triangles, all black, on a log axis, converging at high frequency."""
    with plt.rc_context({"font.family": "serif"}):
        fig, ax = _new_fig(figsize=(8.0, 4.4))
        f = np.geomspace(25, 5000, 260)
        base = 72 - 18 * np.log10(f / 25) ** 1.4
        des = base + 4 * np.exp(-((np.log10(f) - 2.7) / 0.12) ** 2)
        llr = base - 5 * np.exp(-((np.log10(f) - 2.0) / 0.5) ** 2) \
            + 2 * np.exp(-((np.log10(f) - 2.7) / 0.1) ** 2)
        fm = np.arange(1, 280) * 16.0
        meas = 80 - 15 * np.log10(fm / 16) ** 1.3 + RNG.normal(0, 0.6, fm.size)
        ax.semilogx(f, des, color=INK, linestyle="-", linewidth=1.0, label="DES")
        ax.semilogx(f, llr, color=INK, linestyle=":", linewidth=1.2, label="LLR")
        ax.semilogx(fm, meas, linestyle="none", marker="^", markersize=6,
                    markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.8,
                    label="measured")
        ax.set_xlim(10, 5000)
        ax.set_ylim(0, 90)
        ax.set_ylabel("SPL [dB]")
        ax.set_title("microphone")
        ax.legend(loc="lower left")
        return _finish(fig, ax, "sc_mic_like", outdir, [
            _line("DES", f, des, "-"),
            _line("LLR", f, llr, ":"),
            dict(name="measured", x=fm, y=meas, color=INK, marker="^", linestyle="none"),
        ], notes="solid + dotted + hollow triangles, log x, converging")


def fig_shared_colour(outdir):
    """Two blue series and one orange: colour splits off orange, style splits the blues."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y1 = 25 + 8 * np.sin(0.5 * x)
    y2 = 12 + 1.5 * x
    y3 = 38 - 1.2 * x
    ax.plot(x, y1, color="#1f77b4", linestyle="-", linewidth=1.8)
    ax.plot(x, y2, color="#1f77b4", linestyle="--", linewidth=1.8)
    ax.plot(x, y3, color="#ff7f0e", linestyle="-", linewidth=1.8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 45)
    return _finish(fig, ax, "sc_shared_colour", outdir, [
        _line("blue solid", x, y1, "-", color="#1f77b4"),
        _line("blue dashed", x, y2, "--", color="#1f77b4"),
        _line("orange", x, y3, "-", color="#ff7f0e"),
    ], notes="two series share a colour, one does not")


def fig_dashdot_crossing(outdir):
    """Solid and dash-dot crossing at a shallow angle."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y1 = 10 + 2.5 * x
    y2 = 30 - 0.8 * x + 2 * np.sin(x)
    ax.plot(x, y1, color=INK, linestyle="-", linewidth=1.5)
    ax.plot(x, y2, color=INK, linestyle="-.", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    return _finish(fig, ax, "sc_dashdot_crossing", outdir, [
        _line("solid", x, y1, "-"), _line("dashdot", x, y2, "-."),
    ], notes="solid and dash-dot, one shallow crossing")


def fig_fan(outdir):
    """Four solid lines fanning out from one point: merged at the origin, apart later."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    series = []
    for i, k in enumerate((0.8, 1.6, 2.6, 3.8)):
        y = 2 + k * x
        ax.plot(x, y, color=INK, linewidth=1.4)
        series.append(_line(f"k={k}", x, y, "-"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 42)
    return _finish(fig, ax, "sc_fan", outdir, series,
                   notes="four solid lines sharing a start point")


def fig_partial(outdir):
    """Curves covering different parts of the x range."""
    fig, ax = _new_fig()
    xa = np.linspace(0, 6, 200)
    xb = np.linspace(4, 10, 200)
    xc = np.linspace(0, 10, 300)
    ya = 30 - 2 * xa
    yb = 8 + 3 * (xb - 4)
    yc = 14 + 3 * np.sin(0.9 * xc)
    ax.plot(xa, ya, color=INK, linewidth=1.5)
    ax.plot(xb, yb, color=INK, linewidth=1.5)
    ax.plot(xc, yc, color=INK, linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    return _finish(fig, ax, "sc_partial", outdir, [
        _line("left", xa, ya, "-"), _line("right", xb, yb, "-"), _line("dashed", xc, yc, "--"),
    ], notes="solid curves on different x ranges plus a full-width dashed one")


def fig_grey_levels(outdir):
    """Black, grey and light grey: colour separation's job, and must stay that way."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    ys = [30 + 5 * np.sin(0.7 * x), 20 + 5 * np.sin(0.7 * x + 1), 10 + 5 * np.sin(0.7 * x + 2)]
    colours = ["#000000", "#666666", "#aaaaaa"]
    for y, c in zip(ys, colours):
        ax.plot(x, y, color=c, linewidth=1.8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    return _finish(fig, ax, "sc_grey_levels", outdir,
                   [_line(f"grey {c}", x, y, "-", color=c) for y, c in zip(ys, colours)],
                   notes="three grey levels - colour, not style")


def build(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    records = [
        fig_styles(outdir),
        fig_solid_three(outdir),
        fig_crossing(outdir),
        fig_markers(outdir),
        fig_line_markers(outdir),
        fig_mic_like(outdir),
        fig_shared_colour(outdir),
        fig_dashdot_crossing(outdir),
        fig_fan(outdir),
        fig_partial(outdir),
        fig_grey_levels(outdir),
    ]
    manifest = {"figures": records}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv):
    outdir = Path(argv[1]) if len(argv) > 1 else \
        Path(__file__).resolve().parents[1] / "tests" / "samecolour"
    manifest = build(outdir)
    for record in manifest["figures"]:
        print(f"{record['name']:22s} {record['width']}x{record['height']}  {record['notes']}")
    print(f"\n{len(manifest['figures'])} figures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
