"""Generate a corpus of test figures together with exact ground truth.

Every figure is rendered through the Agg canvas and saved from that same raster, so
the pixel coordinates recorded here are the pixel coordinates in the PNG - there is
no resampling step in between to introduce a half-pixel drift.

For each figure we record:
  * the axes bounding box in pixels,
  * every major tick's value and its pixel position,
  * every series' data values and the pixel position of each point.

That is what turns "the detector works" into a measurable claim: the tests compare
detected numbers against these, not against eyeballed expectations.

Usage:  python tools/make_test_plots.py [outdir]
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

# Deterministic data across runs so the recorded truth stays stable.
RNG = np.random.default_rng(20260814)


# --------------------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------------------


def _render(fig) -> np.ndarray:
    """Rasterise a figure to an RGB array using the same canvas the transforms use."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _to_pixels(ax, xy: np.ndarray, height: int) -> np.ndarray:
    """Data coordinates -> (column, row) pixel coordinates with a top-left origin."""
    disp = ax.transData.transform(np.asarray(xy, dtype=float))
    out = np.empty_like(disp)
    out[:, 0] = disp[:, 0]
    out[:, 1] = height - disp[:, 1]
    return out


def _axes_box(ax, height: int) -> dict:
    bbox = ax.get_window_extent()
    return {
        "left": float(bbox.x0),
        "right": float(bbox.x1),
        "top": float(height - bbox.y1),
        "bottom": float(height - bbox.y0),
    }


def _ticks(ax, axis: str, height: int) -> list[dict]:
    """Major tick values inside the current limits, with their pixel positions."""
    if axis == "x":
        values = np.asarray(ax.get_xticks(), dtype=float)
        lo, hi = sorted(ax.get_xlim())
        y_ref = ax.get_ylim()[0]
        keep = values[(values >= lo) & (values <= hi)]
        pts = np.column_stack([keep, np.full(keep.shape, y_ref)])
        px = _to_pixels(ax, pts, height)[:, 0]
    else:
        values = np.asarray(ax.get_yticks(), dtype=float)
        lo, hi = sorted(ax.get_ylim())
        x_ref = ax.get_xlim()[0]
        keep = values[(values >= lo) & (values <= hi)]
        pts = np.column_stack([np.full(keep.shape, x_ref), keep])
        px = _to_pixels(ax, pts, height)[:, 1]
    return [{"value": float(v), "pixel": float(p)} for v, p in zip(keep, px)]


def _series_record(ax, name, x, y, height, color, marker, linestyle) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    px = _to_pixels(ax, np.column_stack([x, y]), height)
    return {
        "name": name,
        "color": color,
        "marker": marker,
        "linestyle": linestyle,
        "x": x.tolist(),
        "y": y.tolist(),
        "px": px[:, 0].tolist(),
        "py": px[:, 1].tolist(),
    }


def _finish(fig, ax, name, outdir, series, notes="") -> dict:
    """Rasterise, save, and assemble the truth record for one figure."""
    img = _render(fig)
    height, width = img.shape[:2]
    path = outdir / f"{name}.png"
    Image.fromarray(img).save(path)

    record = {
        "name": name,
        "file": path.name,
        "width": int(width),
        "height": int(height),
        "notes": notes,
        "xscale": ax.get_xscale(),
        "yscale": ax.get_yscale(),
        "xlim": [float(v) for v in ax.get_xlim()],
        "ylim": [float(v) for v in ax.get_ylim()],
        "axes_box": _axes_box(ax, height),
        "xticks": _ticks(ax, "x", height),
        "yticks": _ticks(ax, "y", height),
        "series": [
            _series_record(ax, s["name"], s["x"], s["y"], height, s["color"], s["marker"], s["linestyle"])
            for s in series
        ],
    }
    plt.close(fig)
    return record


def _new_fig(figsize=(6.4, 4.8), dpi=100, style="default"):
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


# --------------------------------------------------------------------------------------
# the figures
# --------------------------------------------------------------------------------------


def fig_linear_scatter(outdir):
    """The base case: one scatter series, linear axes, default matplotlib spines."""
    fig, ax = _new_fig()
    x = np.linspace(0.5, 9.5, 18)
    y = 3.0 * x + 8.0 + RNG.normal(0, 2.0, x.size)
    ax.scatter(x, y, c="#1f77b4", marker="o", s=42, zorder=3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 50)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("response")
    return _finish(fig, ax, "linear_scatter", outdir,
                   [dict(name="series 1", x=x, y=y, color="#1f77b4", marker="o", linestyle="none")])


def fig_linear_line(outdir):
    """A single smooth curve - exercises the column-scan extractor."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y = 20 + 15 * np.sin(x) + 1.5 * x
    ax.plot(x, y, color="#d62728", linewidth=2.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return _finish(fig, ax, "linear_line", outdir,
                   [dict(name="series 1", x=x, y=y, color="#d62728", marker="none", linestyle="-")])


def fig_multi_scatter_legend(outdir):
    """Three coloured scatter series plus a legend - tests colour separation and naming."""
    fig, ax = _new_fig()
    x = np.linspace(1, 9, 14)
    specs = [
        ("alpha", "#1f77b4", "o", 2.0, 5.0),
        ("beta", "#2ca02c", "s", 3.5, 12.0),
        ("gamma", "#ff7f0e", "^", 5.0, 22.0),
    ]
    series = []
    for name, color, marker, slope, offset in specs:
        y = slope * x + offset + RNG.normal(0, 0.6, x.size)
        ax.scatter(x, y, c=color, marker=marker, s=46, label=name, zorder=3)
        series.append(dict(name=name, x=x, y=y, color=color, marker=marker, linestyle="none"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 70)
    ax.legend(loc="upper left", frameon=True)
    return _finish(fig, ax, "multi_scatter_legend", outdir, series)


def fig_log_y_line(outdir):
    """Semi-log y: the calibrator has to pick log over linear on its own."""
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 300)
    y = 0.05 * np.exp(0.6 * x)
    ax.semilogy(x, y, color="#9467bd", linewidth=2.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlabel("x")
    ax.set_ylabel("intensity")
    return _finish(fig, ax, "log_y_line", outdir,
                   [dict(name="series 1", x=x, y=y, color="#9467bd", marker="none", linestyle="-")])


def fig_loglog_scatter(outdir):
    """Both axes logarithmic."""
    fig, ax = _new_fig()
    x = np.logspace(0, 3, 16)
    y = 2.0 * x**0.75
    ax.loglog(x, y, linestyle="none", marker="D", color="#8c564b", markersize=6)
    ax.set_xlim(1, 1e3)
    ax.set_ylim(1, 1e3)
    return _finish(fig, ax, "loglog_scatter", outdir,
                   [dict(name="series 1", x=x, y=y, color="#8c564b", marker="D", linestyle="none")])


def fig_grid_box(outdir):
    """Gridlines plus a full box frame - the classic false-positive trap for spine detection."""
    fig, ax = _new_fig()
    x = np.linspace(0, 100, 25)
    y = 0.008 * (x - 50) ** 2 + 5
    ax.plot(x, y, color="#17becf", linewidth=2.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.grid(True, which="major", color="0.8", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return _finish(fig, ax, "grid_box", outdir,
                   [dict(name="series 1", x=x, y=y, color="#17becf", marker="none", linestyle="-")],
                   notes="gridlines + full box")


def fig_negative_range(outdir):
    """Negative values on both axes - minus signs in the tick labels."""
    fig, ax = _new_fig()
    x = np.linspace(-5, 5, 21)
    y = x**3 / 10.0
    ax.plot(x, y, color="#e377c2", marker="o", markersize=5, linewidth=1.6)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-15, 15)
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.axvline(0, color="0.7", linewidth=0.8)
    return _finish(fig, ax, "negative_range", outdir,
                   [dict(name="series 1", x=x, y=y, color="#e377c2", marker="o", linestyle="-")],
                   notes="negative values, zero rules")


def fig_offset_text(outdir):
    """Tiny values so matplotlib emits a shared 1e-3 multiplier above the y axis."""
    fig, ax = _new_fig()
    x = np.linspace(0, 4, 30)
    y = 1e-3 * (1.5 + np.cos(x))
    ax.plot(x, y, color="#7f7f7f", linewidth=2.0)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3e-3)
    return _finish(fig, ax, "offset_text", outdir,
                   [dict(name="series 1", x=x, y=y, color="#7f7f7f", marker="none", linestyle="-")],
                   notes="y axis carries a x10^-3 offset label")


def fig_sci_offset(outdir):
    """Forced scientific notation, so the y axis carries a '1e-3' multiplier label.

    The tick labels read 0.0 .. 3.0 but the true values are a thousand times smaller;
    a digitizer that ignores the corner label is out by three orders of magnitude.
    """
    fig, ax = _new_fig()
    x = np.linspace(0, 4, 30)
    y = 1e-3 * (1.5 + np.cos(x))
    ax.plot(x, y, color="#2ca02c", linewidth=2.0)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3e-3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return _finish(fig, ax, "sci_offset", outdir,
                   [dict(name="series 1", x=x, y=y, color="#2ca02c", marker="none", linestyle="-")],
                   notes="y axis uses a 1e-3 scientific-notation offset")


def fig_dashed_multi(outdir):
    """Two dashed lines - the extractor must bridge the gaps rather than emit fragments."""
    fig, ax = _new_fig()
    x = np.linspace(0, 20, 400)
    y1 = 10 + 4 * np.sin(x / 2)
    y2 = 6 + 0.25 * x
    ax.plot(x, y1, color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(x, y2, color="#d62728", linestyle="-.", linewidth=2.0)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    return _finish(fig, ax, "dashed_multi", outdir, [
        dict(name="series 1", x=x, y=y1, color="#1f77b4", marker="none", linestyle="--"),
        dict(name="series 2", x=x, y=y2, color="#d62728", marker="none", linestyle="-."),
    ], notes="dashed and dash-dot lines")


def fig_highdpi_small(outdir):
    """Higher DPI with small fonts - stresses the glyph recogniser."""
    with plt.rc_context({"font.size": 7.0}):
        fig, ax = _new_fig(figsize=(5.0, 3.6), dpi=160)
        x = np.linspace(0, 1, 40)
        y = np.sqrt(x) * 100
        ax.plot(x, y, color="#2ca02c", marker="v", markersize=4, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("fraction")
        ax.set_ylabel("percent")
        return _finish(fig, ax, "highdpi_small", outdir,
                       [dict(name="series 1", x=x, y=y, color="#2ca02c", marker="v", linestyle="-")],
                       notes="dpi 160, 7pt font")


def fig_dark_style(outdir):
    """Light-on-dark figure - nothing may assume a white background."""
    fig, ax = _new_fig(style="dark_background")
    x = np.linspace(0, 6, 24)
    y = 50 * np.exp(-x / 3)
    ax.plot(x, y, color="#00e5ff", marker="o", markersize=5, linewidth=1.8)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 60)
    return _finish(fig, ax, "dark_style", outdir,
                   [dict(name="series 1", x=x, y=y, color="#00e5ff", marker="o", linestyle="-")],
                   notes="dark background style")


def fig_dense_scatter(outdir):
    """Many small markers, some touching - exercises the blob splitter."""
    fig, ax = _new_fig()
    x = np.linspace(0.2, 19.8, 60)
    y = 30 + 18 * np.sin(x / 3) + RNG.normal(0, 1.2, x.size)
    ax.scatter(x, y, c="#ff7f0e", marker="o", s=28, zorder=3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 60)
    return _finish(fig, ax, "dense_scatter", outdir,
                   [dict(name="series 1", x=x, y=y, color="#ff7f0e", marker="o", linestyle="none")],
                   notes="60 closely spaced markers")


def fig_inward_ticks(outdir):
    """Ticks pointing into the axes - the tick finder must try the inner band."""
    fig, ax = _new_fig()
    x = np.linspace(0, 8, 17)
    y = 100 - 8 * x
    ax.plot(x, y, color="#1f77b4", marker="s", markersize=5, linewidth=1.6)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 100)
    ax.tick_params(direction="in", length=6, width=1.2, top=True, right=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return _finish(fig, ax, "inward_ticks", outdir,
                   [dict(name="series 1", x=x, y=y, color="#1f77b4", marker="s", linestyle="-")],
                   notes="inward ticks on all four sides")


BUILDERS = [
    fig_linear_scatter,
    fig_linear_line,
    fig_multi_scatter_legend,
    fig_log_y_line,
    fig_loglog_scatter,
    fig_grid_box,
    fig_negative_range,
    fig_offset_text,
    fig_sci_offset,
    fig_dashed_multi,
    fig_highdpi_small,
    fig_dark_style,
    fig_dense_scatter,
    fig_inward_ticks,
]


def build_corpus(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    records = [builder(outdir) for builder in BUILDERS]
    manifest = {"figures": records}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv: list[str]) -> int:
    outdir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "tests" / "data"
    manifest = build_corpus(outdir)
    for rec in manifest["figures"]:
        print(f"{rec['name']:22s} {rec['width']}x{rec['height']}  "
              f"{len(rec['series'])} series  "
              f"{len(rec['xticks'])}/{len(rec['yticks'])} x/y ticks  {rec['notes']}")
    print(f"\n{len(manifest['figures'])} figures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
