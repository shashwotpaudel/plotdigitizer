"""A held-out corpus, deliberately unlike the one the detector was tuned on.

Every threshold in this project was chosen by looking at ``tests/data``. Reporting
accuracy only on those figures would measure how well the constants were fitted, not
whether the approach generalises. These figures were written afterwards and exercise
things the main corpus never contained:

  * a grey plot background with *white* gridlines, which inverts the usual assumption
    that structure is darker than the paper,
  * JPEG compression, which smears every colour boundary,
  * serif and monospace typefaces,
  * markers whose outline is a different colour from their fill,
  * a plot title and an axis with only three ticks,
  * monochrome serif line art whose digits touch and whose only ink is black.

Usage:  python tools/make_holdout.py [outdir]
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

RNG = np.random.default_rng(70707)


def fig_ggplot_grey(outdir):
    """Grey panel with white gridlines - structure lighter than the background."""
    fig, ax = _new_fig(style="ggplot")
    x = np.linspace(0, 12, 22)
    y = 4 + 2.5 * np.sqrt(x) + RNG.normal(0, 0.25, x.size)
    ax.plot(x, y, color="#0072B2", linewidth=2.0)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    return _finish(fig, ax, "holdout_ggplot", outdir,
                   [dict(name="s", x=x, y=y, color="#0072B2", marker="none", linestyle="-")],
                   notes="ggplot style: grey panel, white gridlines")


def fig_serif_title(outdir):
    """Serif typeface, a title, and only a few ticks per axis."""
    with plt.rc_context({"font.family": "serif", "font.size": 11.0}):
        fig, ax = _new_fig()
        x = np.linspace(0, 30, 16)
        y = 200 - 4 * x
        ax.plot(x, y, color="#8B0000", marker="s", markersize=5, linewidth=1.5)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 220)
        ax.set_xticks([0, 15, 30])
        ax.set_yticks([0, 110, 220])
        ax.set_title("Decay measurement")
        return _finish(fig, ax, "holdout_serif", outdir,
                       [dict(name="s", x=x, y=y, color="#8B0000", marker="s", linestyle="-")],
                       notes="serif font, title, 3 ticks per axis")


def fig_edged_markers(outdir):
    """Markers with a dark outline around a pale fill."""
    fig, ax = _new_fig()
    x = np.linspace(1, 19, 20)
    y = 40 * np.exp(-x / 12) + 5
    ax.plot(x, y, linestyle="none", marker="o", markersize=9,
            markerfacecolor="#ffe066", markeredgecolor="#5f3dc4", markeredgewidth=2.0)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 50)
    return _finish(fig, ax, "holdout_edged", outdir,
                   [dict(name="s", x=x, y=y, color="#ffe066", marker="o", linestyle="none")],
                   notes="two-tone markers: pale fill, dark edge")


def fig_monospace_dense(outdir):
    """Monospace labels and a busy axis."""
    with plt.rc_context({"font.family": "monospace", "font.size": 8.0}):
        fig, ax = _new_fig(figsize=(7.0, 4.0))
        x = np.linspace(0, 1, 200)
        y = np.sin(2 * np.pi * x) * 0.4 + 0.5
        ax.plot(x, y, color="#087f5b", linewidth=1.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return _finish(fig, ax, "holdout_monospace", outdir,
                       [dict(name="s", x=x, y=y, color="#087f5b", marker="none", linestyle="-")],
                       notes="monospace 8pt, many ticks")


def fig_monochrome_log(outdir):
    """Black-ink line art on a log axis, in the style of a LaTeX journal figure.

    Nothing here is coloured, so the ink arrives as a fan of greys from anti-aliasing
    and must come back as one series rather than six. The serif digits also run into
    each other - the two zeros of '100' touch - which is what makes the labels
    unreadable unless merged components are split apart again.
    """
    with plt.rc_context({"font.family": "serif", "font.size": 13.0}):
        fig, ax = _new_fig(figsize=(7.0, 4.2))
        x = np.logspace(1, 3.7, 220)
        y = 90 - 18 * np.log10(x)
        ax.semilogx(x, y, color="black", linestyle=":", linewidth=1.4,
                    marker="^", markevery=12, markersize=6,
                    markerfacecolor="none", markeredgecolor="black")
        ax.set_xlim(10, 5000)
        ax.set_ylim(0, 90)
        ax.set_ylabel("SPL [dB]")
        return _finish(fig, ax, "holdout_monochrome", outdir,
                       [dict(name="s", x=x, y=y, color="#000000",
                             marker="^", linestyle=":")],
                       notes="monochrome serif line art, log x, touching digits")


def _resave(record, outdir: Path, name: str, notes: str, transform) -> dict:
    """Re-encode an existing figure's PNG, keeping its ground truth."""
    source = Image.open(outdir / record["file"])
    target = outdir / f"{name}.png"
    transform(source, target)
    out = dict(record)
    out["name"] = name
    out["file"] = target.name
    out["notes"] = notes
    return out


def fig_jpeg_artifacts(outdir, base):
    """The same figure pushed through lossy JPEG, as a shared screenshot would be."""
    def transform(image, target):
        jpeg = target.with_suffix(".jpg")
        image.convert("RGB").save(jpeg, quality=55)
        Image.open(jpeg).convert("RGB").save(target)
        jpeg.unlink(missing_ok=True)
    return _resave(base, outdir, "holdout_jpeg", "JPEG quality 55 re-encode", transform)


def build(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    records = [
        fig_ggplot_grey(outdir),
        fig_serif_title(outdir),
        fig_edged_markers(outdir),
        fig_monospace_dense(outdir),
        fig_monochrome_log(outdir),
    ]
    records.append(fig_jpeg_artifacts(outdir, records[1]))
    manifest = {"figures": records}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv):
    outdir = Path(argv[1]) if len(argv) > 1 else \
        Path(__file__).resolve().parents[1] / "tests" / "holdout"
    manifest = build(outdir)
    for record in manifest["figures"]:
        print(f"{record['name']:22s} {record['width']}x{record['height']}  {record['notes']}")
    print(f"\n{len(manifest['figures'])} figures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
