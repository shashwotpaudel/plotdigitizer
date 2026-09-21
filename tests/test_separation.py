"""Series that share a colour, told apart by style, marker shape and continuity.

Two corpora are generated on first use. ``tests/samecolour`` is the development set the
thresholds were chosen on; ``tests/samecolour_holdout`` was written afterwards and is
the one that says whether the method generalises.

The most important test here is the dullest: on every figure the detector already
handled, the separator must decline, so that nothing that worked before changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from plotdigitizer.detect.frame import analyse_ink, detect_frame  # noqa: E402
from plotdigitizer.detect.separate import (  # noqa: E402
    SeparationSettings,
    _glyph_classes,
    _style_masks,
    separate_series,
    stroke_thickness,
)
from plotdigitizer.detect.series import discover_series  # noqa: E402
from plotdigitizer.detect.ticks import detect_ticks  # noqa: E402
from plotdigitizer.image_io import load_image  # noqa: E402
from plotdigitizer.pipeline import AutoDigitizer  # noqa: E402

from conftest import Figure  # noqa: E402

DEV_DIR = ROOT / "tests" / "samecolour"
HOLDOUT_DIR = ROOT / "tests" / "samecolour_holdout"
HARD_DIR = ROOT / "tests" / "hard"
ORIGINAL_DIRS = [ROOT / "tests" / "data", ROOT / "tests" / "holdout"]

#: Every series must be at least this pure: points leaking in from a neighbouring
#: curve are the failure that matters, because they look just as convincing.
MIN_PURITY = 0.9
#: Coverage is allowed to fall further, because where two curves genuinely coincide
#: there is nothing to recover. The lowest on the corpora is the dotted line of the
#: microphone figure, which vanishes under the solid one for a third of its length.
MIN_COVERAGE = 0.6
MEAN_COVERAGE = 0.9


def _ensure(directory: Path, module: str) -> dict:
    manifest = directory / "manifest.json"
    if not manifest.exists():
        __import__(module).build(directory)
    return json.loads(manifest.read_text())


def _ensure_originals() -> None:
    from conftest import _ensure_corpus
    _ensure_corpus()
    holdout = ROOT / "tests" / "holdout"
    if not (holdout / "manifest.json").exists():
        from make_holdout import build
        build(holdout)


def pytest_generate_tests(metafunc):  # pragma: no cover - collection hook
    if "samecolour_figure" in metafunc.fixturenames:
        records = []
        for directory, module in ((DEV_DIR, "make_samecolour"),
                                  (HOLDOUT_DIR, "make_samecolour_holdout")):
            records += [Figure(r, directory) for r in _ensure(directory, module)["figures"]]
        metafunc.parametrize("samecolour_figure", records, ids=[f.name for f in records])


@pytest.fixture(scope="module")
def digitizer():
    return AutoDigitizer(device="cpu")


def _scores(figure, digitizer):
    from separation_report import score_figure
    result = digitizer.run(figure.image)
    return result, score_figure(figure.record,
                                [(s.name, s.pixel_points) for s in result.series])


# -- the corpora ---------------------------------------------------------------------


def test_every_series_is_found(samecolour_figure, digitizer):
    result, _ = _scores(samecolour_figure, digitizer)
    assert len(result.series) == len(samecolour_figure.series), (
        f"{samecolour_figure.name}: {[s.style or s.hex_color for s in result.series]}")


def test_series_do_not_mix(samecolour_figure, digitizer):
    _, scores = _scores(samecolour_figure, digitizer)
    for score in scores:
        assert score.purity >= MIN_PURITY, (
            f"{samecolour_figure.name}/{score.truth}: purity {score.purity:.2f}")
        assert score.coverage >= MIN_COVERAGE, (
            f"{samecolour_figure.name}/{score.truth}: coverage {score.coverage:.2f}")


@pytest.mark.parametrize("directory,module", [(DEV_DIR, "make_samecolour"),
                                              (HOLDOUT_DIR, "make_samecolour_holdout")],
                         ids=["development", "holdout"])
def test_coverage_across_corpus(directory, module, digitizer):
    from separation_report import score_figure
    coverages = []
    for record in _ensure(directory, module)["figures"]:
        result = digitizer.run(load_image(directory / record["file"]))
        coverages += [s.coverage for s in score_figure(
            record, [(x.name, x.pixel_points) for x in result.series])]
    assert np.mean(coverages) >= MEAN_COVERAGE


def test_nothing_that_worked_before_is_split():
    """On the figures v0 already handled, every colour must come back unsplit."""
    _ensure_originals()
    for directory in ORIGINAL_DIRS:
        for record in json.loads((directory / "manifest.json").read_text())["figures"]:
            image = load_image(directory / record["file"])
            ink = analyse_ink(image)
            frame = detect_frame(image, ink)
            content = discover_series(
                image, ink, frame,
                x_ticks=detect_ticks(ink, frame, "x").positions,
                y_ticks=detect_ticks(ink, frame, "y").positions)
            for candidate in content.series:
                parts = separate_series(candidate.mask, plot_width=int(frame.width))
                assert parts is None, (
                    f"{record['name']} {candidate.hex_color} split into "
                    f"{[p.description for p in parts]}")


# -- the failures an independent review found ----------------------------------------

#: Each colour in these holds one series, or the separator has no business splitting it:
#: scatters in their own colour, shaded bands, related colours, text, closed and folded
#: curves, sizes of one marker, compression. They must come back as before separation.
KEEP_WHOLE = [
    "scatter_o_150a", "scatter_o_150b", "scatter_o_200", "scatter_diamond_100",
    "scatter_plus_300", "small_dots_ms2", "small_dots_ms3", "grey_band", "blue_band",
    "two_bands", "shaded_span", "watermark", "navy_and_light", "family_reds",
    "family_viridis", "ggplot_three", "raw_and_smoothed", "pale_points_dark_fit",
    "hidpi_line", "legend_unframed_one", "annotated", "ellipse", "voltammogram",
    "thin_loop", "z_fold", "sawtooth", "sawtooth_thin", "thick_dashdot_alone",
    "thick_dotted_alone", "bubble_sizes", "two_square_sizes", "jpeg_scatter", "jpeg_grid",
    "jpeg_legend", "white_markers_line", "white_markers_sine", "dashed_with_markers",
    "markers_on_dotted", "dashed_grid", "speckled_colours",
]
#: And these it must now separate, every series found and none mixed with another.
SEPARATE = [
    "x_dash_04", "x_dash_07", "x_dash_10", "x_dot_07", "x_solid_02", "partial_apart",
    "partial_touching", "partial_overlapping", "three_dashed_055", "three_dashed_04",
    "thick_line_markers", "converging_markers", "crossing_with_x_markers",
    "markers_diamond_circle", "markers_left_right", "markers_x_plus", "dash_and_dashdot",
    "dash_and_longdash", "dash_thin_and_thick", "data_and_fit", "errorbars_and_line",
    "errorbars_two_shapes", "errorbars_horizontal", "errorbars_dashed",
    "errorbars_journal", "hidpi_filled", "hidpi_markers", "hidpi_styles", "hidpi_dashdot",
    "legend_unframed_two", "legend_unframed_four", "journal_three", "markevery_25",
    "markevery_legend", "thin_and_round_dots", "family_greys", "dashdot_white_markers",
]


@pytest.fixture(scope="module")
def hard():
    return {record["name"]: record for record in _ensure(HARD_DIR, "make_hard_cases")["figures"]}


@pytest.mark.parametrize("name", KEEP_WHOLE)
def test_what_is_one_series_stays_one(name, hard, digitizer):
    result = digitizer.run(load_image(HARD_DIR / hard[name]["file"]))
    assert all(s.style == "" for s in result.series), \
        f"{name}: {[s.style for s in result.series]}"


@pytest.mark.parametrize("name", SEPARATE)
def test_what_shares_a_colour_is_told_apart(name, hard, digitizer):
    from separation_report import score_figure
    record = hard[name]
    result = digitizer.run(load_image(HARD_DIR / record["file"]))
    assert len(result.series) == len(record["series"]), \
        f"{name}: {[s.style or s.hex_color for s in result.series]}"
    for score in score_figure(record, [(s.name, s.pixel_points) for s in result.series]):
        assert score.purity >= 0.85, f"{name}/{score.truth}: purity {score.purity:.2f}"


def test_ink_left_unassigned_is_reported(hard, digitizer):
    """A short third curve beside two long ones is lost, and the warning says where."""
    result = digitizer.run(load_image(HARD_DIR / hard["short_curve"]["file"]))
    assert any("in none of the series" in w for w in result.warnings)


def test_can_be_turned_off():
    _ensure(DEV_DIR, "make_samecolour")
    image = load_image(DEV_DIR / "sc_styles.png")
    assert len(AutoDigitizer(device="cpu", separate=False).run(image).series) == 1
    assert len(AutoDigitizer(device="cpu").run(image).series) == 3


def test_separated_series_are_drawn_apart(digitizer):
    _ensure(DEV_DIR, "make_samecolour")
    result = digitizer.run(load_image(DEV_DIR / "sc_styles.png"))
    shown = [s.display_color for s in result.series]
    assert len(set(shown)) == len(shown), "each separated series needs its own colour"
    for colour in shown:
        # Against black ink, a series drawn black would be invisible.
        assert np.linalg.norm(np.asarray(colour, float)) > 90
    assert {s.style for s in result.series} == {"solid line", "dashed line", "dotted line"}
    assert all(s.hex_color == result.series[0].hex_color for s in result.series), \
        "the ink colour is still reported as the ink colour"


# -- the pieces, on drawings made to order -------------------------------------------


def _canvas(width=400, height=300):
    return np.zeros((height, width), np.uint8)


def test_filled_markers_do_not_set_the_stroke_width():
    image = _canvas()
    for x in range(30, 380, 30):
        cv2.circle(image, (x, 150), 6, 1, -1)
    assert stroke_thickness(image.astype(bool)) == 2.0

    line = _canvas()
    cv2.line(line, (10, 100), (390, 140), 1, 3)
    # cv2 draws a "thickness 3" line four to five pixels across.
    assert 3.0 <= stroke_thickness(line.astype(bool)) <= 5.5


AA = cv2.LINE_AA


def _poly(*corners):
    return lambda im, x, y: cv2.polylines(
        im, [np.array([[x + dx, y + dy] for dx, dy in corners])], True, 255, 1, AA)


@pytest.mark.parametrize("draw,name", [
    (lambda im, x, y: cv2.circle(im, (x, y), 5, 255, 1, AA), "hollow circle"),
    (lambda im, x, y: cv2.rectangle(im, (x - 5, y - 5), (x + 5, y + 5), 255, -1, AA),
     "filled square"),
    (_poly((0, -6), (-6, 5), (6, 5)), "hollow triangle"),
    (_poly((0, 6), (-6, -5), (6, -5)), "hollow down triangle"),
    (_poly((-6, 0), (5, -6), (5, 6)), "hollow left triangle"),
    (_poly((6, 0), (-5, -6), (-5, 6)), "hollow right triangle"),
    (_poly((0, -6), (6, 0), (0, 6), (-6, 0)), "hollow diamond"),
    (lambda im, x, y: (cv2.line(im, (x - 5, y), (x + 5, y), 255, 1, AA),
                       cv2.line(im, (x, y - 5), (x, y + 5), 255, 1, AA)), "plus"),
    (lambda im, x, y: (cv2.line(im, (x - 5, y - 5), (x + 5, y + 5), 255, 1, AA),
                       cv2.line(im, (x - 5, y + 5), (x + 5, y - 5), 255, 1, AA)), "cross"),
], ids=["circle", "square", "triangle", "down", "left", "right", "diamond", "plus", "cross"])
def test_markers_are_recognised_and_named(draw, name):
    # Drawn anti-aliased, as every plotting library draws them.
    image = _canvas()
    for x in range(30, 380, 40):
        draw(image, x, 80 + (x % 70))
    classes = _glyph_classes(image > 100, 1.0, SeparationSettings())
    assert len(classes) == 1
    assert classes[0].describe() == name


def test_dashes_line_up_and_debris_does_not():
    dashed = _canvas()
    for x in range(10, 390, 16):
        cv2.line(dashed, (x, 100 + x // 8), (x + 9, 100 + (x + 9) // 8), 1, 2)
    styles = _style_masks(dashed.astype(bool), 2.0)
    assert "dashed" in styles and styles["dashed"].sum() > 0.8 * dashed.sum()

    debris = _canvas()
    rng = np.random.default_rng(3)
    for _ in range(20):
        x, y = rng.integers(20, 380), rng.integers(20, 280)
        angle = rng.integers(0, 180)
        cv2.ellipse(debris, (int(x), int(y)), (6, 6), 0, int(angle), int(angle) + 70, 1, 2)
    assert "dashed" not in _style_masks(debris.astype(bool), 2.0)


def test_crossing_lines_keep_their_identity():
    image = _canvas()
    cv2.line(image, (10, 60), (390, 240), 1, 2)
    cv2.line(image, (10, 220), (390, 90), 1, 2)
    parts = separate_series(image.astype(bool), plot_width=380)
    assert parts is not None and len(parts) == 2
    slopes = []
    for part in parts:
        line = part.centreline
        slope = np.polyfit(line[:, 0], line[:, 1], 1)[0]
        residual = line[:, 1] - np.polyval(np.polyfit(line[:, 0], line[:, 1], 1), line[:, 0])
        assert abs(slope) > 0.2, "each track keeps its own direction through the crossing"
        assert np.abs(residual).max() < 3.0, "and does not swap lines half way"
        slopes.append(slope)
    assert slopes[0] * slopes[1] < 0, "the two parts are the two lines, not one line twice"


@pytest.mark.parametrize("slope", [0.2, 0.4, 0.7])
@pytest.mark.parametrize("style", ["dashed", "dotted"])
def test_shallow_crossings_keep_their_identity(style, slope):
    """Lines crossing at 10-35 degrees, where a short slope estimate used to swap them."""
    image = _canvas(500, 300)
    # matplotlib's proportions at a 1.5 pt line: 3.7 widths of dash, 1.6 of gap.
    period, ink = (11, 7) if style == "dashed" else (5, 2)
    for sign in (1, -1):
        for x in range(10, 490, period):
            y0, y1 = 150 + sign * slope * (x - 250), 150 + sign * slope * (x + ink - 250)
            cv2.line(image, (x, int(round(y0))), (x + ink, int(round(y1))), 1, 2)
    parts = separate_series(image.astype(bool), plot_width=480)
    assert parts is not None and len(parts) == 2
    for part in parts:
        line = part.centreline
        left = np.polyfit(*line[line[:, 0] < 200].T, 1)[0]
        right = np.polyfit(*line[line[:, 0] > 300].T, 1)[0]
        assert left * right > 0, f"a part turns back at the crossing ({left:.2f}, {right:.2f})"


def test_a_closed_curve_is_one_curve():
    image = _canvas()
    cv2.ellipse(image, (200, 150), (150, 80), 0, 0, 360, 1, 2)
    assert separate_series(image.astype(bool), plot_width=380) is None


def test_error_bars_stay_with_their_markers():
    image = _canvas(500, 300)
    cv2.line(image, (10, 40), (490, 260), 1, 2)
    for x in range(40, 480, 40):
        y = 250 - x // 3
        cv2.line(image, (x, y - 18), (x, y + 18), 1, 2)            # the bar
        cv2.line(image, (x - 4, y - 18), (x + 4, y - 18), 1, 1)    # and its caps
        cv2.line(image, (x - 4, y + 18), (x + 4, y + 18), 1, 1)
        cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), 1, -1)
    parts = separate_series(image.astype(bool), plot_width=480)
    assert parts is not None and len(parts) == 2
    markers = next(p for p in parts if p.kind == "markers")
    assert markers.description == "filled square markers"
    assert len(markers.points) == 11


def test_text_in_the_plot_is_not_a_series():
    image = np.zeros((300, 400, 3), np.uint8)
    cv2.line(image, (10, 250), (390, 60), (255, 255, 255), 2)
    cv2.putText(image, "measured response", (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    mask = image[..., 0] > 100
    assert separate_series(mask, plot_width=380) is None


def test_hollow_markers_on_a_line_do_not_derail_it():
    image = _canvas()
    cv2.line(image, (10, 150), (390, 150), 1, 2)
    for x in range(40, 380, 45):
        cv2.circle(image, (x, 150), 6, 0, -1)       # white-filled, as matplotlib draws
        cv2.circle(image, (x, 150), 6, 1, 1)
    cv2.line(image, (10, 60), (390, 100), 1, 1)      # a second, dotted-free line
    parts = separate_series(image.astype(bool), plot_width=380)
    assert parts is not None and len(parts) == 2
    flat = min(parts, key=lambda p: np.ptp(p.centreline[:, 1]))
    assert np.ptp(flat.centreline[:, 1]) < 3.0
