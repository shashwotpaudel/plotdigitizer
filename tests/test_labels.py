"""Label reading: the OCR primitives, and end-to-end tick/value pairing."""

from __future__ import annotations

import numpy as np
import pytest

from plotdigitizer.detect.frame import analyse_ink, detect_frame
from plotdigitizer.detect.labels import detect_labels
from plotdigitizer.detect.ocr import (
    TemplateOCR,
    get_default_ocr,
    parse_number,
    parse_offset,
)
from plotdigitizer.detect.ticks import detect_ticks


@pytest.fixture(scope="session")
def ocr():
    return get_default_ocr()


class TestParseNumber:
    @pytest.mark.parametrize("text,expected", [
        ("0", 0.0), ("42", 42.0), ("-20", -20.0), ("−20", -20.0),
        ("3.5", 3.5), ("0.0030", 0.003), (".5", 0.5), ("+7", 7.0),
        ("1e-3", 1e-3), ("1E3", 1000.0), ("1,234", 1234.0), ("1,234,567", 1234567.0),
        ("10^-2", 0.01), ("10^3", 1000.0),
    ])
    def test_valid(self, text, expected):
        assert parse_number(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "abc", "1?3", "--5", "1.2.3", "time(s)", "?"])
    def test_rejected(self, text):
        assert parse_number(text) is None

    def test_unrecognised_character_poisons_the_whole_label(self):
        """Better to drop a label than to feed the fit a half-guessed number."""
        assert parse_number("1?0") is None


class TestParseOffset:
    @pytest.mark.parametrize("text,expected", [
        ("1e-3", 1e-3), ("1e−3", 1e-3), ("1e6", 1e6),
        ("x10-3", 1e-3), ("×10-3", 1e-3), ("10^5", 1e5),
    ])
    def test_valid(self, text, expected):
        assert parse_offset(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "response", "?"])
    def test_rejected(self, text):
        assert parse_offset(text) is None


class TestTemplateOCR:
    """Read text this project renders itself, isolating OCR from the detection stack."""

    @staticmethod
    def _glyphs_from_text(text: str, size: int = 14):
        from PIL import Image, ImageDraw, ImageFont
        from plotdigitizer.detect.ocr import _candidate_font_files
        from plotdigitizer.detect.labels import _extract_glyphs
        from plotdigitizer.detect.frame import analyse_ink

        font = ImageFont.truetype(str(_candidate_font_files()[0]), size)
        image = Image.new("RGB", (40 + 14 * len(text), 60), (255, 255, 255))
        ImageDraw.Draw(image).text((20, 20), text, fill=(0, 0, 0), font=font)
        arr = np.asarray(image, dtype=np.uint8)
        ink = analyse_ink(arr)
        return _extract_glyphs(ink, 0, arr.shape[0], 0, arr.shape[1])

    @pytest.mark.parametrize("text", [
        "0", "5", "42", "100", "-20", "3.5", "0.0030", "1e-3", "12.5", "678", "94",
    ])
    def test_round_trip(self, ocr, text):
        glyphs = self._glyphs_from_text(text)
        read, confidence = ocr.read_line(glyphs)
        assert read == text, f"read {read!r} for {text!r}"
        assert confidence > 0.5

    def test_empty_input(self, ocr):
        assert ocr.read_line([]) == ("", 0.0)

    def test_engine_is_usable_without_any_downloads(self):
        """The default engine must work offline - that is the whole point of it."""
        engine = TemplateOCR()
        glyphs = self._glyphs_from_text("250")
        assert engine.read_line(glyphs)[0] == "250"


def _axis_labels(figure, axis, ocr):
    ink = analyse_ink(figure.image)
    frame = detect_frame(figure.image, ink)
    tickset = detect_ticks(ink, frame, axis)
    return detect_labels(ink, frame, tickset, axis, ocr)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_every_tick_label_is_read_correctly(figure, axis, ocr):
    """The headline claim: read every axis label in the corpus, exactly."""
    result = _axis_labels(figure, axis, ocr)
    truth = figure.xticks if axis == "x" else figure.yticks

    assert len(result.pairs) == len(truth), (
        f"{figure.name}/{axis}: got {len(result.pairs)} labels, expected {len(truth)}; "
        f"read {[ln.text for ln in result.lines]}"
    )
    for pixel, value in result.pairs:
        nearest = min(truth, key=lambda t: abs(t["pixel"] - pixel))
        assert abs(nearest["pixel"] - pixel) < 2.0, f"{figure.name}/{axis}: stray label"
        assert value == pytest.approx(nearest["value"], rel=1e-6, abs=1e-12), (
            f"{figure.name}/{axis}: read {value} at {pixel:.1f}, expected {nearest['value']}"
        )


def test_scientific_offset_is_applied(corpus, ocr):
    """'1e-3' above the axis must scale the values, not be ignored."""
    result = _axis_labels(corpus["sci_offset"], "y", ocr)
    assert result.multiplier == pytest.approx(1e-3)
    assert max(v for _, v in result.pairs) == pytest.approx(3e-3)


def test_no_false_multiplier_on_ordinary_axes(corpus, ocr):
    """A tick label reading '10' must not be mistaken for a power-of-ten multiplier."""
    for name in ("linear_scatter", "grid_box", "dashed_multi"):
        for axis in ("x", "y"):
            assert _axis_labels(corpus[name], axis, ocr).multiplier == 1.0, f"{name}/{axis}"


def test_log_decades_read_as_powers(corpus, ocr):
    """Log axes are labelled 10^n; the exponent is set smaller and raised."""
    result = _axis_labels(corpus["log_y_line"], "y", ocr)
    values = sorted(v for _, v in result.pairs)
    assert values == pytest.approx([1e-2, 1e-1, 1e0, 1e1, 1e2])


def test_axis_titles_are_not_read_as_labels(corpus, ocr):
    """'response' is rotated alongside the y labels and must not become a number."""
    result = _axis_labels(corpus["linear_scatter"], "y", ocr)
    assert all(parse_number(line.text) is not None for line in result.lines), \
        f"unreadable text leaked in: {[ln.text for ln in result.lines]}"
