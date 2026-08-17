"""Reading the short numeric strings that sit beside tick marks.

Two engines share one interface:

``TemplateOCR`` is the default and needs no model and no network. Axis labels are a
uniquely easy OCR problem - a handful of characters from a fixed set, axis-aligned,
rendered rather than photographed - and the machinery here exploits that by rendering
the same characters from real font files at the *exact pixel size* found in the image
and comparing directly. Matching at native resolution rather than upscaling the crop
is what keeps small labels legible.

``OnnxPPOCR`` (in :mod:`plotdigitizer.models`) is an optional upgrade for scanned or
unusual figures. It is never required: if it cannot be loaded, the template engine
takes over silently.

The vertical position of a glyph within the text line carries as much information as
its shape - it is the only thing separating a full stop from a minus sign - so glyphs
are always compared in a canvas that preserves where they sat on the line.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

__all__ = ["Glyph", "TextLine", "TemplateOCR", "parse_number", "parse_offset",
           "split_superscript", "get_default_ocr"]

# Everything that can appear in a numeric axis label. Unicode minus and the various
# multiplication signs are folded onto their ASCII equivalents after recognition.
CHARSET = "0123456789.,-−+eE×x"

_FOLD = {"−": "-", "×": "x", "E": "e"}


@dataclass
class Glyph:
    """One connected blob of ink from a text line, with its tight bounding box."""

    bitmap: np.ndarray   # float32 in 0..1, ink intensity, cropped to the blob
    x0: int
    y0: int
    x1: int              # exclusive
    y1: int              # exclusive

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)


@dataclass
class TextLine:
    """A group of glyphs forming one label, plus what it was read as."""

    glyphs: list[Glyph]
    text: str = ""
    value: float | None = None
    confidence: float = 0.0

    @property
    def x0(self) -> int:
        return min(g.x0 for g in self.glyphs)

    @property
    def x1(self) -> int:
        return max(g.x1 for g in self.glyphs)

    @property
    def y0(self) -> int:
        return min(g.y0 for g in self.glyphs)

    @property
    def y1(self) -> int:
        return max(g.y1 for g in self.glyphs)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)


# --------------------------------------------------------------------------------------
# fonts
# --------------------------------------------------------------------------------------


def _candidate_font_files() -> list[Path]:
    """Font files worth matching against, most likely first.

    matplotlib ships DejaVu, which renders the majority of scientific figures found in
    the wild. System sans and serif faces are added for figures produced by other tools.
    """
    paths: list[Path] = []
    try:
        import matplotlib
        bundled = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf",
                     "DejaVuSansMono.ttf", "cmr10.ttf", "STIXGeneral.ttf"):
            candidate = bundled / name
            if candidate.exists():
                paths.append(candidate)
    except Exception as exc:  # pragma: no cover - matplotlib is a hard dependency
        log.debug("could not locate matplotlib fonts: %s", exc)

    wanted = ("liberationsans-regular", "liberationserif-regular", "arial", "helvetica",
              "nimbussans-regular", "times", "timesnewroman", "calibri", "verdana")
    try:
        from matplotlib import font_manager
        for entry in font_manager.fontManager.ttflist:
            stem = Path(entry.fname).stem.lower().replace(" ", "").replace("_", "")
            if any(stem.startswith(w) for w in wanted) and Path(entry.fname).exists():
                paths.append(Path(entry.fname))
    except Exception as exc:  # pragma: no cover
        log.debug("font manager scan failed: %s", exc)

    seen: set[str] = set()
    unique = []
    for p in paths:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique[:8]


def _ink_bbox(arr: np.ndarray, threshold: int = 40):
    ys, xs = np.nonzero(arr > threshold)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _render(font: ImageFont.FreeTypeFont, text: str) -> tuple[np.ndarray, int, int] | None:
    """Render text and return (tight bitmap, top offset, left offset) from a fixed origin."""
    pad = 60
    canvas = Image.new("L", (pad * 4, pad * 3), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), text, fill=255, font=font)
    arr = np.asarray(canvas)
    box = _ink_bbox(arr)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1].astype(np.float32) / 255.0, y0, x0


@lru_cache(maxsize=64)
def _font_for_digit_height(path_str: str, digit_height: int) -> ImageFont.FreeTypeFont | None:
    """Find the point size at which this font's digits are ``digit_height`` px tall."""
    path = Path(path_str)
    lo, hi = 4, max(12, digit_height * 6)
    best = None
    for _ in range(24):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(str(path), mid)
        except Exception:
            return None
        rendered = _render(font, "0")
        if rendered is None:
            return None
        height = rendered[0].shape[0]
        best = font
        if height < digit_height:
            lo = mid
        elif height > digit_height:
            hi = mid
        else:
            return font
    return best


def _canvas_size(digit_height: int) -> tuple[int, int]:
    """A canvas generous enough for the widest glyph plus room above and below."""
    pad = max(2, digit_height // 4)
    return digit_height + 2 * pad, int(round(1.6 * digit_height)) + 2 * pad


def _place(bitmap: np.ndarray, top_offset: float, digit_height: int) -> np.ndarray:
    """Draw a glyph into the comparison canvas at its position on the text line.

    ``top_offset`` is how far the glyph's top sits below the top of a digit. Keeping it
    is what distinguishes '.' (bottom) from '-' (middle) from '0' (full height).
    """
    height, width = _canvas_size(digit_height)
    pad = max(2, digit_height // 4)
    canvas = np.zeros((height, width), dtype=np.float32)

    gh, gw = bitmap.shape
    row = int(round(pad + top_offset))
    col = int(round((width - gw) / 2.0))

    r0, c0 = max(0, row), max(0, col)
    r1, c1 = min(height, row + gh), min(width, col + gw)
    if r1 <= r0 or c1 <= c0:
        return canvas
    canvas[r0:r1, c0:c1] = bitmap[r0 - row:r1 - row, c0 - col:c1 - col]
    return canvas


#: How far a glyph may be nudged before matching. Rasterisation puts a character on a
#: pixel grid wherever it happens to land, so the same '5' can sit a pixel left of where
#: the template renders it. Without this slack a full stop - two pixels of ink in an
#: otherwise empty canvas - never correlates with its own template.
_SHIFT_TOLERANCE = 2


def _correlate(query: np.ndarray, template: np.ndarray, shift: int = _SHIFT_TOLERANCE) -> float:
    """Best normalised correlation over small translations of the glyph."""
    if query.shape != template.shape:
        return 0.0
    padded = np.pad(query, shift, mode="constant")
    response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
    if response.size == 0:
        return 0.0
    return float(np.nanmax(response))


def split_touching(glyphs: list[Glyph], digit_height: int) -> list[Glyph]:
    """Cut apart characters that ran together into one connected component.

    In several serif faces - Computer Modern above all, so any figure out of LaTeX -
    the bowls of adjacent zeros touch, and '100' arrives as two components rather than
    three. Recognised as-is, the merged pair matches nothing and the whole label is
    discarded, which is how a perfectly legible axis ends up uncalibrated.

    Digits share one advance width in almost every face, so a component far wider than
    its neighbours is holding more than one of them. It is cut at the deepest valleys
    in its own column-ink profile, searched near where evenly spaced characters would
    meet. A cut is only made where a real valley exists, so a genuinely wide single
    glyph is left alone.
    """
    if not glyphs:
        return glyphs

    # A digit's inked width is a fairly stable fraction of its height across faces -
    # about 0.55 for monospace, 0.65 for Computer Modern, 0.72 for DejaVu Sans - so the
    # height is the scale to measure against. Estimating the character width from the
    # label's own glyphs does not work: '1' is half the width of '0', and taking it as
    # the unit chops every other digit into pieces.
    unit = 0.65 * digit_height
    if unit < 2:
        return glyphs

    out: list[Glyph] = []
    for glyph in glyphs:
        # A single character is never much wider than it is tall.
        if glyph.width <= 1.15 * digit_height:
            out.append(glyph)
            continue

        parts = max(2, int(round(glyph.width / unit)))
        profile = glyph.bitmap.sum(axis=0)

        cuts: list[int] = []
        for k in range(1, parts):
            centre = int(round(k * glyph.width / parts))
            window = max(2, int(0.25 * unit))
            lo, hi = max(1, centre - window), min(glyph.width - 1, centre + window + 1)
            if hi <= lo:
                continue
            local = lo + int(np.argmin(profile[lo:hi]))
            # Only cut where the ink genuinely thins out. Searching a narrow window
            # around the expected boundary is what keeps this from cutting through the
            # hollow middle of a '0', which is a deeper valley than the join itself.
            if profile[local] <= 0.6 * float(np.median(profile)):
                cuts.append(local)

        if not cuts:
            out.append(glyph)
            continue

        edges = [0, *cuts, glyph.width]
        for start, end in zip(edges[:-1], edges[1:]):
            if end - start < 2:
                continue
            piece = glyph.bitmap[:, start:end]
            rows = np.flatnonzero(piece.sum(axis=1) > 0)
            if rows.size == 0:
                continue
            out.append(Glyph(
                bitmap=piece[rows[0]:rows[-1] + 1],
                x0=glyph.x0 + start, y0=glyph.y0 + int(rows[0]),
                x1=glyph.x0 + end, y1=glyph.y0 + int(rows[-1]) + 1,
            ))
    return out or glyphs


def split_superscript(glyphs: list[Glyph]) -> tuple[list[Glyph], list[Glyph]]:
    """Separate a trailing superscript run from the base of a label.

    Only a *trailing* run counts. A leading minus sign also sits above the baseline,
    so anchoring on "raised characters at the end of the label" is what stops '-20'
    from being read as a power of ten.
    """
    if len(glyphs) < 2:
        return list(glyphs), []

    ordered = sorted(glyphs, key=lambda g: g.x0)
    digit_height = max(g.height for g in ordered)
    baseline = float(np.median([g.y1 for g in ordered]))
    raised = baseline - 0.2 * digit_height

    split = len(ordered)
    while split > 0 and ordered[split - 1].y1 <= raised:
        split -= 1

    # Every glyph raised means there is no baseline to compare against.
    if split == 0 or split == len(ordered):
        return ordered, []
    return ordered[:split], ordered[split:]


class TemplateOCR:
    """Recognise axis labels by rendering the same characters from real fonts."""

    name = "template"

    def __init__(self, fonts: "list[Path] | None" = None, min_confidence: float = 0.55):
        self._fonts = [str(p) for p in (fonts or _candidate_font_files())]
        self.min_confidence = min_confidence
        if not self._fonts:  # pragma: no cover - matplotlib always ships fonts
            log.warning("no usable font files found; label recognition will be disabled")

    @lru_cache(maxsize=32)
    def _bank(self, digit_height: int) -> tuple[tuple[str, np.ndarray], ...]:
        """Rendered templates for every character at this exact digit height."""
        entries: list[tuple[str, np.ndarray]] = []
        for path in self._fonts:
            font = _font_for_digit_height(path, digit_height)
            if font is None:
                continue
            zero = _render(font, "0")
            if zero is None:
                continue
            zero_top = zero[1]
            for char in CHARSET:
                rendered = _render(font, char)
                if rendered is None:
                    continue
                bitmap, top, _ = rendered
                entries.append((char, _place(bitmap, top - zero_top, digit_height)))
        return tuple(entries)

    def _reference_height(self, glyphs: list[Glyph]) -> tuple[int, float]:
        """The digit height for this line, and the top row digits occupy."""
        heights = np.array([g.height for g in glyphs], dtype=float)
        digit_height = float(heights.max())
        tall = [g for g in glyphs if g.height >= 0.8 * digit_height]
        digit_top = float(min(g.y0 for g in tall)) if tall else float(min(g.y0 for g in glyphs))
        return int(round(digit_height)), digit_top

    def _read_run(self, glyphs: list[Glyph]) -> tuple[str, list[float]]:
        """Recognise a run of glyphs that all share one text size."""
        digit_height, digit_top = self._reference_height(glyphs)
        if digit_height < 3:
            return "", []
        glyphs = split_touching(glyphs, digit_height)
        bank = self._bank(digit_height)
        if not bank:
            return "", []

        chars: list[str] = []
        scores: list[float] = []
        for glyph in sorted(glyphs, key=lambda g: g.x0):
            canvas = _place(glyph.bitmap, glyph.y0 - digit_top, digit_height)
            best_char, best_score = "", -1.0
            for char, template in bank:
                score = _correlate(canvas, template)
                if score > best_score:
                    best_char, best_score = char, score
            chars.append("?" if best_score < self.min_confidence
                         else _FOLD.get(best_char, best_char))
            scores.append(max(0.0, best_score))
        return "".join(chars), scores

    def read_line(self, glyphs: list[Glyph]) -> tuple[str, float]:
        """Recognise one run of same-sized glyphs. Returns (text, mean confidence).

        Superscripts are separated by the caller, so this only ever sees text at one
        size - which is also what lets the exponent of a log label be matched against
        templates rendered at the exponent's own smaller size.
        """
        if not glyphs:
            return "", 0.0
        text, scores = self._read_run(list(glyphs))
        return text, float(np.mean(scores)) if scores else 0.0


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


def parse_number(text: str) -> float | None:
    """Turn a recognised label into a float, or None when it is not a clean number.

    Refusing to guess is deliberate: a mangled label that still parses would become a
    calibration point and quietly bend the whole axis. The RANSAC fit can survive one
    bad value, but it is cheaper never to produce one.
    """
    if not text or "?" in text:
        return None
    cleaned = text.strip()
    for src, dst in (("−", "-"), ("–", "-"), ("—", "-"), (" ", "")):
        cleaned = cleaned.replace(src, dst)

    # Thousands separators, but only in the 1,234,567 pattern - a lone comma in a
    # European-style decimal would otherwise silently multiply the value by 1000.
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", cleaned):
        cleaned = cleaned.replace(",", "")
    elif re.fullmatch(r"[+-]?\d+,\d+", cleaned):
        cleaned = cleaned.replace(",", ".")

    # Log axes label their decades as a power, e.g. '10^-2'.
    if "^" in cleaned:
        base_text, _, exp_text = cleaned.partition("^")
        base = parse_number(base_text)
        exponent = parse_number(exp_text)
        if base is None or exponent is None or base <= 0:
            return None
        try:
            value = float(base) ** float(exponent)
        except (ValueError, OverflowError):
            return None
        return value if np.isfinite(value) else None

    if not _NUMBER_RE.match(cleaned):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


_OFFSET_RE = re.compile(
    r"^(?:(?P<mant>[+-]?\d+(?:\.\d+)?)\s*[ex]\s*)?"      # optional leading mantissa
    r"(?:10)?\s*\^?\s*(?P<exp>[+-]?\d+)$",
    re.IGNORECASE,
)


def parse_offset(text: str) -> float | None:
    """Interpret a corner multiplier such as '1e-3' or 'x10-3' as a scale factor."""
    if not text or "?" in text:
        return None
    cleaned = text.strip().replace("−", "-").replace("×", "x").replace(" ", "")
    cleaned = cleaned.lstrip("x")
    if not cleaned:
        return None

    direct = parse_number(cleaned)
    if direct is not None and direct != 0:
        # A bare '1e-3' parses straight to a float and is already the multiplier.
        return float(direct)

    match = _OFFSET_RE.match(cleaned)
    if not match:
        return None
    mantissa = float(match.group("mant")) if match.group("mant") else 1.0
    try:
        return float(mantissa * 10.0 ** float(match.group("exp")))
    except (ValueError, OverflowError):
        return None


_DEFAULT: "TemplateOCR | None" = None


def get_default_ocr(prefer_neural: bool = False):
    """The OCR engine to use, falling back to templates whenever anything is missing."""
    global _DEFAULT
    if prefer_neural:
        try:
            from ..models import load_onnx_ocr
            engine = load_onnx_ocr()
            if engine is not None:
                return engine
        except Exception as exc:
            log.info("neural OCR unavailable (%s); using template matching", exc)
    if _DEFAULT is None:
        _DEFAULT = TemplateOCR()
    return _DEFAULT
