"""Image loading, normalised to a plain uint8 RGB array."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["load_image", "SUPPORTED_SUFFIXES", "file_dialog_filter"]

SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")


def load_image(path: str | Path) -> np.ndarray:
    """Load any supported image as an (H, W, 3) uint8 RGB array.

    Transparency is composited onto white rather than dropped: a PNG exported from a
    plotting tool often has a transparent background, and simply discarding the alpha
    would turn it black and invert every downstream darkness test.
    """
    path = Path(path)
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8).copy()


def file_dialog_filter() -> str:
    """Qt file-dialog filter string covering the supported formats."""
    patterns = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
    return f"Images ({patterns});;All files (*)"
