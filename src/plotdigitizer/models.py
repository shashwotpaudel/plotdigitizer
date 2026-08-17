"""Optional neural OCR: a PP-OCR recognition model run through ONNX Runtime.

This is an *upgrade*, never a requirement. The template matcher in
:mod:`plotdigitizer.detect.ocr` reads rendered axis labels essentially perfectly and
needs no network, so it stays the default. The neural model earns its place on figures
the template bank was not built for - scans, photographs of printed pages, unusual or
condensed typefaces - where matching against clean glyph renders breaks down.

Nothing is downloaded until something asks for it, weights are checked against a known
size before use, and every failure path - no network, no onnxruntime, a CUDA provider
that will not load - falls back to the template engine rather than raising. A digitizer
that stops working because a model host is down would be a bad trade for the accuracy
it buys.

The same file serves CPU and GPU: the CUDA execution provider is requested first and
ONNX Runtime silently uses the CPU one when it is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["ModelSpec", "REC_MODEL", "CHAR_DICT", "model_dir", "ensure_model",
           "load_onnx_ocr", "OnnxPPOCR"]


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable artefact, with enough information to validate it."""

    name: str
    url: str
    filename: str
    approx_bytes: int
    sha256: str | None = None       # pinned once known; size is the fallback check

    def path(self) -> Path:
        return model_dir() / self.filename


#: PP-OCRv4 recognition weights, as republished by the RapidOCR project. Recognition
#: only - the labels' locations are already known from the tick geometry, so a text
#: *detection* model would be answering a question we have a better answer to.
REC_MODEL = ModelSpec(
    name="PP-OCRv4 recognition",
    url="https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx",
    filename="ch_PP-OCRv4_rec_infer.onnx",
    approx_bytes=10_900_000,
)

#: The character table the model's CTC head indexes into.
CHAR_DICT = ModelSpec(
    name="PP-OCR character dictionary",
    url="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt",
    filename="ppocr_keys_v1.txt",
    approx_bytes=26_000,
)


def model_dir() -> Path:
    """Where downloaded weights live. Honours XDG_CACHE_HOME."""
    override = os.environ.get("PLOTDIGITIZER_MODEL_DIR")
    if override:
        return Path(override)
    cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(cache) / "plot_digitizer" / "models"


def _looks_complete(path: Path, spec: ModelSpec) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    if spec.sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec.sha256:
            log.warning("%s failed its checksum; discarding", path.name)
            return False
        return True
    # Without a pinned hash, guard against a truncated download.
    return size > 0.5 * spec.approx_bytes


def ensure_model(spec: ModelSpec, allow_download: bool = True,
                 timeout: float = 60.0) -> Path | None:
    """Return a local path to the artefact, downloading it once if permitted."""
    target = spec.path()
    if _looks_complete(target, spec):
        return target
    if not allow_download:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    log.info("downloading %s (~%.1f MB) to %s",
             spec.name, spec.approx_bytes / 1e6, target.parent)
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "plotdigitizer"})
        with urllib.request.urlopen(request, timeout=timeout) as response, \
                open(partial, "wb") as handle:
            while chunk := response.read(1 << 16):
                handle.write(chunk)
        partial.replace(target)
    except Exception as exc:                       # noqa: BLE001 - offline is normal
        log.warning("could not download %s (%s); staying with template OCR", spec.name, exc)
        partial.unlink(missing_ok=True)
        return None

    if not _looks_complete(target, spec):
        log.warning("%s looks incomplete after download; discarding", spec.name)
        target.unlink(missing_ok=True)
        return None
    return target


class OnnxPPOCR:
    """Recognises a text line with a PP-OCR CTC recognition model.

    The interface matches :class:`~plotdigitizer.detect.ocr.TemplateOCR` exactly, so
    the two are interchangeable everywhere in the pipeline.
    """

    name = "onnx"
    #: The model was trained on 48-pixel-tall lines.
    HEIGHT = 48

    def __init__(self, session, charset: list[str]):
        self._session = session
        self._charset = charset
        self._input = session.get_inputs()[0].name

    # -- preprocessing -----------------------------------------------------------

    def _compose_line(self, glyphs) -> np.ndarray:
        """Rebuild one greyscale image of the whole label from its glyph crops.

        The glyphs were split apart by connected components; the recogniser expects a
        line, so they are pasted back at their original relative positions - spacing is
        information, and 0.5 read as 05 would be a silent error.
        """
        x0 = min(g.x0 for g in glyphs)
        y0 = min(g.y0 for g in glyphs)
        x1 = max(g.x1 for g in glyphs)
        y1 = max(g.y1 for g in glyphs)
        canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        for glyph in glyphs:
            region = canvas[glyph.y0 - y0:glyph.y1 - y0, glyph.x0 - x0:glyph.x1 - x0]
            np.maximum(region, glyph.bitmap, out=region)
        # The model wants dark text on light paper.
        return 1.0 - np.clip(canvas, 0.0, 1.0)

    def _prepare(self, line: np.ndarray) -> np.ndarray:
        height, width = line.shape
        scale = self.HEIGHT / max(1, height)
        target_width = max(16, int(round(width * scale)))
        resized = cv2.resize(line, (target_width, self.HEIGHT), interpolation=cv2.INTER_CUBIC)
        # Pad rather than stretch: distorting the aspect ratio hurts short strings most,
        # and axis labels are short strings.
        padded_width = max(32, int(np.ceil(target_width / 8) * 8))
        padded = np.ones((self.HEIGHT, padded_width), dtype=np.float32)
        padded[:, :target_width] = resized
        rgb = np.repeat(padded[None, :, :], 3, axis=0)
        return ((rgb - 0.5) / 0.5)[None].astype(np.float32)

    def _decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy CTC decode: collapse repeats, drop blanks."""
        indices = logits.argmax(axis=1)
        confidences = logits.max(axis=1)
        characters: list[str] = []
        scores: list[float] = []
        previous = -1
        for index, confidence in zip(indices, confidences):
            if index != previous and index != 0:
                position = int(index) - 1
                if 0 <= position < len(self._charset):
                    characters.append(self._charset[position])
                    scores.append(float(confidence))
            previous = int(index)
        return "".join(characters), float(np.mean(scores)) if scores else 0.0

    # -- public API --------------------------------------------------------------

    def read_line(self, glyphs) -> tuple[str, float]:
        if not glyphs:
            return "", 0.0
        try:
            line = self._compose_line(list(glyphs))
            tensor = self._prepare(line)
            outputs = self._session.run(None, {self._input: tensor})
            return self._decode(np.asarray(outputs[0])[0])
        except Exception as exc:                    # noqa: BLE001
            log.warning("neural OCR failed on a label (%s)", exc)
            return "", 0.0


def _preload_cuda_libraries() -> bool:
    """Make pip-installed CUDA libraries visible to ONNX Runtime's provider .so.

    ``onnxruntime-gpu[cuda,cudnn]`` installs CUDA and cuDNN as ordinary wheels under
    ``site-packages/nvidia/``, but its CUDA provider library is dlopen'd by the loader,
    which does not search there. Without help it fails with a bare "libcublasLt.so.13:
    cannot open shared object file" and silently drops to the CPU - on a machine with a
    perfectly good GPU sitting idle.

    Opening the libraries here with RTLD_GLOBAL puts their symbols in the process
    before the provider needs them. Two passes, because the load order matters and the
    dependency graph between them is not worth hard-coding.
    """
    import ctypes
    import sysconfig

    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        directory = sysconfig.get_paths().get(key)
        if directory:
            candidate = Path(directory) / "nvidia"
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    if not roots:
        return False

    libraries: list[Path] = []
    for root in roots:
        libraries.extend(sorted(root.glob("*/lib/*.so*")))
    if not libraries:
        return False

    loaded = 0
    remaining = libraries
    for _ in range(2):
        failed: list[Path] = []
        for library in remaining:
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                failed.append(library)
        if not failed:
            break
        remaining = failed

    log.debug("preloaded %d/%d CUDA libraries", loaded, len(libraries))
    return loaded > 0


def load_onnx_ocr(allow_download: bool = True, prefer_gpu: bool = True) -> "OnnxPPOCR | None":
    """Build the neural OCR engine, or return None if anything is unavailable."""
    try:
        import onnxruntime as ort
    except Exception as exc:                        # noqa: BLE001
        log.info("onnxruntime not installed (%s)", exc)
        return None

    weights = ensure_model(REC_MODEL, allow_download=allow_download)
    dictionary = ensure_model(CHAR_DICT, allow_download=allow_download)
    if weights is None or dictionary is None:
        return None

    charset = dictionary.read_text(encoding="utf-8").splitlines()
    charset.append(" ")

    available = set(ort.get_available_providers())
    use_cuda = prefer_gpu and "CUDAExecutionProvider" in available
    if use_cuda:
        _preload_cuda_libraries()
    providers = (["CUDAExecutionProvider"] if use_cuda else []) + ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(weights), providers=providers)
    except Exception as exc:                        # noqa: BLE001
        log.warning("could not start the ONNX session (%s)", exc)
        return None

    log.info("neural OCR ready on %s", session.get_providers()[0])
    return OnnxPPOCR(session, charset)
