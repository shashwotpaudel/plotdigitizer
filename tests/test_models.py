"""The optional neural OCR engine and its download machinery.

These tests must pass on a machine with no network and no onnxruntime, because that is
a supported way to run this project. Anything that needs the model is skipped rather
than failed when it is not there.
"""

from __future__ import annotations

import pytest

from plotdigitizer.detect.ocr import TemplateOCR, get_default_ocr
from plotdigitizer.models import REC_MODEL, ModelSpec, ensure_model, model_dir


class TestModelCache:
    def test_cache_dir_can_be_overridden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLOTDIGITIZER_MODEL_DIR", str(tmp_path / "models"))
        assert model_dir() == tmp_path / "models"

    def test_cache_dir_follows_xdg(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PLOTDIGITIZER_MODEL_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert model_dir() == tmp_path / "plot_digitizer" / "models"

    def test_missing_model_without_download_returns_none(self, tmp_path, monkeypatch):
        """Offline and not cached is a normal state, not an error."""
        monkeypatch.setenv("PLOTDIGITIZER_MODEL_DIR", str(tmp_path))
        assert ensure_model(REC_MODEL, allow_download=False) is None

    def test_truncated_file_is_rejected(self, tmp_path, monkeypatch):
        """A half-finished download must not be handed to the ONNX runtime."""
        monkeypatch.setenv("PLOTDIGITIZER_MODEL_DIR", str(tmp_path))
        (tmp_path / REC_MODEL.filename).write_bytes(b"not really a model")
        assert ensure_model(REC_MODEL, allow_download=False) is None

    def test_unreachable_url_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLOTDIGITIZER_MODEL_DIR", str(tmp_path))
        spec = ModelSpec(name="nope", url="https://127.0.0.1:9/none.onnx",
                         filename="none.onnx", approx_bytes=1000)
        assert ensure_model(spec, allow_download=True, timeout=2.0) is None
        assert not (tmp_path / "none.onnx.part").exists(), "partial file left behind"


class TestEngineSelection:
    def test_default_engine_needs_nothing_downloaded(self):
        engine = get_default_ocr()
        assert isinstance(engine, TemplateOCR)

    def test_requesting_neural_falls_back_cleanly(self, tmp_path, monkeypatch):
        """With no model available, asking for neural must still return a usable engine."""
        monkeypatch.setenv("PLOTDIGITIZER_MODEL_DIR", str(tmp_path))
        monkeypatch.setattr("plotdigitizer.models.ensure_model", lambda *a, **k: None)
        engine = get_default_ocr(prefer_neural=True)
        assert engine is not None
        assert hasattr(engine, "read_line")


def _neural_engine():
    from plotdigitizer.models import load_onnx_ocr
    engine = load_onnx_ocr(allow_download=False)
    if engine is None:
        pytest.skip("neural OCR model not cached locally")
    return engine


class TestNeuralEngine:
    """Only run where the model has already been fetched."""

    def test_reads_plain_tick_labels(self, corpus):
        from plotdigitizer.detect.frame import analyse_ink, detect_frame
        from plotdigitizer.detect.labels import detect_labels
        from plotdigitizer.detect.ticks import detect_ticks

        engine = _neural_engine()
        figure = corpus["linear_scatter"]
        ink = analyse_ink(figure.image)
        frame = detect_frame(figure.image, ink)
        result = detect_labels(ink, frame, detect_ticks(ink, frame, "y"), "y", engine)

        values = sorted(v for _, v in result.pairs)
        assert values == pytest.approx([0, 10, 20, 30, 40, 50])

    def test_interface_matches_the_template_engine(self):
        engine = _neural_engine()
        text, confidence = engine.read_line([])
        assert text == "" and confidence == 0.0

    def test_runs_on_the_gpu_when_one_is_available(self):
        """The whole point of the ONNX path is that one file serves CPU and GPU."""
        engine = _neural_engine()
        providers = engine._session.get_providers()
        assert "CPUExecutionProvider" in providers
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            assert providers[0] == "CUDAExecutionProvider", (
                "CUDA is available but the session did not use it")
