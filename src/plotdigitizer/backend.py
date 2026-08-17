"""Compute backend selection: NumPy everywhere, CUDA via torch when it is available.

The NumPy implementation is the reference. The torch one exists because three steps in
the pipeline are embarrassingly parallel over pixels and dominate the runtime on large
images: assigning every pixel to its nearest colour cluster, running k-means over
millions of pixels, and normalised cross-correlation for marker template matching.

Everything else in the pipeline is cheap or inherently sequential and stays on the CPU.
A backend is therefore a small object with a handful of hot operations on it, not an
array library abstraction - and the parity test asserts both produce the same answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Backend", "NumpyBackend", "TorchBackend", "select_backend", "describe_devices"]


def _try_import_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception:  # pragma: no cover - depends on the install
        return None


@dataclass
class DeviceInfo:
    kind: str          # "cpu" or "cuda"
    name: str          # human readable, shown in the UI status bar
    torch_available: bool


class Backend:
    """Interface for the pixel-parallel operations the pipeline needs."""

    kind = "cpu"
    name = "CPU"

    # -- colour ------------------------------------------------------------------
    def nearest_center(self, points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each point, the index of and distance to the closest centre.

        ``points`` is (N, C), ``centers`` is (K, C). Returns (labels (N,), distances (N,)).
        """
        raise NotImplementedError

    def kmeans(self, points: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
        """k-means++ seeded Lloyd iterations; returns the (k, C) centres."""
        raise NotImplementedError

    def match_template_ncc(self, image: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Normalised cross-correlation of a single-channel image against a template."""
        raise NotImplementedError

    # -- reporting ---------------------------------------------------------------
    def describe(self) -> str:
        return self.name


class NumpyBackend(Backend):
    kind = "cpu"
    name = "CPU (NumPy)"

    def nearest_center(self, points: np.ndarray, centers: np.ndarray):
        points = np.ascontiguousarray(points, dtype=np.float32)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        best_d = np.full(points.shape[0], np.inf, dtype=np.float32)
        best_i = np.zeros(points.shape[0], dtype=np.int32)
        # Chunked so a large image cannot blow up memory with an (N, K) matrix.
        chunk = max(1, int(4_000_000 / max(1, centers.shape[0])))
        for start in range(0, points.shape[0], chunk):
            block = points[start:start + chunk]
            d = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(d, axis=1)
            best_i[start:start + chunk] = idx
            best_d[start:start + chunk] = d[np.arange(block.shape[0]), idx]
        return best_i, np.sqrt(best_d)

    def kmeans(self, points: np.ndarray, k: int, iters: int = 25, seed: int = 0):
        points = np.ascontiguousarray(points, dtype=np.float32)
        centers = _kmeans_plusplus_init(points, k, seed, self)
        for _ in range(iters):
            labels, _ = self.nearest_center(points, centers)
            new = _recompute_centers(points, labels, k, centers)
            if np.allclose(new, centers, atol=1e-4):
                centers = new
                break
            centers = new
        return centers

    def match_template_ncc(self, image: np.ndarray, template: np.ndarray):
        import cv2
        img = np.ascontiguousarray(image, dtype=np.float32)
        tpl = np.ascontiguousarray(template, dtype=np.float32)
        if tpl.shape[0] > img.shape[0] or tpl.shape[1] > img.shape[1]:
            return np.zeros((0, 0), dtype=np.float32)
        return cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)


class TorchBackend(Backend):
    kind = "cuda"

    def __init__(self, torch, device):
        self._torch = torch
        self._device = device
        try:
            self.name = f"GPU ({torch.cuda.get_device_name(device)})"
        except Exception:
            self.name = f"GPU ({device})"

    def _t(self, arr, dtype=None):
        return self._torch.as_tensor(np.ascontiguousarray(arr), device=self._device,
                                     dtype=dtype or self._torch.float32)

    def nearest_center(self, points: np.ndarray, centers: np.ndarray):
        torch = self._torch
        with torch.no_grad():
            p = self._t(points)
            c = self._t(centers)
            best_i = torch.empty(p.shape[0], dtype=torch.int64, device=self._device)
            best_d = torch.empty(p.shape[0], dtype=torch.float32, device=self._device)
            chunk = max(1, int(16_000_000 / max(1, c.shape[0])))
            for start in range(0, p.shape[0], chunk):
                block = p[start:start + chunk]
                d = torch.cdist(block, c)
                v, i = torch.min(d, dim=1)
                best_i[start:start + chunk] = i
                best_d[start:start + chunk] = v
            return best_i.cpu().numpy().astype(np.int32), best_d.cpu().numpy()

    def kmeans(self, points: np.ndarray, k: int, iters: int = 25, seed: int = 0):
        torch = self._torch
        with torch.no_grad():
            p = self._t(points)
            centers = self._t(_kmeans_plusplus_init(points, k, seed, self))
            for _ in range(iters):
                d = torch.cdist(p, centers)
                labels = torch.argmin(d, dim=1)
                new = centers.clone()
                for j in range(centers.shape[0]):
                    sel = labels == j
                    if bool(sel.any()):
                        new[j] = p[sel].mean(dim=0)
                if bool(torch.allclose(new, centers, atol=1e-4)):
                    centers = new
                    break
                centers = new
            return centers.cpu().numpy()

    def match_template_ncc(self, image: np.ndarray, template: np.ndarray):
        torch = self._torch
        import torch.nn.functional as F

        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            return np.zeros((0, 0), dtype=np.float32)

        with torch.no_grad():
            img = self._t(image)[None, None]
            tpl = self._t(template)[None, None]
            th, tw = tpl.shape[-2:]
            n = float(th * tw)

            tpl_centred = tpl - tpl.mean()
            tpl_norm = torch.sqrt((tpl_centred**2).sum()).clamp_min(1e-12)

            ones = torch.ones_like(tpl)
            local_sum = F.conv2d(img, ones)
            local_sq = F.conv2d(img * img, ones)
            local_mean = local_sum / n
            # Variance via the sum-of-squares identity, clamped against tiny negatives.
            local_var = (local_sq - local_sum * local_mean).clamp_min(0.0)
            local_norm = torch.sqrt(local_var).clamp_min(1e-12)

            corr = F.conv2d(img, tpl_centred)
            out = corr / (local_norm * tpl_norm)
            return out[0, 0].clamp(-1.0, 1.0).cpu().numpy()


def _kmeans_plusplus_init(points: np.ndarray, k: int, seed: int, backend: Backend) -> np.ndarray:
    """k-means++ seeding, done on the CPU - it is sequential and only k passes long."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    k = min(k, n)
    centers = np.empty((k, points.shape[1]), dtype=np.float32)
    centers[0] = points[rng.integers(n)]
    if k == 1:
        return centers
    closest = ((points - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = closest.sum()
        if not np.isfinite(total) or total <= 0:
            centers[j] = points[rng.integers(n)]
        else:
            centers[j] = points[rng.choice(n, p=closest / total)]
        d = ((points - centers[j]) ** 2).sum(axis=1)
        closest = np.minimum(closest, d)
    return centers


def _recompute_centers(points, labels, k, previous):
    out = previous.copy()
    for j in range(k):
        sel = labels == j
        if sel.any():
            out[j] = points[sel].mean(axis=0)
    return out


def describe_devices() -> DeviceInfo:
    """What hardware is actually usable right now, for display and for --device auto."""
    torch = _try_import_torch()
    if torch is None:
        return DeviceInfo("cpu", "CPU (NumPy) - torch not installed", False)
    try:
        if torch.cuda.is_available():
            return DeviceInfo("cuda", f"GPU ({torch.cuda.get_device_name(0)})", True)
    except Exception as exc:  # pragma: no cover
        log.debug("CUDA probe failed: %s", exc)
    return DeviceInfo("cpu", "CPU (NumPy) - no CUDA device", True)


def select_backend(device: str = "auto") -> Backend:
    """Return a backend for ``device`` in {"auto", "cpu", "cuda"}.

    ``auto`` uses CUDA when torch reports a working device and falls back silently.
    An explicit ``cuda`` that cannot be honoured warns and falls back rather than
    failing, because the whole point is that the same code runs on any machine.
    """
    device = (device or "auto").lower()
    if device == "cpu":
        return NumpyBackend()

    torch = _try_import_torch()
    if torch is None:
        if device == "cuda":
            log.warning("CUDA requested but torch is not installed; using CPU. "
                        "Install with: pip install -e '.[torch]'")
        return NumpyBackend()

    try:
        if torch.cuda.is_available():
            backend = TorchBackend(torch, torch.device("cuda:0"))
            # Touch the device once so a broken driver surfaces here, not mid-pipeline.
            backend.nearest_center(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
            return backend
        if device == "cuda":
            log.warning("CUDA requested but torch.cuda.is_available() is False; using CPU.")
    except Exception as exc:
        log.warning("CUDA backend unavailable (%s); using CPU.", exc)
    return NumpyBackend()
