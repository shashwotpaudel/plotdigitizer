"""Backend behaviour, and CPU/CUDA parity where a GPU is present."""

from __future__ import annotations

import numpy as np
import pytest

from plotdigitizer.backend import (
    NumpyBackend,
    describe_devices,
    select_backend,
)


def _cuda_backend():
    backend = select_backend("cuda")
    if backend.kind != "cuda":
        pytest.skip("no CUDA backend available")
    return backend


class TestSelection:
    def test_cpu_is_always_available(self):
        assert select_backend("cpu").kind == "cpu"

    def test_auto_never_raises(self):
        assert select_backend("auto").kind in ("cpu", "cuda")

    def test_cuda_falls_back_instead_of_failing(self):
        """Asking for CUDA on a machine without it must degrade, not crash."""
        assert select_backend("cuda").kind in ("cpu", "cuda")

    def test_describe_devices(self):
        info = describe_devices()
        assert info.kind in ("cpu", "cuda")
        assert info.name


class TestNearestCenter:
    def test_assigns_obvious_clusters(self):
        backend = NumpyBackend()
        centers = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.float32)
        points = np.array([[3, 2, 1], [250, 4, 4], [10, 240, 6]], dtype=np.float32)
        labels, dist = backend.nearest_center(points, centers)
        assert list(labels) == [0, 1, 2]
        assert np.all(dist < 20)

    def test_chunking_matches_single_pass(self):
        rng = np.random.default_rng(0)
        backend = NumpyBackend()
        points = rng.random((9000, 3), dtype=np.float32) * 255
        centers = rng.random((6, 3), dtype=np.float32) * 255
        labels, dist = backend.nearest_center(points, centers)
        # Reference: direct full matrix.
        d = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assert np.array_equal(labels, np.argmin(d, axis=1))
        assert np.allclose(dist, np.sqrt(d.min(axis=1)), atol=1e-3)


class TestKMeans:
    def test_recovers_separated_clusters(self):
        rng = np.random.default_rng(1)
        truth = np.array([[20, 20, 200], [200, 30, 30], [30, 190, 40]], dtype=np.float32)
        points = np.repeat(truth, 400, axis=0) + rng.normal(0, 3, (1200, 3)).astype(np.float32)
        centers = NumpyBackend().kmeans(points, 3, seed=0)
        # Each true colour must have a centre close to it.
        for want in truth:
            assert np.min(np.linalg.norm(centers - want, axis=1)) < 8.0

    def test_k_larger_than_points(self):
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        centers = NumpyBackend().kmeans(points, 5, seed=0)
        assert centers.shape[0] == 2


class TestTemplateMatch:
    def test_peak_at_true_location(self):
        image = np.zeros((64, 80), dtype=np.float32)
        image[30:36, 50:56] = 1.0
        template = np.zeros((6, 6), dtype=np.float32)
        template[:, :] = 1.0
        # Give the template some structure so NCC is well defined.
        template[0, 0] = 0.0
        image[30, 50] = 0.0
        response = NumpyBackend().match_template_ncc(image, template)
        peak = np.unravel_index(np.argmax(response), response.shape)
        assert peak == (30, 50)

    def test_oversized_template_returns_empty(self):
        out = NumpyBackend().match_template_ncc(np.zeros((4, 4), np.float32),
                                                np.zeros((9, 9), np.float32))
        assert out.size == 0


class TestCudaParity:
    """The GPU path must agree with the NumPy reference, not merely run."""

    def test_nearest_center_parity(self):
        gpu = _cuda_backend()
        rng = np.random.default_rng(2)
        points = (rng.random((50_000, 3)) * 255).astype(np.float32)
        centers = (rng.random((8, 3)) * 255).astype(np.float32)
        l_cpu, d_cpu = NumpyBackend().nearest_center(points, centers)
        l_gpu, d_gpu = gpu.nearest_center(points, centers)
        assert np.array_equal(l_cpu, l_gpu)
        assert np.allclose(d_cpu, d_gpu, atol=1e-2)

    def test_kmeans_parity(self):
        gpu = _cuda_backend()
        rng = np.random.default_rng(3)
        truth = np.array([[20, 20, 200], [200, 30, 30], [30, 190, 40]], dtype=np.float32)
        points = np.repeat(truth, 500, axis=0) + rng.normal(0, 4, (1500, 3)).astype(np.float32)
        c_cpu = np.sort(NumpyBackend().kmeans(points, 3, seed=7), axis=0)
        c_gpu = np.sort(gpu.kmeans(points, 3, seed=7), axis=0)
        assert np.allclose(c_cpu, c_gpu, atol=1.0)

    def test_template_match_parity(self):
        gpu = _cuda_backend()
        rng = np.random.default_rng(4)
        image = rng.random((128, 160)).astype(np.float32)
        template = image[40:52, 60:72].copy()
        r_cpu = NumpyBackend().match_template_ncc(image, template)
        r_gpu = gpu.match_template_ncc(image, template)
        assert r_cpu.shape == r_gpu.shape
        assert np.unravel_index(np.argmax(r_cpu), r_cpu.shape) == (40, 60)
        assert np.unravel_index(np.argmax(r_gpu), r_gpu.shape) == (40, 60)
        assert np.allclose(r_cpu, r_gpu, atol=2e-3)
