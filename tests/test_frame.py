"""Frame detection, checked against the recorded axes bounding box."""

from __future__ import annotations

import numpy as np

from plotdigitizer.detect.frame import analyse_ink, detect_frame

# matplotlib's recorded axes bbox sits on the spine centres; a couple of pixels of
# slack covers spine thickness and the sub-pixel centre of mass.
TOLERANCE_PX = 3.0


def test_frame_matches_ground_truth(figure):
    frame = detect_frame(figure.image)
    truth = figure.axes_box
    assert abs(frame.left - truth["left"]) <= TOLERANCE_PX, f"left off in {figure.name}"
    assert abs(frame.right - truth["right"]) <= TOLERANCE_PX, f"right off in {figure.name}"
    assert abs(frame.top - truth["top"]) <= TOLERANCE_PX, f"top off in {figure.name}"
    assert abs(frame.bottom - truth["bottom"]) <= TOLERANCE_PX, f"bottom off in {figure.name}"


def test_frame_confidence_is_reported(figure):
    frame = detect_frame(figure.image)
    assert 0.0 <= frame.confidence <= 1.0
    assert frame.confidence > 0.4, f"{figure.name} produced a low-confidence frame"


def test_background_detected_on_dark_figures(corpus):
    dark = analyse_ink(corpus["dark_style"].image)
    light = analyse_ink(corpus["linear_scatter"].image)
    assert dark.background.mean() < 80, "dark style background should be dark"
    assert light.background.mean() > 200, "default style background should be light"


def test_interior_excludes_the_spines(corpus):
    fig = corpus["grid_box"]
    frame = detect_frame(fig.image)
    r0, r1, c0, c1 = frame.interior_bounds(inset=2)
    assert r0 > frame.top
    assert r1 <= frame.bottom + 1
    assert c0 > frame.left
    assert c1 <= frame.right + 1


def test_blank_image_does_not_crash():
    blank = np.full((100, 120, 3), 255, dtype=np.uint8)
    frame = detect_frame(blank)
    assert frame.confidence == 0.0
    assert frame.width > 0 and frame.height > 0
