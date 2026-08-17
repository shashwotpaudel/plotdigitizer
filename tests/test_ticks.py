"""Tick detection, checked against the recorded tick pixel positions."""

from __future__ import annotations

import numpy as np
import pytest

from plotdigitizer.detect.frame import analyse_ink, detect_frame
from plotdigitizer.detect.ticks import detect_ticks

MATCH_TOLERANCE_PX = 2.0


def _detect(figure, axis):
    ink = analyse_ink(figure.image)
    frame = detect_frame(figure.image, ink)
    return detect_ticks(ink, frame, axis)


def _truth_positions(figure, axis):
    ticks = figure.xticks if axis == "x" else figure.yticks
    return np.array([t["pixel"] for t in ticks], dtype=float)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_every_true_tick_is_found(figure, axis):
    """Each major tick matplotlib drew must have a detection near it."""
    found = _detect(figure, axis).positions
    truth = _truth_positions(figure, axis)
    assert found.size >= 2, f"{figure.name}/{axis}: only {found.size} ticks found"

    missed = [t for t in truth if np.min(np.abs(found - t)) > MATCH_TOLERANCE_PX]
    assert not missed, (
        f"{figure.name}/{axis}: missed ticks at {np.round(missed, 1).tolist()}; "
        f"found {np.round(np.sort(found), 1).tolist()}"
    )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_no_spurious_ticks(figure, axis):
    """Nothing may be reported as a major tick that matplotlib did not draw as one."""
    found = _detect(figure, axis).positions
    truth = _truth_positions(figure, axis)
    spurious = [f for f in found if np.min(np.abs(truth - f)) > MATCH_TOLERANCE_PX]
    assert not spurious, (
        f"{figure.name}/{axis}: extra ticks at {np.round(spurious, 1).tolist()}"
    )


def test_inward_ticks_use_the_inner_band(corpus):
    tickset = _detect(corpus["inward_ticks"], "x")
    assert tickset.side == "inside"
    assert tickset.count >= 5


def test_outward_ticks_use_the_outer_band(corpus):
    tickset = _detect(corpus["linear_scatter"], "x")
    assert tickset.side == "outside"


def test_log_axis_minor_ticks_are_separated(corpus):
    """On the semilog figure the majors are the decades; minors must not join them."""
    tickset = _detect(corpus["log_y_line"], "y")
    assert tickset.count == len(corpus["log_y_line"].yticks)
    assert len(tickset.minor) > 0, "log axis should expose minor ticks too"
    gaps = np.diff(np.sort(tickset.positions))
    assert gaps.std() / gaps.mean() < 0.05, "major decades should be evenly spaced"


@pytest.mark.parametrize("axis", ["x", "y"])
def test_regularity_reported(figure, axis):
    assert 0.0 <= _detect(figure, axis).regularity <= 1.0
