"""Combining series into one, and splitting them back apart.

Two traces of the same curve, or a curve digitized in pieces, are more useful as one
series - but deciding they belong together is a judgement that often turns out wrong.
So a combine records what it consumed. The result carries a full description of each
original inside ``Series.sources``, and splitting rebuilds them exactly: same points,
same names, same colours, same extraction settings.

That is what makes combining feel like stacking layers rather than flattening them. A
permanent combine is the same operation with the provenance simply not recorded - still
undoable, but no longer splittable once the undo history has moved on.
"""

from __future__ import annotations

import copy

import numpy as np

from .detect.extract import ExtractionMode, ExtractionSettings
from .pipeline import Series

__all__ = ["combine_series", "split_series", "series_to_source", "source_to_series"]


def series_to_source(series: Series) -> dict:
    """Describe a series completely enough to rebuild it later."""
    return {
        "name": series.name,
        "color": tuple(int(v) for v in series.color),
        "visible": bool(series.visible),
        "settings": series.settings.copy(),
        "pixel_points": np.asarray(series.pixel_points, dtype=float).copy(),
        # A source may itself have been a combine; keeping its provenance lets a stack
        # of combines be unwound one layer at a time instead of collapsing at the first.
        "sources": copy.deepcopy(series.sources),
    }


def source_to_series(source: dict, calibration=None) -> Series:
    """Rebuild a series from a stored source description."""
    settings = source.get("settings")
    if not isinstance(settings, ExtractionSettings):
        settings = ExtractionSettings(mode=ExtractionMode.CURVE)

    points = np.asarray(source.get("pixel_points", []), dtype=float)
    if points.size == 0:
        points = np.empty((0, 2), dtype=float)

    series = Series(
        name=source.get("name", "Series"),
        color=tuple(source.get("color", (31, 119, 180))),
        settings=settings.copy(),
        pixel_points=points.copy(),
        data_points=np.empty((0, 2), dtype=float),
        visible=bool(source.get("visible", True)),
        sources=copy.deepcopy(source.get("sources")),
    )
    series.recompute_data(calibration)
    return series


def combine_series(members: list[Series], name: str | None = None,
                   color: tuple[int, int, int] | None = None,
                   keep_sources: bool = True, calibration=None) -> Series:
    """Merge several series into one.

    Points are concatenated and ordered by x, which is what makes the result behave as
    a single curve for extraction, plotting and export. With ``keep_sources`` the
    originals are recorded inside the result and :func:`split_series` can restore them.

    The combined series deliberately carries no ``mask``: it no longer corresponds to
    one region of ink, so re-extracting it from a mask would be meaningless. The
    sources keep their own masks, which come back with them on a split.
    """
    if not members:
        raise ValueError("combine_series needs at least one series")

    stacked = [np.asarray(s.pixel_points, dtype=float).reshape(-1, 2) for s in members]
    points = np.vstack(stacked) if stacked else np.empty((0, 2), dtype=float)
    if points.shape[0]:
        points = points[np.argsort(points[:, 0], kind="stable")]

    combined = Series(
        name=name or " + ".join(s.name for s in members),
        color=tuple(color) if color else tuple(members[0].color),
        settings=members[0].settings.copy(),
        pixel_points=points,
        data_points=np.empty((0, 2), dtype=float),
        visible=True,
        sources=[series_to_source(s) for s in members] if keep_sources else None,
    )
    combined.recompute_data(calibration)
    return combined


def split_series(series: Series, calibration=None) -> list[Series]:
    """Restore the series a combine consumed.

    Returns an empty list when there is nothing recorded - a permanent combine, or a
    series that was never combined - so the caller can tell "cannot split" from
    "split into nothing".
    """
    if not series.sources:
        return []
    return [source_to_series(source, calibration) for source in series.sources]


def describe_sources(series: Series) -> str:
    """Human-readable summary of what a split would produce."""
    if not series.sources:
        return ""
    names = [str(source.get("name", "Series")) for source in series.sources]
    if len(names) <= 3:
        return ", ".join(names)
    return f"{', '.join(names[:3])} and {len(names) - 3} more"
