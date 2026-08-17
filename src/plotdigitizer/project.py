"""Saving and reloading a digitizing session.

A session is worth keeping: the calibration and any hand corrections represent real
effort, and a figure often needs a second pass after someone looks at the numbers.

Only the pixel points are stored. Data values are always recomputed from the pixels
through the saved calibration on load, so a file can never hold coordinates that
disagree with the axes that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .detect.extract import ExtractionMode, ExtractionSettings
from .detect.frame import PlotFrame
from .pipeline import DigitizationResult, Series

__all__ = ["save_project", "load_project", "PROJECT_SUFFIX"]

PROJECT_SUFFIX = ".pdproj"
_FORMAT_VERSION = 2


def _settings_to_dict(settings: ExtractionSettings) -> dict:
    payload = dict(vars(settings))
    payload["mode"] = settings.mode.value
    return payload


def _settings_from_dict(payload: dict) -> ExtractionSettings:
    known = {f for f in vars(ExtractionSettings()).keys()}
    filtered = {k: v for k, v in payload.items() if k in known}
    filtered["mode"] = ExtractionMode(payload.get("mode", "scatter"))
    return ExtractionSettings(**filtered)


def _sources_to_dict(sources):
    """Serialise combine provenance, recursively for stacked combines."""
    if not sources:
        return None
    out = []
    for source in sources:
        points = np.asarray(source.get("pixel_points", []), dtype=float)
        out.append({
            "name": source.get("name", "Series"),
            "color": [int(v) for v in source.get("color", (31, 119, 180))],
            "visible": bool(source.get("visible", True)),
            "settings": _settings_to_dict(source["settings"]) if source.get("settings")
                        else None,
            "pixel_points": [[float(x), float(y)] for x, y in points.reshape(-1, 2)],
            "sources": _sources_to_dict(source.get("sources")),
        })
    return out


def _sources_from_dict(stored):
    """Rebuild combine provenance in the shape :mod:`plotdigitizer.compose` expects."""
    if not stored:
        return None
    out = []
    for entry in stored:
        points = np.asarray(entry.get("pixel_points", []), dtype=float)
        out.append({
            "name": entry.get("name", "Series"),
            "color": tuple(entry.get("color", (31, 119, 180))),
            "visible": bool(entry.get("visible", True)),
            "settings": _settings_from_dict(entry.get("settings") or {}),
            "pixel_points": points.reshape(-1, 2),
            "sources": _sources_from_dict(entry.get("sources")),
        })
    return out


def save_project(path: str | Path, result: DigitizationResult,
                 image_path: str | Path | None = None) -> Path:
    """Write a session to a ``.pdproj`` JSON file."""
    path = Path(path)
    payload = {
        "format": _FORMAT_VERSION,
        "image": str(image_path) if image_path else None,
        "image_shape": list(result.image_shape),
        "frame": result.frame.to_dict(),
        "calibration": result.calibration.to_dict(),
        "warnings": result.warnings,
        "confidence": result.confidence,
        "series": [
            {
                "name": series.name,
                "color": list(series.color),
                "visible": series.visible,
                "settings": _settings_to_dict(series.settings),
                # Pixel coordinates only - values are derived on load.
                "pixel_points": [[float(x), float(y)] for x, y in series.pixel_points],
                # Provenance of a combined series, so a split survives save and reload.
                "sources": _sources_to_dict(series.sources),
            }
            for series in result.series
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_project(path: str | Path) -> tuple[DigitizationResult, str | None]:
    """Read a session back. Returns the result and the image path it referenced."""
    path = Path(path)
    payload = json.loads(path.read_text())
    version = payload.get("format", 1)
    if version > _FORMAT_VERSION:
        raise ValueError(
            f"{path.name} was written by a newer version of plotdigitizer "
            f"(format {version}, this build understands {_FORMAT_VERSION})"
        )

    calibration = Calibration.from_dict(payload["calibration"])
    frame = PlotFrame.from_dict(payload["frame"])

    series: list[Series] = []
    for entry in payload.get("series", []):
        points = np.asarray(entry.get("pixel_points", []), dtype=float)
        if points.size == 0:
            points = np.empty((0, 2), dtype=float)
        item = Series(
            name=entry.get("name", "Series"),
            color=tuple(entry.get("color", (31, 119, 180))),
            settings=_settings_from_dict(entry.get("settings", {})),
            pixel_points=points,
            data_points=np.empty((0, 2), dtype=float),
            visible=bool(entry.get("visible", True)),
            sources=_sources_from_dict(entry.get("sources")),
        )
        item.recompute_data(calibration)
        series.append(item)

    shape = payload.get("image_shape") or [0, 0]
    result = DigitizationResult(
        image_shape=(int(shape[0]), int(shape[1])),
        frame=frame,
        calibration=calibration,
        series=series,
        warnings=list(payload.get("warnings", [])),
        confidence=dict(payload.get("confidence", {})),
    )
    return result, payload.get("image")
