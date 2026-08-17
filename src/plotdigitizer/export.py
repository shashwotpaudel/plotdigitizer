"""Writing extracted data out.

CSV is the deliverable, so the layouts here cover the three shapes people actually
need: columns side by side for a spreadsheet, tidy rows for a dataframe, and one file
per series for downstream scripts.

Numbers are formatted with :func:`repr`-grade precision rather than rounded for
display. Rounding here would silently discard accuracy the pipeline worked to earn.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

__all__ = ["ExportLayout", "csv_string", "write_csv", "write_json", "ExportOptions"]


class ExportLayout(str, Enum):
    """How multiple series are arranged in one CSV."""

    COMBINED = "combined"    # x1,y1,x2,y2,... side by side, padded to the longest
    LONG = "long"            # series,x,y - one row per point
    SEPARATE = "separate"    # one file per series

    @property
    def label(self) -> str:
        return {
            ExportLayout.COMBINED: "Combined columns (x, y per series)",
            ExportLayout.LONG: "Tidy rows (series, x, y)",
            ExportLayout.SEPARATE: "One file per series",
        }[self]


@dataclass
class ExportOptions:
    layout: ExportLayout = ExportLayout.COMBINED
    delimiter: str = ","
    include_header: bool = True
    #: Significant digits; None keeps full precision.
    precision: int | None = None
    visible_only: bool = True
    #: Also write the pixel coordinates each value came from.
    include_pixels: bool = False


def _format(value: float, precision: int | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if precision is None:
        return repr(float(value))
    return f"{float(value):.{precision}g}"


def _selected(series_list, options: ExportOptions):
    chosen = [s for s in series_list if (s.visible or not options.visible_only)]
    return [s for s in chosen if s.data_points.shape[0] > 0]


def csv_string(series_list, options: ExportOptions | None = None) -> str:
    """Render series to CSV text using the combined or long layout."""
    options = options or ExportOptions()
    chosen = _selected(series_list, options)
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=options.delimiter, lineterminator="\n")

    if not chosen:
        if options.include_header:
            writer.writerow(["x", "y"])
        return buffer.getvalue()

    pixels = options.include_pixels

    if options.layout is ExportLayout.LONG:
        if options.include_header:
            writer.writerow(["series", "x", "y"] + (["x_px", "y_px"] if pixels else []))
        for series in chosen:
            for index, (x, y) in enumerate(series.data_points):
                row = [series.name, _format(x, options.precision),
                       _format(y, options.precision)]
                if pixels:
                    px, py = series.pixel_points[index]
                    row += [_format(px, options.precision), _format(py, options.precision)]
                writer.writerow(row)
        return buffer.getvalue()

    # Combined: series side by side, shorter ones padded with blanks.
    if options.include_header:
        header: list[str] = []
        for series in chosen:
            header.extend([f"{series.name} x", f"{series.name} y"])
            if pixels:
                header.extend([f"{series.name} x_px", f"{series.name} y_px"])
        writer.writerow(header)

    longest = max(s.data_points.shape[0] for s in chosen)
    width = 4 if pixels else 2
    for row in range(longest):
        cells: list[str] = []
        for series in chosen:
            if row < series.data_points.shape[0]:
                x, y = series.data_points[row]
                cells.extend([_format(x, options.precision), _format(y, options.precision)])
                if pixels:
                    px, py = series.pixel_points[row]
                    cells.extend([_format(px, options.precision),
                                  _format(py, options.precision)])
            else:
                cells.extend([""] * width)
        writer.writerow(cells)
    return buffer.getvalue()


def read_csv_series(path: str | Path, calibration, name: str | None = None,
                    delimiter: str | None = None):
    """Read a two-column CSV back in as a series, for comparison against an extraction.

    The values are mapped through ``calibration`` into pixel positions, because pixels
    are what a series is stored in - that way an imported reference lands on the figure
    where those numbers actually are, and any disagreement with the calibration shows up
    as a visible offset rather than hiding in a column of numbers.
    """
    from .detect.extract import ExtractionMode, ExtractionSettings
    from .pipeline import Series

    path = Path(path)
    text = path.read_text()
    if delimiter is None:
        delimiter = "\t" if "\t" in text.splitlines()[0] else \
            ";" if ";" in text.splitlines()[0] else ","

    values: list[tuple[float, float]] = []
    for row in csv.reader(text.splitlines(), delimiter=delimiter):
        if len(row) < 2:
            continue
        try:
            values.append((float(row[0]), float(row[1])))
        except ValueError:
            continue                      # header or blank line
    if not values:
        raise ValueError(f"{path.name}: no numeric rows found")

    data = np.asarray(values, dtype=float)
    pixels = calibration.to_pixel(data)
    series = Series(
        name=name or path.stem,
        color=(233, 30, 99),
        settings=ExtractionSettings(mode=ExtractionMode.CURVE),
        pixel_points=pixels,
        data_points=data,
    )
    return series


def _safe_name(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_ ") else "_" for c in name]
    return "".join(keep).strip().replace(" ", "_") or "series"


def write_csv(path: str | Path, series_list, options: ExportOptions | None = None) -> list[Path]:
    """Write series to disk. Returns every file written."""
    options = options or ExportOptions()
    path = Path(path)

    if options.layout is not ExportLayout.SEPARATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(csv_string(series_list, options))
        return [path]

    chosen = _selected(series_list, options)
    written: list[Path] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, series in enumerate(chosen, start=1):
        target = path.with_name(f"{path.stem}_{index}_{_safe_name(series.name)}{path.suffix or '.csv'}")
        single = ExportOptions(layout=ExportLayout.COMBINED, delimiter=options.delimiter,
                               include_header=options.include_header,
                               precision=options.precision, visible_only=False)
        target.write_text(csv_string([series], single))
        written.append(target)
    return written


def write_json(path: str | Path, result, options: ExportOptions | None = None) -> Path:
    """Write the full digitization - calibration included - as JSON."""
    options = options or ExportOptions()
    path = Path(path)
    payload = {
        "calibration": result.calibration.to_dict(),
        "frame": result.frame.to_dict(),
        "confidence": result.confidence,
        "warnings": result.warnings,
        "series": [
            {
                "name": series.name,
                "color": series.hex_color,
                "mode": series.settings.mode.value,
                "points": [[float(x), float(y)] for x, y in series.data_points],
            }
            for series in _selected(result.series, options)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path
