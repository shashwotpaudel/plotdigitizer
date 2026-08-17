"""Headless command line: image in, CSV out.

Runs exactly the same pipeline the GUI does, for batch work over a directory of
figures. Because nobody reviews the overlay in this mode, it prints what it inferred -
axis limits, scales, series found - and returns a non-zero exit status when the
calibration could not be read, so a script never silently writes out garbage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .backend import describe_devices
from .export import ExportLayout, ExportOptions, write_csv, write_json
from .image_io import SUPPORTED_SUFFIXES, load_image
from .pipeline import AutoDigitizer, DigitizationResult

log = logging.getLogger("plotdigitizer")


def _describe(result: DigitizationResult) -> str:
    x, y = result.calibration.x, result.calibration.y
    lines = [
        f"  frame      : x {result.frame.left:.1f}..{result.frame.right:.1f}, "
        f"y {result.frame.top:.1f}..{result.frame.bottom:.1f} "
        f"(confidence {result.confidence.get('frame', 0):.2f})",
        f"  x axis     : {x.to_data(result.frame.left):.6g} .. "
        f"{x.to_data(result.frame.right):.6g}  [{x.scale.value}] "
        f"(confidence {result.confidence.get('x_axis', 0):.2f})",
        f"  y axis     : {y.to_data(result.frame.bottom):.6g} .. "
        f"{y.to_data(result.frame.top):.6g}  [{y.scale.value}] "
        f"(confidence {result.confidence.get('y_axis', 0):.2f})",
    ]
    for series in result.series:
        first = series.data_points[0] if series.count else np.array([np.nan, np.nan])
        last = series.data_points[-1] if series.count else np.array([np.nan, np.nan])
        lines.append(
            f"  {series.name:10s}: {series.count:4d} points  {series.hex_color}  "
            f"{series.settings.mode.value:7s}  "
            f"x {first[0]:.4g}..{last[0]:.4g}"
        )
    return "\n".join(lines)


def _inputs(paths: list[str]) -> list[Path]:
    collected: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            collected.extend(sorted(
                p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES))
        else:
            collected.append(path)
    return collected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plotdigitizer",
        description="Automatically digitize 2D plots: detect the axes, read the limits, "
                    "extract the data, write a CSV.",
    )
    # Optional, because --devices reports on the machine and --gui can open an empty
    # window; both are useless if they insist on an image first.
    parser.add_argument("images", nargs="*", default=[],
                        help="image file(s) or a directory of images")
    parser.add_argument("-o", "--output",
                        help="output CSV path (default: alongside each image). With "
                             "several inputs this is treated as a directory.")
    parser.add_argument("--layout", default=ExportLayout.COMBINED.value,
                        choices=[m.value for m in ExportLayout],
                        help="how multiple series share the file (default: combined)")
    parser.add_argument("--json", action="store_true",
                        help="also write a .json holding the calibration and points")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="compute device (default: auto - GPU when one is usable)")
    parser.add_argument("--max-series", type=int, default=8,
                        help="upper bound on how many series to separate (default: 8)")
    parser.add_argument("--ocr", default="template", choices=["template", "neural"],
                        help="label reader: 'template' needs no download and is the "
                             "default; 'neural' downloads a PP-OCR model once, for "
                             "scanned or unusual figures")
    parser.add_argument("--colour-tolerance", type=float, default=28.0,
                        help="how far a pixel may sit from a series colour, in Lab units")
    parser.add_argument("--precision", type=int, default=None,
                        help="significant digits in the CSV (default: full precision)")
    parser.add_argument("--gui", action="store_true",
                        help="open the image in the desktop app instead of exporting")
    parser.add_argument("-q", "--quiet", action="store_true", help="only report problems")
    parser.add_argument("--devices", action="store_true",
                        help="report the available compute devices and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")

    if args.devices:
        info = describe_devices()
        print(f"selected: {info.name}")
        print(f"torch installed: {'yes' if info.torch_available else 'no'}")
        return 0

    if args.gui:
        from .ui.app import main as gui_main
        return gui_main([args.images[0]] if args.images else [])

    if not args.images:
        parser.error("no images given")

    images = _inputs(args.images)
    if not images:
        parser.error("no image files found")

    digitizer = AutoDigitizer(device=args.device, max_series=args.max_series,
                              colour_tolerance=args.colour_tolerance,
                              ocr_engine=args.ocr)
    if not args.quiet:
        print(f"device: {digitizer.backend.describe()}")

    options = ExportOptions(layout=ExportLayout(args.layout), precision=args.precision)
    failures = 0

    for image_path in images:
        if not image_path.exists():
            log.error("%s: not found", image_path)
            failures += 1
            continue
        try:
            image = load_image(image_path)
        except Exception as exc:
            log.error("%s: could not read image (%s)", image_path.name, exc)
            failures += 1
            continue

        result = digitizer.run(image)

        if args.output and len(images) == 1 and not Path(args.output).is_dir():
            target = Path(args.output)
        else:
            directory = Path(args.output) if args.output else image_path.parent
            target = directory / f"{image_path.stem}.csv"

        written = write_csv(target, result.series, options)
        if args.json:
            write_json(target.with_suffix(".json"), result, options)

        if not args.quiet:
            print(f"\n{image_path.name}  ({result.elapsed_seconds * 1000:.0f} ms on {result.device})")
            print(_describe(result))
            print(f"  -> {', '.join(str(p) for p in written)}")

        for warning in result.warnings:
            log.warning("%s: %s", image_path.name, warning)

        if not result.calibrated:
            log.error("%s: axes could not be calibrated automatically - the CSV holds "
                      "uncalibrated values. Open it in the GUI (--gui) to set the limits.",
                      image_path.name)
            failures += 1
        elif not result.series:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
