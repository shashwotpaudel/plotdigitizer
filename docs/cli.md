---
title: Command line
---

# Command line

The `plotdigitizer` command runs exactly the same detection as the window, without
opening one. It suits batches of similar figures.

```bash
plotdigitizer figure.png -o out.csv
plotdigitizer figures/ -o out/ --device auto
```

Because nobody is reviewing the overlay, it prints what it inferred for every figure —
frame, axis limits, scale types, confidences and the series it found — and returns a
non-zero exit status rather than quietly writing numbers it could not calibrate.

## Options

| Option | Meaning |
|---|---|
| `images` | One or more image files, or a directory of them |
| `-o`, `--output` | Output path. With several inputs this is treated as a directory. Defaults to writing alongside each image |
| `--layout {combined,long,separate}` | How multiple series share the file. Default `combined` |
| `--json` | Also write a `.json` holding the calibration and the points |
| `--device {auto,cpu,cuda}` | Compute device. Default `auto`, which uses a GPU when one is usable |
| `--max-series N` | Upper bound on how many series to return, counting those split from one colour. A colour whose split would go over it stays one series. Default 8 |
| `--ocr {template,neural}` | Label reader. Default `template`, which needs no download |
| `--colour-tolerance F` | How far a pixel may sit from a series colour, in Lab units. Default 28 |
| `--no-separate` | Keep each colour as one series, as v0 did, instead of splitting series that share a colour by line style, marker shape and continuity |
| `--precision N` | Significant digits in the CSV. Default is full precision |
| `--gui` | Open the image in the window instead of exporting |
| `-q`, `--quiet` | Only report problems |
| `--devices` | Report the available compute devices and exit |

## Layouts

`combined` puts each series side by side as `Series 1 x, Series 1 y, Series 2 x, …`,
padding shorter series with blanks. `long` writes tidy rows of `series, x, y`.
`separate` writes one file per series, named after the output path.

## Exit status

`0` when every figure was calibrated and produced at least one series. `1` if any figure
was missing or unreadable, could not be calibrated automatically, or yielded no series.

In a script, that means you can trust a zero exit and inspect anything else:

```bash
if plotdigitizer figures/ -o out/ -q; then
    echo "all figures digitized"
else
    echo "some figures need checking by hand" >&2
fi
```

An uncalibrated figure still writes its CSV, but the values are in pixel-derived units
rather than the figure's, so the non-zero status is the signal that matters. Open those
in the window with `--gui` and set the axes by hand.

## What it prints

```
device: GPU (NVIDIA GeForce RTX 3090)

multi_scatter_legend.png  (446 ms on cuda)
  frame      : x 80.0..576.0, y 58.0..427.0 (confidence 1.00)
  x axis     : 0.00288846 .. 9.99711  [linear] (confidence 0.98)
  y axis     : 0.0160078 .. 69.9366  [linear] (confidence 0.97)
  Series 1  :   14 points  #2da02b  scatter  x 1.01..8.99
  Series 2  :   14 points  #1c78b4  scatter  x 1.01..8.99
  Series 3  :   14 points  #fe8113  scatter  x 1.01..8.99
  -> out/multi_scatter_legend.csv
```

Series that share a colour are split, and each says how it is drawn:

```
microphone.png  (602 ms on cpu)
  frame      : x 100.0..720.0, y 53.0..392.0 (confidence 1.00)
  x axis     : 9.98333 .. 5021.07  [log10] (confidence 0.91)
  y axis     : -0.0720999 .. 89.9128  [linear] (confidence 0.97)
  Series 1  :  528 points  #030303  curve    x 24.88..4921  (solid line)
  Series 2  :  146 points  #030303  curve    x 24.88..688.7  (dotted line)
  Series 3  :  208 points  #030303  scatter  x 16..4588  (hollow triangle markers)
  -> out/microphone.csv
```

With `--json`, each series in the JSON carries the same description as `style`.

Anything that needs checking is logged after the figure's summary, and is still shown
with `-q`. When a split leaves some of a colour's ink in no series, the warning says
where:

```
short_curve.png  (360 ms on cpu)
  frame      : x 80.0..576.0, y 58.0..427.0 (confidence 1.00)
  x axis     : 0.00288846 .. 9.99711  [linear] (confidence 0.98)
  y axis     : -0.00643292 .. 9.9974  [linear] (confidence 0.97)
  Series 1  :  493 points  #020202  curve    x 0.04319..9.957  (solid line)
  Series 2  :  493 points  #020202  curve    x 0.04319..9.957  (solid line)
  -> out/short_curve.csv
WARNING: short_curve.png: Some #020202 ink between x = 0.06334 and 1.272 is in none of the series found in it; check that stretch against the figure
```

That figure has three curves. The third leaves through the top of the plot at x = 1.3,
which is too short to be kept as a series.

The confidences are worth reading. Anything below about 0.5 on an axis means the fit was
loose and the figure deserves a look in the window.

## Neural label reading

`--ocr neural` downloads a text-recognition model (about 11 MB, cached in
`~/.cache/plot_digitizer/models/`) and uses it instead of the built-in reader. It is for
scans, photographs and unusual typefaces. On ordinary rendered figures the default is
more accurate — it knows about raised exponents, which a general recogniser flattens —
so there is no reason to reach for it unless the default is struggling.

If the model cannot be downloaded, the tool logs it and carries on with the default
reader rather than failing.
