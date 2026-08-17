---
title: FAQ
---

# FAQ

## Limitations

Read this section before trusting a number.

### What kinds of plot does it read?

2D line and scatter plots. Axes may be linear, log10, natural log, reciprocal or dates.

It does **not** read bar charts, pie or doughnut charts, polar plots, ternary diagrams,
box plots, heatmaps, contour maps or anything three-dimensional. Given one it will still
find a frame and some ink and produce output, but that output will not mean anything.

### Curves drawn in the same colour arrive merged

Series are separated by colour. A figure printed in black ink has one colour, so every
curve in it lands in a single series.

[Trace stroke and sweeping](workflows.md#black-and-white-figures) exist for exactly this
and pull the curves apart by following their continuity instead. It is a manual step,
and on a busy monochrome figure it is the bulk of the work.

### Overlapping curves cannot be separated

Where two curves genuinely lie on top of each other, nothing in the image says which
points belong to which. A trace through such a region follows whichever curve it was
already on, and its output there may belong to either.

This is not a tuning problem — the information is absent. On the figure used to develop
these tools, three curves converge above roughly 700 Hz and every trace, seeded on any
of them, returns the same path through that region.

The status bar reports how many columns had a second stroke running alongside. **A quiet
report is not a promise the trace stayed on one curve**: where two strokes fuse into a
single one there is no second candidate for it to notice. Treat overlapping regions as
uncertain and say so when you report the numbers.

### Legends without a frame

A legend drawn with a box around it is detected and excluded. One drawn without a frame
is not, so its sample lines and markers may appear as stray data points. Lasso and
delete them.

### Date axes are supported but not detected

The calibration handles date axes correctly, but detection will not recognise one on its
own. Set **Type** to *Date* and enter the two values by hand.

### Series names

Series are named `Series 1`, `Series 2` and so on. Legend text is not read, so names
have to be typed if you want meaningful ones. Double-click a series in the list.

### Spike selection needs enough points

**Select spikes** compares each point with its neighbours, which requires enough
neighbours to be meaningful. Below 15 points it declines and says so rather than
selecting something arbitrary on a sparse scatter.

### Non-linear axes need three ticks

Choosing between linear, log and reciprocal requires at least three readable tick
labels. Two points sit on every scale equally well, so with only two the axis stays
linear rather than guessing. Set it by hand if you know better.

## How accurate is it?

Accuracy is measured against figures rendered from known data, so the comparison is with
the numbers that were actually plotted rather than with a second estimate.

| corpus | figures | axis calibration | mean point error | worst |
|---|---|---|---|---|
| `tests/data` | 14 | 14/14 within 1.5 px | 0.136 % | 0.71 % |
| `tests/holdout` | 6 | 6/6 within 1.5 px | 0.138 % | 0.75 % |

Error is a percentage of the plotted axis extent. On a log axis that means a percentage
of the decades shown, which is the only unit that means the same thing at both ends of
the axis. 0.14 % is about a fifth of a pixel on a 500-pixel axis.

Reproduce it from a clone:

```bash
python tools/accuracy_report.py --device auto
python tools/accuracy_report.py --device auto --data tests/holdout
```

The two corpora are separate on purpose. Every threshold in the detection code was
chosen while looking at the first one, so quoting only its numbers would measure how
well those constants were fitted rather than whether the approach generalises. The
second was written afterwards and never used for tuning; it covers a grey-panelled
style with no axis lines, JPEG compression, serif and monospace labels, two-tone markers
and monochrome line art. Four of those six broke the detector the first time they were
run.

## Does it need a GPU?

No. Everything works on the CPU.

With PyTorch installed, `--device auto` uses a GPU for the pixel-parallel parts —
assigning pixels to colours, clustering, template matching. On figures of ordinary size
this saves almost nothing: a 640×480 figure takes about 0.2–0.4 seconds either way. It
earns its place on large scans.

The AppImage does not bundle PyTorch, and there is no GPU variant to download. Bundling
PyTorch and the CUDA runtime produces a 3.1 GB file, and a single release asset may not
exceed 2 GB. It is also the least useful thing to package that way, since the image
cannot supply the one piece that actually has to match your machine — the driver.

For a GPU, install from source with the `torch` extra. If you would rather have the
image anyway, the recipe is in the repository and builds locally:

```bash
./packaging/build-appimage.sh cuda
```

Accuracy is identical either way; only speed differs.

## Troubleshooting

### The axes came out wrong

Check the confidence badge in the Calibrate panel first.

If a tick label was misread, the fit ignores the outlier rather than averaging it in,
and reports that it did — so a single bad label is usually harmless. If several were
misread, the axis will be wrong: type the four values yourself, and drag the handles
onto the ticks they correspond to.

Very small or unusual fonts are the common cause. `--ocr neural` sometimes helps on
scans; on ordinary rendered figures the default reader is better.

### The wrong number of curves was found

Too many usually means something structural was mistaken for data — an unframed legend,
an annotation, or an axis style the frame detector misread. Lasso the strays and delete
them, or delete the whole spurious series.

Too few means curves share a colour. See
[black-and-white figures](workflows.md#black-and-white-figures).

### A curve was traced as a line when it is a scatter

Change **Mode** in the Series panel; it re-extracts at once. The automatic guess uses the
fact that markers are identical to each other while pieces of a line vary with the local
slope, which is right on ordinary figures but cannot be right on all of them.

### The preview chart looks nothing like the figure

Almost always the axis scale. A log axis read as linear produces a shape that is
obviously wrong in the preview and only subtly wrong in the numbers, which is why the
preview is there.

### The AppImage will not start

It needs glibc 2.28 or newer — Debian 10, Ubuntu 18.10, RHEL 8, Fedora 29 and later.
Check yours with `ldd --version`. On an older system, install from source instead;
there is nothing in the tool itself that requires a recent system, only in the Qt build
the image bundles.

If it exits complaining about a Qt platform plugin, the usual cause is a missing X
library. On Debian and Ubuntu:

```bash
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1
```

If it does nothing at all, your system may lack FUSE. Either install it, or extract and
run without it:

```bash
./plotdigitizer-v0-x86_64.AppImage --appimage-extract
./squashfs-root/AppRun
```

### It cannot find my GPU

Run `plotdigitizer --devices`. If it reports CPU while you have an NVIDIA card, PyTorch
is either not installed or not built for CUDA:

```bash
pip install -e ".[gpu,torch]"
```

An outdated driver is the other common cause. The tool falls back to the CPU rather than
failing, so a missing GPU only costs speed.

## Other questions

### Where are my settings kept?

Window layout, recent files and export options in `~/.config/plotdigitizer/`. Autosaved
sessions in `~/.cache/plot_digitizer/autosave/`. The optional neural model in
`~/.cache/plot_digitizer/models/`.

### Can I get the pixel coordinates as well as the values?

Yes — tick **Also write pixel coordinates** in the Export panel, or pass `--json` on the
command line. Useful for checking a result later, or for re-deriving values if the
calibration turns out to have been wrong.

### Does it send anything anywhere?

No. It works entirely offline. The only network access it can make is downloading the
optional neural OCR model, which happens once, only if you ask for `--ocr neural`.

### Can I use it on a headless machine?

The `plotdigitizer` command needs no display. The window does; over SSH use X
forwarding, or run the command line and inspect the CSV.
