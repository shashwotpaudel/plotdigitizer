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

### Curves drawn in the same colour

Series are separated by colour first. Where several share a colour — a figure printed
in black ink is the usual case — they are told apart the way a reader would:

- by **marker shape**: circles, squares, triangles, diamonds, crosses, plus signs, stars;
- by **line style**: solid, dashed, dotted, dash-dot;
- by **continuity**, where several lines share a style: each one is followed through
  the places where it crosses another.

A marker drawn on a line goes to the line it sits on. Markers on no line are a scatter
series of their own. Lettering inside the plot — annotations, the words of a legend — is
recognised and left out. Error bars are recognised and set aside, so the marker under
them is still known by its shape; the series gets one point per marker, at its centre,
and the ends of the bars are not read. A closed curve, or one that turns back on itself
like a hysteresis loop, stays one series.

What it cannot do:

- Two scatter series drawn with the **same marker in the same colour** come back as one.
  Nothing in the image separates them. Size is not used either: the same shape drawn at
  two sizes is one series.
- Two lines in the **same style** that run along each other for a long stretch, or cross
  at a very shallow angle, can come out of it swapped. See the next section.
- A line that spans less than about a sixth of the plot's width is not kept as a series:
  a steep curve that leaves through the top early, or a short tangent drawn at one end.
  Its points are missing from the result.
- **Noisy lines**, such as measured data drawn as a jagged trace, are the weakest case.
  The colour may come back as one series, or the zigzags of the trace may be taken for
  small hollow markers and end up in a marker series.
- A line with markers only a few pixels apart comes back as one series, markers and all.
- **Step curves**, such as survival curves, are separated, but their vertical risers are
  often missing or partly given to the other curve.
- On a very small figure, where a dash is only a few pixels long, dashes can be taken
  for dots. The lines still separate; only the description in the Series list is wrong.

The split also checks its own result. If more than about a third of a colour's ink ends
up in none of the series it found, the split is not trusted and the colour stays one
series. If more than a tenth is left over, the split is kept and you are warned, with
the range of x where the left-over ink lies. In the window the warning appears when
detection finishes; on the command line it is printed with the figure's summary.

If a figure is split where you did not want it to be, click the arrow beside
**Auto-digitize** and untick **Split series that share a colour**, or pass
`--no-separate` on the command line. Each colour then comes back as one series, as in
v0. The window remembers the choice between sessions, and while it is off the status bar
says so after each detection. [Trace stroke and sweeping](workflows.md#black-and-white-figures)
remain for anything the automatic pass gets wrong.

### Overlapping curves cannot be separated

Where two curves genuinely lie on top of each other, nothing in the image says which
points belong to which.

The automatic split does not guess there. Where two lines in one style coincide, both
series report the shared ink. Where a dotted or dashed line runs under a solid one, its
dots and dashes fuse into the solid line and it is left with a gap. Where lines run
through a dense band of markers, the band goes to the markers and the lines are left
with gaps. A gap is often the honest answer rather than a fault.

A trace through such a region follows whichever curve it was already on, and its output
there may belong to either.

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

Where the series share a colour, the legend's lettering is left out with the rest of the
text in the plot, and its sample lines are too short to be taken for series. Its sample
markers can still turn up as extra points in a marker series.

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

### Series that share a colour

Telling apart series drawn in one colour is measured the same way, against black-ink
figures drawn from known data. Two numbers matter for each true series. **Purity** is
the fraction of the detected series' points that lie on that curve and not on another;
**coverage** is the fraction of the curve the series reaches. A point counts as on a
curve within 3 pixels of it.

| corpus | figures | series count right | mean purity | mean coverage |
|---|---|---|---|---|
| `tests/samecolour` | 11 | 11/11 | 0.999 | 0.969 |
| `tests/samecolour_holdout` | 7 | 7/7 | 1.000 | 0.955 |

Purity is the one to watch: a series that has picked up points from a neighbouring
curve looks exactly as convincing as a clean one. Coverage falls short mostly where
curves coincide, which is expected.

As before, the second set was written after the method was fixed. On its first run six
of the seven figures came out right. The seventh, 320×240 with one-pixel lines, came
back with two extra grey series; the fault was in the colour separation and was already
there in v0. It was fixed after that run, and the set has been run after every change
since as a check that nothing got worse, so the held-out numbers above are no longer
free of tuning. First-run figures were 6/7, 0.996 and 0.941.

A third set, `tests/hard`, holds 86 figures built from the failures a review found in
the first version of the split. It was used to fix them, so like `tests/samecolour` it
shows how well the fixes took, not how the method does on figures it has not seen. It
mixes three kinds of figure:

| figures in `tests/hard` | figures | series count right | v0 count right | mean purity | mean coverage |
|---|---|---|---|---|---|
| series share a colour and should be split | 37 | 37/37 | 0/37 | 0.997 | 0.975 |
| colours already tell the series apart, or only one series | 40 | 37/40 | 36/40 | 0.886 | 0.870 |
| known weak cases: noisy, short and step lines, crowded markers, near-identical shades | 9 | 4/9 | 0/9 | 0.732 | 0.618 |

The second group should come back at least as well as it did before the split existed.
The three figures it gets wrong are colour problems v0 also gets wrong: a dashed grid in
a colour of its own, a JPEG legend and speckled colours. One figure in it is worse than
in v0: a pale, noisy line drawn under a smoothed copy of itself in a darker shade, where
the pale line's coverage falls from 0.84 to 0.74. Over all 86 figures the split gets
78/86, mean purity 0.923 and coverage 0.891; v0 gets 36/86, 0.399 and 0.409.

```bash
python tools/separation_report.py
python tools/separation_report.py --data tests/samecolour_holdout
python tools/separation_report.py --data tests/hard
```

On the two original corpora, where every series has a colour of its own, the numbers in
the first table are the same as in v0.

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
them, or delete the whole spurious series. On a figure whose series share a colour, one
line can also come back in two pieces; **Combine** them.

Too few means series share a colour and could not be told apart — the same marker twice,
two lines in one style lying on top of each other, a line too short to count, or a
figure the split did not trust and kept as one series per colour. Check that **Split
series that share a colour** is ticked; the status bar says so when it is not. See
[curves drawn in the same colour](#curves-drawn-in-the-same-colour) and
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
./plotdigitizer-v1.1-x86_64.AppImage --appimage-extract
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
