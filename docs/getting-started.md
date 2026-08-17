---
title: Getting started
---

# Getting started

## Install

### AppImage

The simplest route on Linux. Download it from the
[releases page](https://github.com/shashwotpaudel/plotdigitizer/releases), make it
executable, and run it — there is nothing to install and nothing to uninstall.

```bash
chmod +x plotdigitizer-v0-x86_64.AppImage
./plotdigitizer-v0-x86_64.AppImage
```

It needs glibc 2.28 or newer — Debian 10, Ubuntu 18.10, RHEL 8, Fedora 29 and later.
On an older system, install from source instead. See the
[FAQ](faq.md#the-appimage-will-not-start) if it does not start.

### From source

```bash
git clone https://github.com/shashwotpaudel/plotdigitizer.git
cd plotdigitizer
python3 -m venv .venv
.venv/bin/pip install -e .
```

That gives you two commands inside the virtual environment: `plotdigitizer-gui` for the
window and `plotdigitizer` for the command line. For GPU acceleration, install
`.venv/bin/pip install -e ".[gpu,torch]"` instead — see
[does it need a GPU](faq.md#does-it-need-a-gpu) for whether that is worth it.

## Your first figure

Open an image:

```bash
plotdigitizer-gui my-figure.png
```

You can also drag an image onto the window, or paste a screenshot straight from the
clipboard with <kbd>Ctrl</kbd>+<kbd>V</kbd>.

Detection runs immediately. Within a second or so you should see:

- a dashed blue rectangle around the plot area
- four orange handles labelled `x1`, `x2`, `y1`, `y2` sitting on the axes
- coloured dots covering the data points it found
- the **1 Calibrate** panel filled in, with a confidence badge

### 1. Check the calibration

The panel on the right shows the values it read from the tick labels and the scale type
it chose for each axis. Look at the four numbers. If they match the figure, the hard
part is already done.

If a value is wrong, type the correct one. If a handle is in the wrong place, drag it —
grab it by the labelled arrow at its end or by the short stub where the line pokes out
past the plot. The line across the middle of the figure is only a guide, so it never
gets in the way of clicking a data point.

A green badge means it is confident. Amber means check carefully. Red means it could not
read the axes and you should enter all four values yourself.

### 2. Check the curves

Open **2 Series**. Each curve it separated is listed with its colour and point count.
Click one to make it active; its points are the ones you can edit.

If a curve was traced as a line when it is really a scatter — or the reverse — change
**Mode** and it re-extracts immediately.

### 3. Check the numbers

Open **3 Data**. This is the table that will be exported, with a small preview chart
below it. The chart is the quickest check available: if the shape does not match the
figure, something is wrong upstream, and it is usually the axis scale.

Editing a number in the table moves the corresponding point on the figure, so the two
can never drift apart.

### 4. Export

Open **4 Export**, choose a layout, and press **Export CSV**. The preview shows exactly
what will be written before you write it.

| Layout | Shape |
|---|---|
| Combined columns | `Series 1 x, Series 1 y, Series 2 x, …` side by side |
| Tidy rows | `series, x, y` — one row per point |
| One file per series | a separate CSV for each |

Values are written at full precision by default. Tick **Also write pixel coordinates**
to record where in the image each number came from.

## Fixing what it got wrong

Nothing about the automatic pass is final.

- **Left-click** the figure to add a point, **drag** one to move it, and
  **right-drag** to select a group.
- Selected points can be deleted, or **moved to another series** — the usual fix when a
  curve picked up its neighbour's points.
- <kbd>Ctrl</kbd>+<kbd>Z</kbd> undoes anything, including the automatic detection itself.

The [interface reference](interface.md) covers every tool. If your figure is black and
white, or its curves overlap, read [workflows](workflows.md) — those cases need a
different approach.

## Saving your work

**Save session** writes a `.pdproj` file holding the calibration and every correction,
so you can come back to a figure later. The tool also autosaves every 30 seconds while
there is unsaved work, and offers to recover it if something goes wrong.
