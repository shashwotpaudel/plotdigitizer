---
title: The interface
---

# The interface

The window is a canvas showing your figure, a stack of four panels on the right, and a
magnifier that follows the cursor.

## Toolbar

| Button | What it does |
|---|---|
| **Open image** | Load a figure. <kbd>Ctrl</kbd>+<kbd>O</kbd> |
| **Recent** | Files opened before |
| **Auto-digitize** | Run detection again from scratch. <kbd>Ctrl</kbd>+<kbd>D</kbd> |
| **Undo / Redo** | <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> |
| **Fit / 100%** | Zoom to the window, or to actual pixels |
| **Snap** | Pull clicked points onto the centre of the nearest stroke |
| **Lasso** | Whether a selection drag draws a box or a freehand shape |
| **Trace stroke** | Click a curve to follow it into its own series. <kbd>Ctrl</kbd>+<kbd>T</kbd> |
| **Points / Handles / Frame** | Show or hide each overlay |
| **Export CSV** | <kbd>Ctrl</kbd>+<kbd>E</kbd> |

## Mouse and keyboard

On the canvas:

| Action | Result |
|---|---|
| Left-click | Add a point to the active series |
| Left-drag | Lay a trail of points along the drag |
| <kbd>Alt</kbd> or <kbd>Ctrl</kbd> + left-drag | **Sweep** along a curve — see [workflows](workflows.md#sweeping-along-a-curve) |
| Drag a point | Move it |
| Right-drag | Select points — box or freehand |
| <kbd>Shift</kbd> + right-drag | Add to the selection |
| <kbd>Ctrl</kbd> + right-drag | Remove from the selection |
| Right-click a point | Select or deselect just that one |
| <kbd>Esc</kbd> | Clear the selection |
| <kbd>Delete</kbd> | Delete the selection, or the current point |
| Arrow keys | Nudge the selected point one pixel |
| <kbd>Shift</kbd> + arrows | Nudge a quarter of a pixel |
| <kbd>Ctrl</kbd> + arrows | Nudge ten pixels |
| Scroll wheel | Zoom around the cursor |
| Middle-drag, or <kbd>Space</kbd>+drag | Pan |
| <kbd>Ctrl</kbd>+<kbd>V</kbd> | Load an image from the clipboard |

Nothing on the canvas destroys data as a side effect of a drag. Selecting is always
separate from acting on the selection.

## The magnifier

The inset at the bottom right shows the pixels under the cursor at 8×, with a crosshair
on the exact position that will be recorded. Use it when placing a point by hand: the
canvas can stay at a comfortable zoom while you still see individual pixels.

## Panel 1 — Calibrate

Four reference points define the mapping from pixels to numbers: two on the x axis
(`x1`, `x2`) and two on the y (`y1`, `y2`). Each has a value you can type and a handle
you can drag.

**Type** sets the axis scale — Linear, Log10, Loge, Reciprocal or Date. Detection picks
this by testing which scale best predicts where the ticks actually are, so a log axis is
normally identified without help. It needs three or more readable ticks to choose
anything other than linear; two points sit on every scale equally well, so with only two
it stays linear rather than guessing.

The badge at the top reports confidence:

- **green** — read from the tick labels and fits them closely
- **amber** — fitted, but loosely; check the values against the figure
- **red** — could not be read; type all four values yourself

**Handle positions (px)** shows where the handles sit in the image. **Re-detect axes**
runs the whole detection again, and asks first if you have made manual corrections.

## Panel 2 — Series

Lists every curve found, with its colour, name and point count. Click one to make it
active. Double-click to rename it. Untick it to hide it and leave it out of the export.

**Extraction** re-reads the active series from the image:

| Control | Effect |
|---|---|
| **Mode** | *Scatter* takes one point per marker; *Line* traces the stroke |
| **X step** | Sampling interval along a line, in pixels |
| **Smoothing** | Median filter width for a traced line |
| **Bridge gaps up to** | How wide a gap to cross before treating it as a real break |
| **Min blob size** | Ignore marker blobs smaller than this fraction of a typical one |
| **Limit points** | Thin the result down to a fixed count |

**Combine…** merges several series into one. **Split** puts a combined series back into
the ones it came from — see [combining segments](workflows.md#combining-segments).

**Find stray points** highlights points that look like they belong to a different curve,
with a sensitivity slider. It only ever selects; nothing changes until you act on the
selection. It needs at least 15 points to say anything, and declines on sparse scatters
rather than guessing.

## Panel 3 — Data

The table that will be exported, and a preview chart of it.

The chart is the fastest sanity check in the application. A mis-chosen axis scale, an
inverted axis or a curve that jumped onto its neighbour all show up as an obviously
wrong shape long before they would be noticeable in a column of numbers.

Editing a cell moves the point on the figure to match, so the table and the overlay can
never disagree.

## Panel 4 — Export

Choose the CSV layout, delimiter, precision and whether to include a header row or the
pixel coordinates. The preview shows the actual text that will be written.

**Copy** puts the same text on the clipboard. **Save session** / **Open session** store
and reload your work as a `.pdproj` file. **Import CSV as series** loads numbers back
onto the figure, which is useful for comparing against a reference extraction.

## Selection bar

When points are selected, a bar appears under the figure reading, for example,
`35 of 847 points selected`. It is only present when there is a selection, so its
buttons cannot be pressed by accident.

| Button | Effect |
|---|---|
| **Move to** | Hand the points to another series, or to a new one |
| **Delete** | Remove them |
| **Keep only** | Remove everything *except* them |
| **Invert** | Select what is not currently selected |
| **Clear** | Deselect |

**Move to** is usually the right answer when a trace strayed: those points belong to a
different curve, not in the bin.

## Status bar

Shows the cursor position in both pixels and data units, the zoom level, and which
compute device is in use. Messages about what just happened appear here, including
warnings when a trace found another stroke running alongside it.
