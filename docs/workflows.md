---
title: Workflows
---

# Workflows

The automatic pass handles ordinary coloured figures well. These are the cases that need
something more.

## Black-and-white figures

Curves are separated by colour. A journal figure printed in black ink has only one
colour, so every curve on it belongs to the same cluster and the automatic pass returns
them merged into a single series. No threshold can fix that — the information simply is
not in the colour.

What *does* distinguish the curves is that each one is continuous. Two tools use that.

### Trace stroke

Press **Trace stroke** (<kbd>Ctrl</kbd>+<kbd>T</kbd>) and click once on a curve. It
follows that curve column by column, choosing at each step the ink that best continues
the direction it has been travelling, and coasting across gaps so dotted and dashed
lines survive. The result becomes its own series.

This is fully automatic and usually recovers most of a curve from one click. It works
because a curve *goes somewhere* while its neighbours merely pass nearby.

Delete the merged series first, then trace each curve in turn.

### Sweeping along a curve

Hold <kbd>Alt</kbd> (or <kbd>Ctrl</kbd>, since some desktops use <kbd>Alt</kbd>+drag to
move windows) and drag roughly along a curve. Every image column your drag crosses is
read off the stroke beneath your path.

The difference from **Trace stroke** matters: your path tells it *which* curve you mean.
Where several curves run close together or cross, no analysis can decide which one was
intended, but you can — and dragging along the right one is how you say so.

Two properties worth knowing:

- The reading is one point per column, not one per mouse event, so a quick drag and a
  slow drag over the same curve give identical data.
- You do not need to be accurate. A path wobbling several pixels either side of the
  curve still lands on the stroke; only ink within a narrow corridor of your path is
  considered, which is what keeps it off the neighbouring curve.

Each sweep becomes its own series named `Segment 1`, `Segment 2` and so on.

## Combining segments

Tracing a curve in pieces is often easier than getting it in one go — especially where
it passes behind a legend or through a crowded region. Each piece is a separate series,
so a bad piece can be deleted on its own without redoing the rest.

When the pieces are right, select **Combine…** in the Series panel, tick the ones that
belong together, name the result, and confirm. The default keeps the originals inside
the combined series: the list marks it `[3 merged]` and **Split** restores them exactly
as they were — same points, names, colours and settings. Choose *Combine permanently*
only if you are sure.

A combined series survives saving and reloading a session with its split still
available.

## Overlapping curves

Where two curves genuinely run on top of each other, there is nothing in the image to
say which points belong to which. Any tool that claims otherwise is guessing.

What happens in practice: a trace follows whichever curve it is already on and continues
through the overlap, so its output through that stretch may belong to either curve. The
status bar reports how many columns had a second stroke running alongside, but a quiet
report is **not** a promise the trace stayed put — where curves fuse into a single
stroke there is no second candidate to notice.

The practical approach:

1. Trace or sweep each curve through the region where they are clearly separate.
2. Select the points in the overlapping stretch — a lasso, or **Select spikes** if the
   trace jumped visibly.
3. Delete them, or **Move to** the series they belong to if you can tell.
4. Treat what remains in that region as uncertain, and say so when you report it.

## Cleaning up a trace

A trace that wandered leaves two signatures, and different tools find each.

**Isolated spikes** — single points thrown off the curve. **Select spikes** finds these.

**A displaced run** — a whole stretch that stepped onto another curve and back. These
are invisible to a simple deviation test, because once enough displaced points fill the
comparison window they *become* the local trend. Spike selection looks for the pair of
jumps instead: an anomalous step away matched with a later step back marks everything
between them.

Either way, review the highlighted points before acting. Selection changes nothing on
its own.

## Batch processing

For many figures of a similar style, the [command line](cli.md) runs the same pipeline
without a window:

```bash
plotdigitizer figures/ -o out/ --device auto
```

It writes one CSV per figure and prints what it inferred for each — axis limits, scale
types and series found. It exits non-zero if any figure could not be calibrated, so a
script will notice rather than quietly collect nonsense.

Batch mode suits figures that are consistent and simple. Nobody is reviewing the
overlay, so anything unusual is better opened in the window.

## Comparing against a reference

**Import CSV as series** loads existing numbers back onto the figure, mapped through the
current calibration. The imported series is drawn over the image, so a disagreement
between your extraction and the reference shows up as a visible offset rather than
hiding in a column of numbers.
