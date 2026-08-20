# Figures for the Notion pages

Source artifacts for the Notion pages at
<https://app.notion.com/p/226810b52c0d4880b4d0fc8fa4f89012>. This directory holds only what can be
*verified* or *regenerated*.

| File | What it is |
| --- | --- |
| `verify_params.py` | Derives the parameter count analytically and checks it against `sum(p.numel())`. |
| `verify_facts.py` | Ground-truths the optimizer, initialization and norm claims, and diffs the preset field by field against SmolLM2-135M's released config. |
| `figstyle.py` | House style for figures. |
| `make_figures.py` | Generates every figure. |
|  `out/` | Rendered PNGs, uploaded by hand. |

**Never source a number from this repo's own code.** A config field that changes no shape is
invisible to a parameter count, so a preset can disagree with the released config while every
shape-based check passes. `verify_facts.py` diffs the two.

**Open the figures; do not read their captions.** A caption cannot show a text collision, an
overlapping label, or an axis that renders off the plot.

**A figure cannot be replaced through the API.** Notion serves image blocks through short-lived
presigned URLs that rotate between read and write, so swapping one means dragging the file onto
the block in the interface.

The palette is three hues — blue `#2166AC`, gold `#DDAA33`, rose `#BB5566` — with worst-adjacent
color-vision-deficiency deltaE 21.3. A fourth hue drops that to 4.5-9.3, because any teal or
green collides with the rose on the red-green axis. Past three categories, encode with lightness
and direct labels instead.
