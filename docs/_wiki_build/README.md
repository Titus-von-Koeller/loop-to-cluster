# Wiki build area

Source artifacts for the Notion pages at
<https://app.notion.com/p/226810b52c0d4880b4d0fc8fa4f89012>. Editing rules and the standard for
the pages themselves are in `docs/CONVENTIONS.md`; this directory holds only what can be
*verified* or *regenerated*.

| File | What it is |
| --- | --- |
| `verify_params.py` | Derives the parameter count analytically and checks it against `sum(p.numel())`. |
| `verify_facts.py` | Ground-truths the optimizer, initialization and norm claims, and diffs the preset field by field against SmolLM2-135M's released config. |
| `figstyle.py` | House style for figures. |
| `make_figures.py` | Generates every figure. |
| `figures/` | Rendered PNGs, uploaded by hand. |

Three rules earned the hard way.

**Never source a number from this repo's own code.** A config field that changes no shape is
invisible to a parameter count, which is how `initializer_range` reached the pages as 0.02 when
the released config says 0.041666.... `verify_facts.py` exists to catch exactly that.

**Open the figures; do not read their captions.** Two shipped with text collisions that no
caption could reveal, and three judgements made from captions alone turned out backwards.

**A figure cannot be replaced through the API.** Notion serves image blocks through short-lived
presigned URLs that rotate between read and write, so swapping one means dragging the file onto
the block in the interface.

The palette is three hues — blue `#2166AC`, gold `#DDAA33`, rose `#BB5566` — with worst-adjacent
colour-vision-deficiency deltaE 21.3. A fourth hue drops that to 4.5-9.3, because any teal or
green collides with the rose on the red-green axis. Past three categories, encode with lightness
and direct labels instead.
