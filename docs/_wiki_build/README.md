# Wiki build area

Source artifacts for the Notion wiki at
<https://app.notion.com/p/226810b52c0d4880b4d0fc8fa4f89012>

**Notion is the source of truth.** Nothing here mirrors page prose, and nothing here should
start to: a second copy only drifts, and a stale one gets published over the live wiki by
someone who mistakes it for the source. This directory holds what can be *verified* or
*regenerated* — the scripts that check the numbers the wiki quotes, and the figures it
embeds.

| File | What it is |
| --- | --- |
| `verify_params.py` | Derives the parameter count analytically and checks it against `sum(p.numel())`. |
| `verify_facts.py` | Ground-truths the optimizer, init and norm claims the wiki makes, and diffs the preset field by field against SmolLM2-135M's released config. |
| `figstyle.py` | House style. Palette chosen by CVD validation, not by eye. |
| `make_figures.py` | Generates every figure. |
| `figures/` | Rendered PNGs, uploaded to Notion by hand. Notion serves image blocks through short-lived presigned URLs that rotate between read and write, so an existing figure cannot be swapped through the API — replacing one means dragging the file onto the block in the UI. |

The palette is three hues (blue `#2166AC`, gold `#DDAA33`, rose `#BB5566`), worst-adjacent
CVD deltaE 21.3. Adding a fourth hue drops that to 4.5-9.3 because any teal or green
collides with the rose on the red-green axis. Past three categories, encode with lightness
and direct labels instead.
