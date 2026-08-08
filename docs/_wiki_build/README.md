# Wiki build area

Source artifacts for the Notion wiki at
<https://app.notion.com/p/226810b52c0d4880b4d0fc8fa4f89012>

**Notion is the source of truth.** Nothing here mirrors page prose, and nothing here should
start to: a second copy only drifts, and a stale one gets published over the live wiki by
someone who mistakes it for the source. This directory holds what can be *verified* or
*regenerated* — the scripts that check the numbers the wiki quotes, and the figures it
embeds.

## Before you edit

Rules earned the hard way. Each one cost a real mistake.

- **Titus edits this wiki too.** Never restore, move, or delete anything you did not add
  yourself. A page that differs from what you expected has been *edited*, not damaged — say
  "this changed" and ask, rather than "this broke" and revert. Phrasing you think is clumsy
  may be a deliberate cut.
- **Who writes what.** Prose, structure, figures and corrections are Claude's. Retrieval
  questions and *Interrogate this section* blocks are Titus's — producing those is the
  exercise, so leave the gaps flagged rather than filling them.
- **Open the figures; do not read their captions.** The PNGs are in `figures/`. Two shipped
  with text collisions that no caption could reveal, and three judgements made from captions
  alone turned out backwards.
- **The wiki documents SmolLM2-135M, not whatever `build_model` happens to return.** A field
  that changes no shape is invisible to a parameter count, which is how `initializer_range`
  reached the wiki as 0.02 while the released config says 0.041666… `verify_facts.py` now
  diffs the preset against the released config for exactly this reason. Never source a wiki
  number from the lab's own code.
- **Every number is derived from the config, measured, or absent.** Do not import a
  coefficient from a paper written against a different implementation — activation memory in
  particular has no constant that survives a change of attention kernel.

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
