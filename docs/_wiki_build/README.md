# Wiki build area

Source artifacts for the Notion wiki at
https://app.notion.com/p/226810b52c0d4880b4d0fc8fa4f89012

| File | What it is |
| --- | --- |
| `verify_params.py` | Derives the parameter count analytically and checks it against `sum(p.numel())`. |
| `verify_facts.py` | Ground-truths the optimizer, init and norm claims the wiki makes. |
| `figstyle.py` | House style. Palette chosen by CVD validation, not by eye. |
| `make_figures.py` | Generates every figure. |
| `figures/` | Rendered PNGs, uploaded to Notion. |
| `pages/` | Page sources. Notion is the source of truth after publication. |

The palette is three hues (blue `#2166AC`, gold `#DDAA33`, rose `#BB5566`), worst-adjacent
CVD deltaE 21.3. Adding a fourth hue drops that to 4.5-9.3 because any teal or green
collides with the rose on the red-green axis. Past three categories, encode with lightness
and direct labels instead.
