---
name: theme-design
description: Judgment for the theme program — measuring Titus's color vision and choosing or evolving editor and graphing themes from measurements, not taste. Use when touching _palette.py, _viz.py, theme-gallery.py, calibrate-vision.py, editor theme overrides in dotfiles, or any color/contrast decision. Findings live with the instruments; program state lives in CLAUDE.md's queue; this file carries the method.
---

# Theme design, measured

The program (Titus's framing): determine independently the best *editor* theme (best-in-class
as the field, then self-evolved) and the best *graphing* theme, characterize how the two
interact, and choose the best combination — every step from measurement.

## The instruments, and where their knowledge lives

- `notebooks/pytorch-basics/theme-gallery.py` — the exhibit color system and the field's
  palettes under three instruments (as designed, Machado deuteranopia, grayscale), the editor
  theme measured on its own grounds, and the lineage of the rules (Bertin through Munzner).
- `notebooks/pytorch-basics/calibrate-vision.py` — the observer model: a Weibull psychometric
  over weighted LMS-opponent distance, exact grid posterior, information-gain-generated
  stimuli. **Current findings and how to read them are in its closing prose**, next to the
  live numbers; do not restate them elsewhere.
- `~/dotfiles/home/editors/vscode/settings.jsonc` — the applied override layer; its block
  comments are the precedent for method and bar (workbench ~6:1 by day, AA by night).
- `notebooks/pytorch-basics/calibration-responses.jsonl` — every response, append-only;
  size_px and gap_px ride along because they are stimulus parameters.

## Method rules, each earned by a measurement that contradicted a guess

- Never state a contrast without compositing alpha onto the actual page first — Horizon's
  night comments are 30% alpha; the un-composited probe called them fine, the exhibit did not.
- Evolve, don't repaint: keep the theme's hue and saturation, walk lightness to the bar; drop
  alpha where it launders contrast away.
- Report per-axis **thresholds** (tau/sqrt(w)); raw axis weights chase the grid ceiling
  because opponent-axis units are arbitrary.
- Patch size and separation are stimulus parameters: fixed pixels, logged per response,
  near-abutting patches for the sensitive and ecologically honest comparison. Grounds run in
  blocks, never per-trial alternation — adaptation is part of the measurement.
- Exhibit scale does not transfer to glyph scale: color discrimination collapses for small
  fields, and editor tokens are ~10px. Editor-theme decisions wait for text-sized stimuli.
- The background is a variable to search, not only a condition to control (queued: threshold
  as a smooth function of ground luminance and warmth).
- An information-optimal 4AFC trial sits near threshold: to the observer, most trials should
  feel nearly indistinguishable, and "I'm mostly guessing" is the instrument working. The
  occasional easy trial is an anchor (5%) — with lapse pinned by a long log, easy trials carry
  almost no information, so keep the anchor share minimal.
- Greedy one-step EIG needs a dense candidate set to deliver: a coarse magnitude grid (~2.8x
  steps) lost ~28% of achievable information per trial when the threshold fell between steps;
  a two-stage coarse-then-fine sweep per direction recovers it.
- A surface's beauty is allowed to vote and never to overrule the instruments; Titus's eyes
  outrank both — his comparison across a gallery row is the final measurement.
- Verify a theme change by **pixel-sampling a screenshot against the expected hexes**, never by
  impression: an eyeballed screenshot once confirmed a completely dormant override layer as
  "applied" — the reader saw what they expected. The VSCode application gotchas that made it
  dormant (autoDetectColorScheme makes the preferred* theme keys operative; bracket-pair
  colorization is its own layer above textMate rules; notebook.cellEditorBackground does not
  inherit editor.background) live as comments in dotfiles settings.jsonc — read them before
  editing the override layer.

## Debugging an applied theme (earned 2026-09-02, when the whole layer was dormant)

1. **Pixel-sample first**: CDP screenshot of the real surface → PIL crop → Counter of hexes.
   The measured hex tells you *which layer is rendering* — theme default, override, or a third
   party — where an eyeball only confirms expectations.
2. **If overrides are dormant, check the active theme name**: with autoDetectColorScheme on,
   `workbench.preferredLightColorTheme` / `preferredDarkColorTheme` pick the theme and
   `workbench.colorTheme` is inert; a `"[Theme Name]"` block whose name doesn't match the
   *active* variant exactly applies to nothing, silently.
3. **Know the layers**: textMate token rules do not reach bracket-pair colorization (own
   `editorBracketHighlight.*` keys) or semantic tokens; each surface has its own background
   key with its own default chain (`editor.background`, `notebook.editorBackground`,
   `notebook.cellEditorBackground` — which does NOT inherit from editor — `terminal.background`;
   the chat webview follows panel chrome, not the editor).
4. **Apply and verify per surface**: `nh home switch .` lands the symlink in seconds and VSCode
   picks it up live, no reload; then re-sample every surface kind touched — plain editor,
   native notebook, terminal, chat — because each can dissent independently.

## What "pretty" means here — the aesthetics the program applies

Measured legibility is the floor, not the goal; these four theories shape choices above it,
each with its operational form:

- **Processing fluency** (Reber, Schwarz, Winkielman): what is easy to encode feels good.
  Operationally: the fewest simultaneous signals that still carry the information — hue count
  per line down; structure (punctuation, operators, brackets, indent guides, line boxes) at
  near-ink so identifiers, literals, and data marks are the figure. The editor's quiet-structure
  layer and the exhibits' one-base-one-accent rule are the same principle at two scales.
- **Berlyne's inverted U**: pleasure peaks at intermediate complexity. Mute toward calm, never
  toward flat — one expressive accent family stays alive (Horizon's warm corals) so the page
  keeps its character. If a quieting pass makes a surface feel dead, it overshot the ridge.
- **Ecological valence** (Palmer, Schloss): color preference is accumulated personal
  association, so the optimal accents are colors Titus names as loved — an input only he can
  give (asked 2026-09-02, pending). Until then the theme's own hues stand in.
- **Peak shift** (Ramachandran): mild exaggeration of a signature reads as more beautiful than
  the original. Licensed only on rare surfaces (links, errors, selection) and never on body
  tokens; wants glyph-scale data first.

Two operational corollaries, both applied and liked:

- **Luminance hierarchy**: chrome > page > code well in light themes, direction inverted (page
  > well, i.e. wells deepest) in dark ones. One ground step (~3 L*) is a surface's only role
  marker — a single figure-ground cue instead of borders, and dense mono text takes the lowest
  luminance. Every text surface joins the hierarchy: a webview page reading chrome keys
  (sideBar/panel) will otherwise show a second paper on the same screen.
- **Content over commentary**: comments sit a deliberate contrast step below body tokens
  (context, not figure), with the italic carrying the rest of the distinction — but never below
  4.5:1 on the deepest surface they appear on.

Beauty votes through these; the instruments still veto, and Titus's eyes outrank both.
- Results live with their instrument or in Titus's Notion (his hands only); CLAUDE.md carries
  rules, routing, and resume points.

## Standing verdicts (dated, superseded by newer measurements in the instruments)

- 2026-09-02, 602 trials: the exhibit-scale (104px) stage is **converged** — 68% CIs on all
  six per-axis thresholds within ±5%, further clicking at this scale is low-yield. Numbers and
  reading guidance live in calibrate-vision.py's closing prose. The next informative data is
  glyph-scale (queued, decides the editor theme) and the ground search.
- 2026-09-02, 440 trials: every candidate palette's worst pair is lapse-limited-visible at
  104px on both grounds — exhibit-scale palette choice is free of CVD constraints for Titus.
  Sequential house ramp is cividis (applied system-wide with re-measured ink crossovers);
  Okabe-Ito categorical and blue-orange POLARITY stay. Night ground reads ~20% finer than day.
  Horizon stays the editor theme for now, with the measured token/workbench override layer
  applied in dotfiles; the switch-vs-evolve decision waits on glyph-scale data.
