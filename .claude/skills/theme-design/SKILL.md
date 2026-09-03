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
- `notebooks/pytorch-basics/_observer.py` — **the one observer model** (v2: CAM16-UCS
  geometry, fitted slope/lapse, free confusion-axis orientation, threshold smooth in
  ground lightness, small-field exponent), fit from the shared jsonl and cached beside it.
  Every instrument reads this fit — measurement sharpens preference constraints without a
  second copy anywhere. Change the model here, nowhere else.
- `notebooks/pytorch-basics/calibrate-vision.py` — the discrimination instrument on that
  model: EIG-generated odd-one-out trials over seven grounds (Horizon, Selenized, Modus,
  GitHub dark) and three patch sizes (104/16/10 px — the glyph-scale stage and the
  ground-threshold search run in the same loop). **Current findings and how to read them
  are in its closing prose**, next to the live numbers; do not restate them elsewhere.
- `~/dotfiles/home/editors/vscode/settings.jsonc` — the applied override layer; its block
  comments are the precedent for method and bar (workbench ~6:1 by day, AA by night).
- `~/.claude/skills/titus-preferences/SKILL.md` — his standing functionality and aesthetics
  preferences across all programs; theme choices must respect it, and new preferences he
  states go there.
- `notebooks/pytorch-basics/calibrate-aesthetics.py` — the preference side of the
  interlock: preferential Bayesian optimization over a CAM16-UCS theme space (duels,
  comprehension micro-tasks, find-highlight hunts), with the vision fit's thresholds as
  hard constraints refit live from the shared jsonl. Findings in its closing prose; data
  in `aesthetics-responses.jsonl`.
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
  fields, and editor tokens are ~10px. Editor-theme decisions wait for text-sized stimuli —
  now measured in calibrate-vision's 10/16 px blocks; the fitted small-field exponent is the
  number that decides evolve-vs-switch.
- The background is a variable to search, not only a condition to control: threshold is
  modeled as a smooth function of ground lightness (2 params, not per-ground axes), so
  every measured ground sharpens predictions for grounds never shown. Warmth joins the
  model only when the ground family decouples it from lightness.
- A model swap is validated by re-deriving the old verdicts: v2 (CAM16-UCS, free confusion
  axis) reproduced v1's night-advantage and lapse from the same data before its new claims
  were trusted — and corrected v1's assumed psychometric slope (fitted beta ~ 1.2, not 2),
  which recalibrated every threshold number downstream.
- An information-optimal 4AFC trial sits near threshold: to the observer, most trials should
  feel nearly indistinguishable, and "I'm mostly guessing" is the instrument working. The
  occasional easy trial is an anchor (5%) — with lapse pinned by a long log, easy trials carry
  almost no information, so keep the anchor share minimal.
- Greedy one-step EIG needs a dense candidate set to deliver: a coarse magnitude grid (~2.8x
  steps) lost ~28% of achievable information per trial when the threshold fell between steps;
  a two-stage coarse-then-fine sweep per direction recovers it.
- APCA is the stricter master on dark grounds: WCAG 4.5:1 on the Horizon night page is only
  Lc ~54, under the Lc 60 body bar — a floor checked in WCAG units alone silently under-
  delivers by night (measured 2026-09-03: 131 of 256 sampled dark themes passed 4.5:1 and
  failed Lc 60). Check both, and solve to whichever bar is farther.
- Distinguishability floors can falsify a theme's own role split: Horizon's day string
  (#F6661E) and number (#F77D26) sit ~3 dE apart in CAM16-UCS — inside 2× the measured day
  threshold — so string and number are one literal family in anything built on these
  measurements, and any per-role color plan is checked pairwise against the thresholds
  before it is searched or shipped.
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

- **Elevation, not inset** (Titus's correction of the first attempt, which sank code into
  wells): code sits on the SAME paper everywhere — plain editor, notebook cell, terminal —
  and the notebook page drops one step (~3 L*) below it, so content cells read as raised
  cards behind a quiet border. Consistency of the code ground is itself the fluency cue.
  Collapsed cells render on the page tone (no VSCode key reaches the folded strip), which
  completes the metaphor: folding flattens the card into the page. VSCode notebooks have no
  shadow key; border + ground step is the supported depth cue. Every text surface joins the
  same system: a webview page reads chrome keys (sideBar/panel) and will otherwise show a
  second paper on the same screen.
- **Content over commentary**: comments sit a deliberate contrast step below body tokens
  (context, not figure), with the italic carrying the rest of the distinction — but never below
  4.5:1 on the deepest surface they appear on.
- **Reading typography**: running prose is a CENTERED reading column — one ~70-character
  measure (40em at 17px serif) shared by body and headings so the column keeps a single
  internal left edge, leading 1.6 body / 1.3 headings, kerning pinned; code cells and
  in-markdown tables run full ensemble width as deliberate full-bleed breakouts. Prose and
  code share a central axis, not an edge (Titus dropped the shared-edge constraint: moving
  to code is a context switch anyway; symmetric margins read calmer than a one-sided
  desert). Prose sits on the page, only machine artifacts are cards — and **elevation
  tracks open state, never selection** (Titus, confirmed): every expanded cell carries the
  identical shadow, every collapsed one flattens into the page. Under fold hygiene the
  raised card happens to follow the reader's focus — emergent, not the rule.
- **Shadows in a flat design**: y-offset only (light from top-center), two layers (tight
  contact + wide ambient), alphas low, tinted with the page's darkened hue on warm paper —
  never gray-black. **Negative spread on both layers** (`0 5px 14px -6px`): side chrome
  (gutters, focus indicators) repaints over anything drawn beneath it during scroll, so a
  shadow that reaches sideways flickers there. And radius without overflow-clipping rounds
  each element's OWN paint only — every background-painting child (monaco's `.margin`
  gutter, `.monaco-editor-background`) needs the radius itself or its corners bleed.
- **Embedded monaco follows the host palette by variable capture**: capture the host
  theme's `--vscode-*` values on body before monaco's own theme shadows the same names on
  its container, then pin `.monaco-editor` to the captured copies. The Claude Code diff
  pane works this way (dotfiles patch `claude_code_diff_theme.py`), so palette changes in
  settings.jsonc flow through with no re-patching.
- **Iterate live, bake the winner**: candidate styles go into the running workbench through
  the CDP CSS domain (and adoptedStyleSheets for shadow DOM) for instant screenshots; only
  the converged values are baked into dotfiles (workbench-fonts.css, notebook-prose.py,
  settings.jsonc). Mechanics live in the vscode-keyhole driving notes and cdplab.py.

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
- 2026-09-03, 748 trials, observer v2: the 104-px verdicts survive re-derivation in
  CAM16-UCS (numbers in calibrate-vision's closing prose); **no color-vision deficiency
  signal** — confusion-axis orientation unconstrained, red–green threshold 1.5× blue–yellow
  where anomalous trichromacy shows several-fold. Constraints for the aesthetics search now
  come from this fit (day ΔE ~3.2, night ~2.5 at 104 px, 2× margin pairwise). Glyph-scale
  and ground-family data collection is live; those verdicts wait on his clicks.
