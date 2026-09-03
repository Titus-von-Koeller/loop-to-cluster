# CLAUDE.md — loop-to-cluster

Titus is learning distributed training, from a single-GPU loop to a sharded multi-GPU one. Marc
Sun (accelerate maintainer) reviews weekly and needs to see how Titus is reasoning, not only what
he concluded — so the notes are his own writing and live in Notion. This repo holds what runs.

He is an experienced engineer new to this domain: what is expensive for him is knowing where
things are and which version does what, not reasoning about them.

## Environment

Name the project directory in the same command. A bare `pixi run` from a parent directory walks
*upward* for a manifest and can bind a different workspace, and ambient direnv activation is no
safer, since an agent's working directory resets between commands.

```bash
cd /home/titus/src/loop-to-cluster && pixi run python ...
```

**Always `CUDA_VISIBLE_DEVICES=0`, when running single GPU workloads.** GPU 1 drives a display, so its clocks move with the
compositor and its memory starts several GiB down — a timing taken there is not a measurement.
Use `direnv exec <abs-dir> <cmd>` when you also need `nvcc` or nix's `libstdc++`. A missing tool
is not a dead end: `nix shell nixpkgs#<pkg> -c <cmd>`.

Resolved versions come from `pixi list <package>` rather than from memory or from a list here
that would go stale. `pixi.toml` records why each dependency comes from PyPI rather than
conda-forge; one reason is the torch ABI and it still holds.

Skill directories are watched live: adds, removals, and `SKILL.md` edits are picked up within
the session. Only plugin-backed pieces of a skill folder — `hooks/`, `.mcp.json`, `agents/` —
need `/reload-plugins`.

marimo finds `[tool.marimo]` with `find_nearest_pyproject_toml`, which walks *upward* from the
working directory, so a kernel started in `notebooks/pytorch-basics/` still picks up this repo's
settings; started outside the repo it silently falls back to 79 columns, eager re-execution and no
format-on-save. That config is `lru_cache`d, so an edit to it needs a server restart, and in
VSCode the extension owns the server -- reload the window, since restarting the kernel will not
do it. It is young enough that its installed source under `.pixi/envs/` settles what
its documentation does not — `--mcp` is hidden from `--help`, `custom_css` does not expand `~` —
so check `marimo config describe`, then that source in a subagent, and only then the web.

A single notebook overrides that config from its own PEP 723 header, which
`ScriptConfigManager` merges at the *highest* precedence — above this repo's `pyproject.toml`.
The tutorials under `notebooks/pytorch-basics/` use it to set `on_cell_change = "autorun"`,
because the repo-wide `lazy` marks a cell stale instead of running it and a widget that does not
update is not a widget. Three sections are stripped from a script header for security
(`runtime.auto_instantiate`, `experimental.isolate_apps`, `display.custom_css`), which is
convenient here rather than limiting: opening one of those notebooks still executes nothing.
Same `lru_cache`, so the same window reload.

The VSCode extension contributes a *native* notebook rather than embedding marimo's web app,
so anything routed through marimo's own HTML — `display.custom_css`, the `--marimo-*-font`
variables — never reaches it. Cells there are governed by VSCode's `editor.*` and
`notebook.*` settings instead.

## Editing notebooks

A marimo notebook is a Python file: each cell is a function body, the last expression
statement is what renders, the `return` tuple is what other cells may use, an underscore
prefix keeps a name cell-local, and a top-level name may be defined by exactly one cell in
the file. marimo tracks reassignment, not mutation, so a mutation belongs in the cell that
creates what it mutates — several tutorial cells carry comments saying so, and moving the
mutation out breaks their idempotency.

**Folding.** `@app.cell(hide_code=True)` is the only folding the file knows. Policy in
`notebooks/pytorch-basics/`: a visible cell carries only code the narrative asks the reader
to read; display plumbing lives in hidden cells; the mutation demos keep their rendering
inline because snapshot timing is the demonstration. The VSCode extension (vscode-marimo
0.17.2) applies `hide_code` by collapsing cell input the first time a notebook is opened in a
window session, and thereafter compares against its own window-lifetime memory of what it collapsed — a manual
expand never updates that memory, so an expanded cell stays expanded through any number of
tab reopens. Folds "gone" while `git diff` is clean is therefore a session artifact, not
damage. A window reload (`ctrl+alt+shift+m`) resets that memory and refolds on the next
activation — but only the cells the editor has materialized, so on a long notebook some
cells stay expanded and, the memory now marking them folded, are never retried. The
reliable spot fix is per cell: the collapse chevron, or the "marimo: Hide cell code"
command, which also persists. That is stock 0.17.2. On this machine the extension is
patched — `home/editors/marimo.nix` in the dotfiles repo, re-applied on every extension
update — so every notebook activation re-folds every `hide_code` cell and the file is
the authority on visibility; a manual expand lasts until the next tab switch. The stock
description stays because it is what any other machine, and any upstream report, sees.

Every notebook edit ends with a fold audit — not only polish passes: any cell whose code the
narrative does not ask the reader to read carries `hide_code=True` in the file, checked before
the commit that carries the edit. Display logic and narrative code do not share a cell: where
they are mixed, split — the content cell stays visible and may end by rendering the object it
made, the display-assembly cell folds. Names flow between cells, so the split costs nothing.
The standing exceptions: the mutation demos, whose render timing is the demonstration, and
cells whose in-loop `mo.output` redraw *is* the exhibit (07's trainer) — those can't split the
display out, but their trailing displays (a stat row after the loop) still can and do.

**Interactive cells.** Mechanics that cost a day to learn: a UI element is inert unless bound
to a global name (`mo.ui.array` for groups); a button's `on_change` fires only if its value
changes (`value=0, on_click=lambda v: v + 1`); `mo.ui.anywidget` takes no `on_change` — watch
its `.value` from a downstream cell, and dedupe (first click of a fresh widget counts).
No surface reliably repaints the cell the user just interacted with, so displays that must
change live in their own cells, controls stay static or trial-agnostic, and recorded data is
derived from state at event time — never from a rendering's closure — with a state-generation
guard so a stale surface drops a click instead of mis-recording. A cell-local (underscore)
name referenced from inside an exported function resolves only if it is defined *above* that
function in the cell: a later definition stays unmangled in the function body and NameErrors
at call time — **only under `marimo run`/`edit`**, because script execution shares one
namespace and never mangles. So helpers precede their exported callers, and an interactive
instrument's checks include a served-page smoke test, not just the script run (measured
2026-09-03, calibrate-aesthetics). A `marimo run` page confines outputs to a ~700 px prose
column; a stimulus that needs width (two code pages at true editor sizes) breaks out with
`position:relative; left:50%; transform:translateX(-50%); width:min(96vw, 1400px)` on the
widget's root — and stimulus `<pre>` blocks use `overflow:hidden`, never `auto`, so no
scrollbar chrome joins the picture being judged. Reaction-time instruments carry a pause
that hides the stimulus and restarts the clock on resume (an exposed stimulus lets the
decision form off the clock), auto-pause on tab hide, and a `paused` flag per record. UI elements interpolate into
`mo.md` f-strings (how controls sit on a styled ground); multi-paragraph `mo.md` inside an
`hstack` lays its paragraphs out horizontally in the VSCode renderer — use explicit stacks.
Interactive instruments are verified by clicking them in a real browser (`marimo run` +
`xdg-open`); the embedded surfaces lie (driving-notes in the vscode-keyhole skill).

**Never rewrite a notebook on disk while it is open in the editor.** The extension syncs
cells by id in transactions, and an external rewrite produces
`ValueError: Cell 'X' already exists` or spurious multiple-definition errors that exist
only in the editor session while the file stays valid. Recovery: `ctrl+alt+m` in the
focused notebook closes it *discarding* the stale model and reopens it from disk — a fresh
deserialize of the cells, though not of the folds, which take the window reload above.
Discarding is the point: saving a stale model writes the
broken merge over the good file, and can silently drop `hide_code` from any cell whose
`marimo.options` metadata was lost. The window reload named under Environment remains the
fallback when the whole session is confused. After any editor save, `git diff` and look
for decorator churn.

A live session that reopens into spurious "this cell redefines variables" errors resyncs
with Restart Kernel followed by Run All — cheaper than the window reload, measured working.

`ctrl+alt+m` assumes the notebook's session is live and resyncs on reopen. A notebook that sat
open but never ran when the rewrite landed can wedge the extension for the whole window session
instead: its first run collides as above, and after a kernel restart or session shutdown the
reopened editor never re-attaches — cells queue, mark stale, the kernel idles at zero CPU. Only
the window reload clears it (established by walking a full recovery attempt). Before rewriting a
notebook that is open in a tab but has no live kernel, close the tab; failing that, verify
headless and let the first editor run wait for the reload. In the meantime,
`pixi run marimo run <nb>.py --headless --no-token -p <port>` plus VSCode's Simple Browser is a
fully rendered, interactive stand-in for the wedged editor surface.

**Checks.** `pixi run marimo check --strict` catches duplicate names and unparsable cells
without running anything. The content check is executing the file itself — from inside
`notebooks/pytorch-basics/`, `CUDA_VISIBLE_DEVICES=0 pixi run python <nb>.py` — where exit 0
means every cell ran, and a failing cell exits nonzero (both directions measured). Run it from
the notebooks directory, never the repo root: `root="data"` is cwd-relative, and a root-level
run silently downloads a second 82 MB FashionMNIST copy where the hook once caught one.

**Precedence.** The marimo extension ships a pairing skill whose contract is "never edit the
notebook file; all changes through code mode." That contract governs working *inside* someone's
live session; this repo's polish flow — edit on disk, verify headless, reopen — governs
autonomous passes and wins here. The pairing contract still applies when driving a kernel Titus
is actively using.

**The upstream baseline is commit `891febb`** — the eight tutorials exactly as converted.
`git diff 891febb -- <notebook>` shows everything ours against everything inherited; the
original's text is provenance to consult, not a boundary to respect.

**Converted-tutorial traps** (each cost a polish agent a debugging loop): the converter emits
GitLab math (``$`…`$``, ```` ```math ```` fences) which marimo renders literally — the
supported syntax is `$…$`/`$$…$$`. It also re-attaches standalone comments as trailing
comments on the *wrong* lines — diff comment placement against the upstream commit
(`891febb`) when a cell reads oddly. Exception messages interpolated into `mo.md` need
sanitizing (PyTorch embeds ANSI escapes in them). Vega scheme names can invert orientation
against the house constants — `blueorange` is the reverse of `POLARITY`; diverging exhibits
use `POLARITY`/`RAMP` explicit ranges, and non-Vega renderers (graphviz) compute their hexes
from `_palette` (there's a `tint()` there for mixing toward a card's white). Ruff's isort
config keeps first-party `_viz`/`_palette` imports in the same block as third-party ones —
no blank line. Cells chained through the filesystem (save→load) have no DAG edge: file-order
keeps them correct headless, but a lone re-run of the save cell won't re-trigger its loads.

## Claims

Version-dependent behaviour, defaults and API semantics are where confident wrongness happens.
Write the claim as a sentence, then check whether your check tests *that sentence* — verifying an
adjacent fact is worse than not checking, because it issues a false receipt.
`torch.backends.cudnn.allow_tf32` really is `True` by default, and the claim that it guarded this
model's matmuls was false anyway.

Cheapest check first: `inspect.signature` or `getsource` on one function costs a few tokens where
reading a module costs thousands. Send real source spelunking to a subagent, so it never enters
this conversation.

Prefer the forward-looking API to the one gathering dust — `set_float32_matmul_precision` over
`allow_tf32`, `dtype=` over `torch_dtype=`. It usually also names the thing that actually matters.

Three claim species the 01–08 pass caught repeatedly: a sentence about an *unseeded random
quantity* must attach certainty to the prediction, never to rounded equality ("exactly what a
uniform distribution gives" was false ~2% of draws — loop the draw to check); a *layout* claim
about a drawn exhibit ("read it bottom-up") survives every value check, so assert the layout
fact itself (rank direction vs traversal direction); and a *forward link* is a claim about its
target — read the target's headings before promising it answers a question. Exhibits gated
behind `mo.stop` are invisible to a headless run: their claims need a standalone replication
script.

## The one enforced rule

**A script in `scripts/` imports nothing from this repo.** Duplication between scripts is
correct: it lets one change without disturbing the baseline it is compared against.
`tests/test_boundary.py` enforces it.

## The rules improve

Every session ends with a rules audit: what did the work teach about these documents
themselves? Mechanics — sync hazards, recoveries, traps, checks — land in this file in the same
session, in the commit that closes the work. Intent-level changes (SKILL.md entries,
pedagogy.md) are proposed to Titus, never self-ratified — and a proposal he has not yet ruled
on is parked below rather than left to die with the conversation. A friction met twice is a
documentation bug.

This file is one reef among several: **the global reef map** — the registry of every living
knowledge file across his machine and projects, and the tending protocol that maintains
them — lives in the `titus-preferences` skill. Founding, moving, or retiring any reef
updates that map in the same commit.

**Queue** (work Titus has named but not yet aimed a session at; take an item only on his go).
This queue is the operational truth — resume points and mechanics; the owner-level ledger is
`~/src/orga/membrane/inbox.md` (local-only, never push), where each deferred program carries a
one-line spark. When a queue item is opened, closed, or founded, mirror the one-liner there:

- The full polish pass over 01 and 05–08: **COMPLETE 2026-09-02** — 01 fcbbed4, 05 108479e,
  06 fcfb57b, 07 baad8e5, 08 ee20cb2, all pushed; salvage stash dropped after both halves
  landed; the agents' rule-learnings are folded into "Editing notebooks", "Claims", and
  "Stopping agents" here, `tint()` promoted to _palette.py, and the invariants hook now
  flags only relative dead links. Residuals for Titus's eyes (headless runs can't see
  rendering): 06's KaTeX math cells and its graphviz card on both themes; and any notebook
  open in a tab was rewritten on disk — ctrl+alt+m reloads it before running there.
- Code-reading efficiency in the editor across languages: research + measure (semantic
  highlighting, token-color overrides per the Horizon findings, font/ligature choices), using
  the calibration data as it accumulates. Named 2026-09-02.
- Screen calibration (hardware/ICC) — named 2026-09-02 as "another day"; until then the
  vision-calibration data is relative-to-this-screen, which its prose says.
- Residual for Titus (2026-09-03): calibrate-vision.py was rewritten on disk (v2) while
  open in a VSCode tab with a live kernel — `ctrl+alt+m` in that tab before clicking there
  again. Until then its stale session keeps recording valid v1-generated trials; the log
  stays sound either way.
- **Ownership note (2026-09-03): the four measurement items below — ground optimization,
  observer-model refinements, confusion-axis re-rank, and the glyph-scale stage — are
  absorbed into the aesthetics-instrument session's brief** (Titus's expansion order): one
  session owns the measurement↔preference interlock, so sharpened measurements regenerate
  the constraint set its optimizer searches. They stay listed for their content; resume
  points now live with that session's instrument.
- Ground optimization: **INSTRUMENTED 2026-09-03** — calibrate-vision's blocks now cycle
  seven grounds (Horizon day/night, Selenized light/dark, Modus white/black, GitHub dark)
  and the observer models threshold as a smooth function of ground lightness (gL, fitted).
  Warmth stays out of the model until the ground family decouples it from lightness (noted
  in _observer.py); reading comfort remains the preference instrument's half. Verdicts wait
  on trials at the new grounds.
- Calibration observer-model refinements: **SHIPPED 2026-09-03 as _observer.py v2** —
  CAM16-UCS space, fitted slope (β ≈ 1.2, correcting v1's assumed 2) and lapse, free
  confusion-axis rotation (the red–green asymmetry, generalized), threshold smooth in
  ground lightness, small-field exponent for the glyph stage. Exact grid posterior
  (3.6M cells, chunked, cached, incremental per click) — still CPU; GPU (numpyro/BoTorch)
  remains sanctioned when the grid needs to grow past a few thousand trials.
- The theme gallery's sequential prose still describes the retired blue RAMP; update it to
  the cividis house ramp once the swap survives Titus's reading.
- Confusion-axis re-rank of the theme gallery's dropdown: **DONE 2026-09-03** — ordered by
  worst adjacent-pair discriminability under the fitted observer, rainbows pinned last by
  the monotonicity rule. Finding for Titus: measured order puts magma/inferno first and
  cividis eighth — the house ramp's CVD-safety rationale is not binding for his eyes; the
  applied cividis choice stays his to revisit (see the gallery-prose item above).
- Interlock session resume point (2026-09-03, all pushed through this line): preference
  instrument serving on :2919 (pane `calibrate-aesthetics`, lock `port-2919`) awaiting his
  duels; vision v2 awaits his clicks at the new grounds/sizes (open tab needs ctrl+alt+m
  first — residual above). Next when data or he arrives: γ (glyph) and gL-at-new-grounds
  verdicts from his trials; add ground *warmth* to _observer once the ground family
  decouples it; fold his loved colors into the aesthetics prior (still unanswered); GPU
  posterior only if the grid must grow (sanctioned, not yet needed); the cividis-prose
  queue item remains untaken.
- The aesthetics-preference instrument: **BUILT 2026-09-03** —
  `notebooks/pytorch-basics/calibrate-aesthetics.py` (torch-free, serves under `marimo run`),
  exactly the kicked-off design: preferential Bayesian optimization (Bradley–Terry duels
  with reaction-time-scaled slopes, Laplace-approximated GP posterior, Thompson-plus-EIG
  acquisition, 7% uniform probes, 5% champion-vs-worst anchors) over a 9-axis CAM16-UCS
  theme space plus a binary polarity block coordinate; comprehension micro-tasks and
  find-highlight time-to-find hunts share the loop; floors (WCAG 4.5 + APCA 60/45 + 2× the
  measured CAM16-UCS thresholds refit live from calibration-responses.jsonl) are hard
  constraints realized by walking lightness to the bar; Ou–Luo/Berlyne/warm-tilt prior
  mean. Verified: strict check, headless exit 0, 160-trial synthetic sitting recovers a
  planted optimum (100% day / 47% night at ~35 duels each), deterministic replay from the
  log across fresh kernels; browser-verified 2026-09-03 (12 duels + both task modes clicked
  over CDP against a served copy: rows carry both timestamps, double-click dedupes, grounds
  pixel-match the log, and the loop resumed correctly across a server restart). Data:
  `aesthetics-responses.jsonl`. **Serving on port 2919** (zellij pane `calibrate-aesthetics`,
  lock `port-2919`) — Titus clicks next. Ecological-valence prior is a stand-in
  until he names loved colors (asked in the intro prose).
- The theme program (named 2026-09-02): determine independently the best *editor* theme for
  Titus (best-in-class as the starting field — Selenized, Modus, GitHub accessible, Horizon —
  then self-evolved: token-color overrides tuned by his calibration data and the gallery's
  contrast instruments) and the best *graphing* theme independently; then characterize how the
  two interact (shared grounds, simultaneous contrast, accent-vs-data-hue collisions) and
  determine the best combination. Builds on calibrate-vision data; the deliverable is a ranked,
  measured recommendation plus the override files to apply it. Titus explicitly deferred the
  glyph-scale build on 2026-09-02 ("not a green light for now") — his focus is the learning
  notebooks; the applied best-guess theming (Horizon + measured overrides, cividis house ramp,
  Okabe-Ito categorical) stands until he re-opens this. A question about the stage is not a go.

**Stopping agents gracefully.** When work must stop (credits, interruption), message running
agents "commit what is verified, then stop" instead of killing them: a kill converts in-flight
verified work back into future spend. Kill only when spend must stop this instant, then stash
uncommitted edits and park a resume point in the queue (measured: one killed agent's full
re-read, versus two whose partial commits survived). Salvaging a stash later: finish the
*intent*, not the diff — an imported-but-unused name in stashed WIP is the fingerprint of the
interrupted edit it was for. Reading one file out of a multi-file stash:
`git diff 'stash@{0}^' 'stash@{0}' -- <path>` (the `stash show -p -- <path>` form errors).

**A go is standing.** Once Titus has aimed work — "go", or "go on the things you proposed" —
sessions keep moving through it and through the queue autonomously; the proposals block gates
*text ratification*, never work he has already directed. Parking directed work behind another
confirmation is the failure this sentence records.

- Resume the interrupted series pass (stopped 2026-09-02, credits): 01 and 05 have partial
  commits pushed (through 55c45d6, d1614ea, 2e2e01d); 06 and 07 were mid-verification with
  nothing staged (07 had found a falsified claim: "hundred and twenty printed loss values" is
  100); 01's phase-3 and 08's mid-pass edits sit in `git stash` ("WIP from stopped polish
  agents"). Re-dispatch one agent per notebook with the same instructions as before.
- Apply the computed Horizon token overrides (the remaining syntax layer; workbench layer
  already exists in VSCode settings.json with the same method at 6:1). Day, on #FDF0ED:
  strings #F6661E → #A13A06 (6.04:1), functions #1D8991 → #15646A (6.16:1), comments
  (drop the 50% alpha) → #605A59 (6.08:1); day variables override #BF0E37 already present
  (5.66:1). Night, on #1C1E26: comments (drop the 30% alpha) → #82858F (4.51:1), variables
  solid #E95678 (4.80:1, alpha was the whole problem). Same hue and saturation throughout —
  walk lightness only. Target keys: editor.tokenColorCustomizations per theme.

**Results live with their instrument** (notebook prose) or in Titus's Notion — this file
carries rules, routing, and resume points only. The calibration program's current findings
and their interpretation are in calibrate-vision.py's closing section.

**Proposals awaiting Titus** (ratify by moving into the target file; reject by deleting):

- none right now.

## Notion

Notion is Titus's own knowledgebase: sessions do not read from it as authority or write to it
at all unless he explicitly asks, per request. Where this file says results or understanding
"live in Notion," that means he writes them there — it is never an instruction to touch it.
This repo does not mirror Notion either: a second copy drifts, and a stale copy gets read as
authoritative by whoever finds it first. Refer to a topic by name, never by number, since a
numbered cross-reference breaks every other page as soon as the order changes.

## Commits

Straight to `main`, never a branch: more than one agent works here, so a branch only diverges from
what the other is committing. Stage paths explicitly — `git add <paths>`, never `git add -A`.
**Ask before staging anything under `scripts/` or `notebooks/`**; those are Titus's.

**Push every commit or commit batch to origin immediately.** Unpushed local-only commits are a
single-disk risk for no benefit. A push that fails (no key, no network) is reported in the
summary, never left silent.

## Style

United States spelling. `.githooks/pre-commit` is the formatting standard and enforces itself; run
it rather than trusting a description of what it checks. Expand every acronym on first use. Code and prose document themselves and never reference the conversation
that produced them.

## Where to look

`DECISIONS.md` before changing a design choice under `scripts/`: it carries reasons, and a reason
that no longer holds is grounds to revisit rather than to comply. `figures/README.md` before
touching the figure pipeline.
