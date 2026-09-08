# Authoring these learning notebooks

Visible code must contribute to the explanation. Arrange each exhibit in reading
order: introduce the question, show the relevant code nearby, show its result,
then interpret it. A valid dependency graph does not guarantee a readable page.
Fold prose, chart styling, table assembly and other display plumbing with
`hide_code=True`; keep the computation the reader is learning visible. Controls
belong beside the experiment they affect. Optional benchmarks follow the core lesson.

The displayed evidence must come from that visible computation. Do not reimplement
the experiment inside a hidden chart cell. Export meaningfully named results and
let display cells consume them; `addition_traces` and `neighbor_spacing` in09 are
examples. Local underscore names are useful for scaffolding, not a default naming
policy or a substitute for designing dependencies.

Marimo tracks cell dependencies, not internal mutations of models, tensors,
optimizers, RNGs, files or device-wide settings. Keep a stateful experiment's
initialization and mutations within one clearly owned execution. Export fresh
result records or detached/cloned tensor snapshots when downstream code needs a
stable observation. A clone is still mutable: downstream consumers must treat it
as read-only. Rebuild the experiment on rerun; rerendering a chart must not train.
Restore temporary global settings in `finally` and remove temporary hooks.

Verify a meaningful rerun/change case, not just initial execution. For09, changing
the addition input must change its chart data, and later mutation of the source
model must not alter the exported gradient snapshot's interpretation. Check that
optional training stays button-gated in notebook use. Inspect the rendered page
in addition to lint and numerical checks; record which surface was inspected.

Notebook changes also require validation in the native VSCode marimo extension,
Titus's working surface. A browser preview or headless run does not substitute for
it. Confirm the project interpreter, execute the affected path, exercise a relevant
reactive change, and inspect outputs, reading order, code folds and optional-work
gates there. Record the notebook revision, extension version, actions and outcome.
If extension validation is blocked, preserve the candidate and name the unverified
behavior; do not describe the notebook as fully validated. Reuse unaffected evidence
and scope checks to the change rather than rerunning expensive training by default.

Follow the repository CLAUDE.md for environment, folding and editing workflow.
Use code mode when collaborating inside an active live session. For autonomous
work, inspect and preserve relevant state, close an unused notebook, then use the
supported disk-edit/validate/reopen flow. A stalled approval is not a reason to
overwrite an open model or stop unrelated work. Do not automate approval clicks.

Origin: direct authoring feedback during the mixed-precision revision, September8.
Apply these principles economically when touching another notebook; do not launch
a series-wide rewrite merely to impose uniformity.
