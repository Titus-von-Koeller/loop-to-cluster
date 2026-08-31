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

A *new* skill directory is discovered live, but an *edit* to an existing `SKILL.md` serves from a
cached payload until the session restarts.

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
0.17.2) applies `hide_code` by collapsing cell input when the notebook is (re)opened, and
thereafter reacts only to changes — expanding a cell by hand is transient editor state that
writes nothing and is not re-collapsed until the next reopen. Folds "gone" while `git diff`
is clean is therefore a session artifact, cured by reopening. Persist visibility with the
`marimo.hideCellCode` / `marimo.showCellCode` commands, never the collapse chevron, which
persists nothing.

**Never rewrite a notebook on disk while it is open in the editor.** The extension syncs
cells by id in transactions, and an external rewrite produces
`ValueError: Cell 'X' already exists` or spurious multiple-definition errors that exist
only in the editor session while the file stays valid. Recovery order matters: close the
tab *without saving* — saving writes the stale merge over the good file, and can silently
drop `hide_code` from any cell whose `marimo.options` metadata was lost — then reload the
window and reopen. After any editor save, `git diff` and look for decorator churn.

**Checks.** `pixi run marimo check --strict` catches duplicate names and unparsable cells
without running anything. The content check is executing the file itself —
`CUDA_VISIBLE_DEVICES=0 pixi run python notebooks/pytorch-basics/<nb>.py` — where exit 0
means every cell ran.

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

## The one enforced rule

**A script in `scripts/` imports nothing from this repo.** Duplication between scripts is
correct: it lets one change without disturbing the baseline it is compared against.
`tests/test_boundary.py` enforces it.

## Notion

Notion is where understanding lives, and this repo does not mirror it — a second copy drifts, and
a stale copy gets read as authoritative by whoever finds it first. Titus edits the pages, so a
passage that differs from what you expected has been *edited*, not damaged: name it and confirm
before touching it. Refer to a topic by name, never by number, since a numbered cross-reference
breaks every other page as soon as the order changes. Propose rather than write.

## Commits

Straight to `main`, never a branch: more than one agent works here, so a branch only diverges from
what the other is committing. Stage paths explicitly — `git add <paths>`, never `git add -A`.
**Ask before staging anything under `scripts/` or `notebooks/`**; those are Titus's.

## Style

United States spelling. `.githooks/pre-commit` is the formatting standard and enforces itself; run
it rather than trusting a description of what it checks. Expand every acronym on first use. Code and prose document themselves and never reference the conversation
that produced them.

## Where to look

`DECISIONS.md` before changing a design choice under `scripts/`: it carries reasons, and a reason
that no longer holds is grounds to revisit rather than to comply. `figures/README.md` before
touching the figure pipeline.
