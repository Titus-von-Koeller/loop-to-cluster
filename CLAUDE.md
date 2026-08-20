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

**Always `CUDA_VISIBLE_DEVICES=0`.** GPU 1 drives a display, so its clocks move with the
compositor and its memory starts several GiB down — a timing taken there is not a measurement.
Use `direnv exec <abs-dir> <cmd>` when you also need `nvcc` or nix's `libstdc++`. A missing tool
is not a dead end: `nix shell nixpkgs#<pkg> -c <cmd>`.

Resolved versions come from `pixi list <package>` rather than from memory or from a list here
that would go stale. `pixi.toml` records why each dependency comes from PyPI rather than
conda-forge; the reason is the torch ABI and it still holds.

A *new* skill directory is discovered live, but an *edit* to an existing `SKILL.md` serves from a
cached payload until the session restarts.

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
