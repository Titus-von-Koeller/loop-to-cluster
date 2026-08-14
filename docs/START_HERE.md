# Start here

`CLAUDE.md` is the standard, `PROFILING.md` the measurement contract, `docs/CONVENTIONS.md` the
standard for the Notion pages. This file carries only what those cannot: current state,
decisions already taken, and what not to bring back.

## State

Restructured 2026-08-07 after a code review by Marc Sun. Clean and green.

- `scripts/` — `00-basic-loop.py` is the baseline: SmolLM2-135M from a random-initialized
  config, wikitext-2, three epochs. Untracked and Titus's; never stage his paths.
- `l2c/` — profiling harness. 21 tests, plus one per study script, since `test_boundary.py`
  parametrizes over `scripts/*.py`.
- `bench/` — empty. Results and figures land here.
- `.claude/skills/` — `write-chapter`, `refine-chapter`. A *new* skill directory is discovered
  live, but an *edit* to an existing `SKILL.md` serves from a cached payload until restart, so
  read from disk when it matters.

## Next

1. **`profile-script`**, at `.claude/skills/profile-script/SKILL.md`. Takes a study script,
   produces its twin following `PROFILING.md` exactly, runs it, writes JSON to `bench/results/`
   and a figure to `bench/figures/`. Put the twin template in the skill directory so generating
   one is filling slots rather than recalling prose — that is what keeps measurement identical
   across topics.
2. **Run it on the baseline** and check the signals in `PROFILING.md`.
3. **Topics**, one per script, each a copy-and-modify of the baseline.

In parallel, the Notion pages. *The loop* and *Memory and compute* have been rewritten and are the model to follow. Next is *Mixed precision*, which is step one of the plan on the parent page.

## Decided

- **Random init** via `AutoConfig.from_pretrained` then `from_config`, because a pretrained
  checkpoint forfeits the `ln(V)` prediction — the only analytic number available.
- **Real text**, not a synthetic batch. The loop is where model, data, optimizer and loss meet,
  and a synthetic tensor removes one of the four.
- **`model(ids, labels=ids).loss`**, which HuggingFace's own course uses. Measured identical in
  peak memory to an explicit `cross_entropy` with `del logits`.
- **Nested epoch loop**, with the dataset sized so an epoch completes: `train[:5%]` gives 128
  steps per epoch, so the outer loop is real rather than decoration.
- **No learning-rate schedule.** A decaying schedule makes the learning rate a function of the
  total step count, so a later script that changes the step budget is no longer comparable.
- **Keep `cfg.dtype` and `cfg.use_cache = False`.** Neither is canonical and both prevent a
  silent error: bfloat16 parameters that halve every ledger row, and a key-value cache
  belonging to no row.
- **No eval loop for now**, which makes this a demonstration of the loop rather than complete
  training — the memorization question stays unanswered until it is added.

## Do not resurrect

A previous design was replaced wholesale. Traces in git history or stale comments stay there:

- `steps/stepN_*/` directories, `prediction.toml`, `NOTES.md` as a per-step lab notebook
- a shared `collect` / `publish` / row-spec reporting layer
- `build_model` / `build_loader` / `build_batches` / `training_step` wrappers
- a shared argparse module, or a subprocess runner driving several arms at once
- `results.jsonl`

Study scripts are read linearly and modified independently, so shared helpers cost
comprehension and let one experiment perturb another.
