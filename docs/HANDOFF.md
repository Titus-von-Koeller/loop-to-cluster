# Handoff — read this before touching anything

You are picking up a project that got away from the previous instance. It works, it is
tested, and it is **the wrong shape for its purpose**. Your job is to change the shape,
not to add capability.

Read this file, then `CLAUDE.md`, then produce a written plan and **stop**. Do not explore
the codebase before you have read both. Do not implement before Titus has approved the
plan. The previous instance explored first, built continuously, and that is precisely how
it went wrong.

## 1. What this project is for

Titus is learning distributed training, to onboard onto HuggingFace **accelerate**. The
method is: *predict the numbers, then measure them, then explain the gap.* One concept per
step, from a single-GPU loop to sharded multi-GPU.

**This is a learning exercise, not a codebase.** The deliverable is not working software.
The deliverable is Titus understanding the mechanisms at the level of the public torch API.
Working software is the substrate.

Two consequences that are easy to get wrong:

- A step file exists to be **read**. If the concept is buried, the file has failed even if
  it runs perfectly.
- Titus does the predicting and the deriving. If you do it for him, you have deleted the
  exercise, however good your numbers are.

## 2. How the previous instance failed — do not repeat these

These are specific, and each one has a specific guard.

**It expanded scope on every piece of feedback.** Asked for a baseline plus mixed
precision, it delivered four precision arms, a comparison driver, a saved-tensor ledger,
a least-squares module, a subprocess runner and 37 tests — 2,100 lines. Each addition was
individually defensible. Cumulatively it replaced "a simple loop Titus writes" with "a
harness Claude wrote."
→ *Guard: when you think of an addition, write it in a "not now" list and carry on. Adding
nothing is the default this phase.*

**It did the learner's work.** It filled in `prediction.toml` and wrote the derivations in
`NOTES.md` — which is Titus's lab notebook. It optimized for a complete demo instead of a
complete exercise.
→ *Guard: never write a value into `prediction.toml`. Never write prose into `NOTES.md`.
If you know a number, do not state it anywhere, including commit messages.*

**It applied its own architectural rule inconsistently.** The project rule is "if a lesson
touches it, duplicate it; if no lesson touches it, it goes in the package." It verified the
training loop had not leaked into the shared package, and never checked the reverse. 49% of
`steps/step1_training_loop/train.py` is reporting and serialization.
→ *Guard: for every line in a step file, ask "is this the lesson?" If no, it belongs in
`l2c/`.*

**It went below the stated abstraction level.** It traced into `transformers` and `torch`
source to explain behaviour. Each excursion was provoked by a measurement that did not
match, which is defensible — but it never asked whether that depth was wanted. It was not.
→ *Guard: the depth ceiling in `CLAUDE.md` is binding. accelerate source is in scope;
torch and transformers internals are not.*

**It over-corrected when criticized.** Told the learning had been compromised, it invented
a "spoiler quarantine" that was paternalistic and misidentified the risk — seeing a number
in a chat log is not the same as having derived it. It swung from too little care to too
much.
→ *Guard: when Titus corrects you, make the specific correction. Do not generalize it into
a new regime.*

**It asserted a claim about accelerate without checking accelerate.** It wrote that
accelerate leaves `float32_matmul_precision` to the caller. It does not — see
`state.py:1023`. accelerate was installed the whole time. The error survived two turns and
was nearly relayed to a colleague.
→ *Guard: accelerate is installed. Read it before claiming anything about it. File and line
or do not say it.*

**It was verbose.** 37-line module docstrings before any code; 1,500-word `NOTES.md`
against a natural 400.
→ *Guard: the 400-word cap in `CLAUDE.md` is a hard cap. Step docstrings: 12 lines.*

## 3. Current state

`HEAD` is green and demonstrable:

```bash
cd /home/titus/src/loop-to-cluster
CUDA_VISIBLE_DEVICES=0 pixi run pytest tests/ -q          # 37 passed
CUDA_VISIBLE_DEVICES=0 pixi run python steps/step1_training_loop/train.py
CUDA_VISIBLE_DEVICES=0 pixi run bash steps/sweep.sh       # fits R^2 = 1.00000
CUDA_VISIBLE_DEVICES=0 pixi run python steps/step2_mixed_precision/compare.py
```

What exists:

| | |
| --- | --- |
| `l2c/harness/` | measure, ledger, predict, precision, report, fit, runner |
| `l2c/common/` | model construction (SmolLM2-135M config), packed wikitext-2 |
| `steps/step1_training_loop/` | complete, and a deliberate worked reference |
| `steps/step2_mixed_precision/` | complete — but it is **Claude's** work, and Titus wants to do it |
| `tests/` | 37 tests over the harness arithmetic |
| `results.jsonl`, `runs/` | measured evidence, tracked on purpose |

Steps 3–7 do not exist. Roadmap is in `README.md`.

## 4. Your work queue — in this order, nothing else

### 4.1 Refactor reporting out of the step files

This is first because it is an ordering constraint, not a preference. Titus will hand-write
5–15 concept lines per step. In a 300-line file he cannot find them.

Move to `l2c/harness/`:

- the `actual` dict assembly → a `collect(...)` taking the measured objects
- the `rows` list → built in the harness from prediction plus actual
- all printing, `environment()`, `record()`, `--json-out` → one `report.publish(...)`
- shared argparse knobs → `harness/cli.py::common_args(parser)`, so a step declares only
  its own flag and `--precision` stands out as the step's subject
- the 7-kwarg `DataLoader` → `data.build_loader(...)`, rationale living in `data.py`

Keep visible in the step:

- the `phases → warmup → timed → anatomy` skeleton. **Do not collapse this into
  `lab.run(step_fn)`.** It is the profiling methodology being taught, not plumbing.
- the device loss buffer and the absence of `.item()` in the loop
- `loss_from` and `training_step` — the concept lines

Target: ~12 docstring, ~8 imports, ~4 step-specific args, ~8 setup, ~10 concept, ~33
measured run, ~5 publish. **~90 lines total.**

### 4.2 Collapse three persistence paths into one

`results.jsonl`, `runs/*.json` and `--json-out` overlap; the last two are identical. Keep
`runs/*.json` keyed by config — one file per run, no append-log conflicts, and the runner's
cache and the permanent record become the same artifact. Point `fit.py` at it.

### 4.3 Separate the insights from the answers

Two buckets. Getting this wrong either loses hard-won findings or spoils the exercise.

**Bucket A — instrument and library facts. Always visible. Not answers to anything.**
Home: `docs/findings.md`, plus the docstrings where they already live.

- `memory_allocated()` returns allocator *block* sizes, not tensor sizes — why
  `state_inventory` exists
- autograd saves a Linear's weight *transposed* — why the ledger classifies by provenance
- a weight whose input does not require grad is never saved — why the cast-count invariant
  is `<=` and not `==`
- AdamW allocates its moments lazily on the first `step()`
- accelerate couples TF32 to torch.compile (`state.py:1023`)
- `Accelerator.backward` divides by `gradient_accumulation_steps` (`accelerator.py:2840`);
  `clip_grad_norm_` unscales first (`:2944`)

**Bucket B — answers to specific step predictions. Sealed per step.**
Home: `steps/stepN_*/reference/` — the worked `prediction.toml`, `NOTES.md` and measured
`runs/*.json`, git-tracked, with one line saying "open after you have committed your own."
Moved, never deleted.

Step 1 is the exception: it stays complete and in place as a labeled worked example. It is
the pattern and the demo.

### 4.4 Set step 2 up as an exercise

- `train.py`: scaffold generated; the concept lines stubbed as `NotImplementedError` with a
  comment naming what belongs there. Where useful, point at the **accelerate** call that
  does the same job — never the torch answer. Roughly 8 lines: a grad scaler, an autocast
  context around the forward, and the ordering among the scaler calls and the clip.
- `prediction.toml`: keys present, values absent, one comment per key saying what to derive.
- `NOTES.md`: headings only — What this step adds / Prediction working / Result / What
  surprised me / Bare torch → accelerate. Note the 400-word cap in the file.
- Move Claude's version to `reference/` and purge step-2 rows from `runs/`.
- **Default the exercise to `--batch-size 8 --seq-len 256`.** That holds tokens/step at
  2048 while halving the `B·heads·T²` attention term, so exactly one term moves. It tests
  whether Titus understood *which terms scale with what* — the reference config cannot,
  because there tokens and memory move together and an error hides in the correlation.
  Make it explicit in the scaffold that `reference/` was measured at a different config, so
  nobody later compares the tables cell by cell and concludes the harness drifted.

### 4.5 Stop

Do not start steps 3–7. Do not build the profiler pass. Do not build the gradient
histogram. Both are wanted eventually and are recorded in §6.

## 5. Rules

Binding, from `CLAUDE.md`:

- **Ownership.** Claude writes `l2c/`, `tests/`, step scaffolding, argparse, dataloaders,
  reporting, JSON. Titus writes `prediction.toml`, `NOTES.md`, and the 5–15 concept lines.
- **Depth ceiling.** Explain at the level of the public torch API and its documented
  behaviour. State *what* an API does and what it costs. Do not trace into torch or
  transformers source to explain *how*. accelerate source is always in scope, file and line
  encouraged.
  - This governs **teaching surfaces** — step files and `NOTES.md`. `l2c/` internals may go
    as deep as correctness requires; the ledger cannot function without naming autograd node
    types. But `l2c/` docstrings state what a function *guarantees*, not how it works.
  - The test: if Titus can read a step file and its NOTES and predict correctly without
    opening `l2c/`, the ceiling is holding.
- **`NOTES.md`: 400 words per step, hard cap.**
- US spelling. Ruff-clean (`E,F,I,UP,B,SIM,RUF`, line length 95). Python 3.14 idiom — PEP
  649 makes annotations lazy, so no `from __future__ import annotations`; `type` aliases,
  `StrEnum`, `tomllib`, `datetime.UTC`.
- Code documents itself and never references the conversation that produced it. No "as
  discussed", no "originally this was", no narrating a fix.

## 6. Wanted eventually, not now

- **Profiler pass over the precision arms.** Step-time predictions missed by roughly 48%
  and the current diagnosis is a description, not a model. `torch.profiler` is available
  and unused; it would split GEMM from elementwise from launch gaps and turn "the kernels
  are small" into something predictive.
- **Gradient-magnitude histogram** against fp16's smallest normal and subnormal. The
  current fp16 conclusion rests on zero skipped updates, which evidences no *overflow* and
  says nothing about *underflow*.

## 7. Environment

- Two GPUs. **Always `CUDA_VISIBLE_DEVICES=0`** — GPU 1 drives a display, so its clocks
  move with the compositor and its memory starts several GiB down.
- **Always `cd /home/titus/src/loop-to-cluster && pixi run ...`.** A bare `pixi run` walks
  upward and binds a different workspace, silently.
- Python 3.14, torch 2.13 (CUDA 13 wheels from PyPI), transformers 5.x, accelerate 1.14.
- `accelerate` is installed but imported by nothing. It is there so claims about it can be
  checked against source.
- First data run tokenizes wikitext-2 into `.cache/`; after that everything is offline.

## 8. How to know you are done

- Both step files under ~100 lines, and the concept lines findable in five seconds.
- `pytest tests/` still 37 passed; `ruff check` clean.
- Step 1 still runs and still reports the same numbers. **The refactor is behaviour-
  preserving — if a measured value moves, you broke something.**
- One persistence path.
- `steps/step2_mixed_precision/` contains no derived values and no prose by you.
- `docs/findings.md` exists and contains no answer to any unstarted step's prediction.
