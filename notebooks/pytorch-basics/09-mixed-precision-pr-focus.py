# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F

    return F, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From one correct update to a useful Accelerate PR

    *A focused companion to [Mixed precision](09-mixed-precision.py), for my own review.*

    My review target is a complete integration test for **multi-GPU training, mixed
    precision and gradient accumulation**. It compares Accelerate training with a
    plain-PyTorch reference using a tiny, locally initialized Gemma 4 model. Four named
    cases share one worker, so one cohesive PR currently makes the behavior easiest
    to review. PR count is a consequence of the design, not a target.

    **Start with sections 1, 2 and 4.** They explain an effective update, the actual
    assertions, and the precision/scaler behavior I need to follow in the code.
    Section 3 is an optional, executable explanation of the historical token-weighting
    bug. Section 5 explains the architectural and CI decisions. I do not need to
    finish the full mixed-precision curriculum before reviewing this contribution.

    I have enough background when I can explain which examples contribute to one
    update, why the reference is meaningful, what changes at each accumulation
    boundary, and how a broken wrapper would make an assertion fail. Any remaining
    code question should lead to a focused explanation, not another broad study detour.

    This notebook runs two small CPU experiments. It performs no downloads, training
    epochs, benchmarks or GPU launches. The actual two-GPU validation is a separate
    test run; a working teaching example is not evidence that the distributed test ran.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What should stay the same: one training update

    The model makes predictions using adjustable numbers called weights or parameters.
    The **loss** measures how wrong its predictions are on a batch of examples.
    Backward computes a **gradient**: how changing each weight would change that loss.
    The **optimizer** uses those gradients to change the weights. With stochastic
    gradient descent (SGD), the update is simply weight minus learning rate times gradient.

    The ordinary loop is forward → loss → backward → optimizer step. `backward()`
    adds gradients to `.grad`; `zero_grad()` clears them. Accumulation changes when
    I clear and apply gradients. Distributed Data Parallel (DDP) changes which
    examples each process sees and averages gradients between processes. Automatic
    mixed precision (AMP) changes the arithmetic used for selected operations.

    A **rank** is one participating process, usually assigned one GPU here. A
    **microbatch** is a smaller chunk processed before the complete weight update.

    Before comparing runs, specify what one update represents. With eight equally
    weighted examples, these layouts should represent the same update in exact
    arithmetic, provided the forward computation is independent across examples:

    | Layout | Examples per rank and microbatch | Microbatches per update | Global examples |
    | --- | ---: | ---: | ---: |
    | One process | 8 | 1 | 8 |
    | Two DDP processes | 4 | 1 | 8 |
    | Two DDP processes, accumulation | 2 | 2 | 8 |

    The prepared PR compares the first layout with both distributed layouts.
    Same batch size is not enough:
    initialization, actual examples, objective normalization, optimizer settings and
    update count must also agree. Disable dropout for this comparison. Batch-dependent
    layers and uncontrolled randomness can invalidate the equivalence; a shared seed
    does not make every distributed computation identical.

    Floating-point addition also depends on order. Compare within justified numerical
    tolerances, not necessarily bit for bit. A large systematic discrepancy deserves
    investigation before widening a tolerance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Read the complete contribution

    In the prepared Accelerate checkout, read `tests/test_causal_lm_training.py`
    (the launcher and assertions) alongside
    `src/accelerate/test_utils/scripts/external_deps/train_causal_lm.py` (the worker).
    The contribution builds on upstream `b795b483`. Its public PR description should
    name the tested revision, exact environments and any remaining CI limits.

    ### Four cases, one claim

    The claim is: **Accelerate's preparation and training wrappers preserve the
    intended update when we distribute, accumulate and change computation precision.**

    | Case | Computation | Microbatches per update on each rank | What it adds |
    | --- | --- | ---: | --- |
    | FP32 DDP | FP32 | 1 | Data distribution and distributed training |
    | FP32 accumulation | FP32 | 2 | Backward normalization and delayed updates |
    | BF16 accumulation | BF16 autocast | 2 | Prepared model's precision policy with accumulation |
    | FP16 accumulation | FP16 autocast plus scaler | 2 | Scaling, overflow skip and recovery with accumulation |

    Every case compares **one GPU running plain PyTorch on a batch of eight** with
    **two ranks using Accelerator on the same eight examples per effective update**.
    With accumulation factor two, each rank processes two examples twice before
    updating: $2	ext{ ranks}	imes2	ext{ examples}	imes2	ext{ microbatches}=8$.
    There are ten attempted updates. In FP16 only, the second deliberately gets an
    infinite gradient and must be skipped; the following finite update must succeed.
    That leaves nine successful updates in both FP16 paths.

    ### Why this model and data?

    A small random **Gemma 4 text model** has two dense layers: one sliding-attention
    and one full-attention layer. This is a real 2026 Transformers architecture with
    20,912 parameters here, using an adaptation of HF's tiny test configuration.
    Vocabulary 32, sequence length 12, no dropout and no cache keep it inexpensive.
    It has no vision tower, mixture of experts or pretrained checkpoint. Language
    quality is irrelevant: this test checks integration, not model quality.

    SGD uses learning rate 0.001. Larger trial rates amplified harmless rounding
    through successive updates. Lowering the rate must not hide a wrong update:
    the assertions therefore also measure disagreement relative to how far the
    reference's weights actually moved. The deliberate half-gradient check proves
    that this fixture still discriminates meaningful errors.

    Inputs are fixed locally using a separate random-number generator; model
    initialization is seeded identically. All rows have eleven next-token targets.
    Equal counts intentionally avoid making this test a separate investigation of
    variable-token loss normalization.

    ### Follow the prepared loop

    ```python
    local_batch = global_batch // (accelerator.num_processes * accumulation_steps)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    for (batch,) in loader:
        with accelerator.accumulate(model):
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
    ```

    This excerpt omits observation and the deliberate overflow. Read it with the
    worker's actual code. `prepare` moves/wraps the components and distributes loader
    batches. `accumulate` establishes microbatch boundaries. `backward` applies the
    configured accumulation scaling; **do not divide the loss by that factor again**.
    The optimizer wrapper delays step/zeroing on intermediate microbatches. The last
    backward synchronizes rank gradients and allows the effective update.

    `accelerator.reduce(loss.detach(), reduction="mean")`-style reporting is separate:
    it combines observations, not gradients. The worker averages microbatch losses
    within an update and then rank losses. Equal valid-target counts make that mean
    correct for this fixture. Correct logging alone cannot repair a wrong update.

    ### What each assertion buys me

    - **Ten reported losses and final weights agree** with the ordinary-PyTorch,
      same-precision, full-batch reference within measured per-precision tolerances.
      A second bound limits the norm of weight disagreement to 15% of the reference's
      actual weight movement. Correct BF16 runs are about 5%; halved gradients produce
      about 54% and fail. Absolute tolerances alone had missed that fault.
      The reference has no Accelerator preparation/backward/optimizer wrapper, so
      the same wrapper bug cannot automatically infect both sides.
    - **Both ranks report their observations, and final replicas agree exactly.**
      Inspecting only rank zero could miss divergence on the other GPU.
    - **Stored parameters remain FP32; an actual internal linear projection produces
      the requested dtype.** This detects an AMP setting silently doing nothing,
      which a tolerant numerical comparison could miss.
    - **Intermediate accumulation steps leave weights unchanged.** Gradients may grow;
      weights must wait for the update boundary.
    - **The deliberate FP16 overflow skips the update, halves the scale, and the next
      finite update changes weights.** The initial scale128 avoids an accidental
      startup-overflow experiment; this is a controlled integration check.

    The driver uses pytest functions, `tmp_path` and four named parametrized cases,
    plus Accelerate's existing launch and device/dependency helpers. PyTorch supplies
    `torch.testing.assert_close`; pytest is the test runner and authoring style.

    Agreement is bounded evidence. It does not prove every model/kernel/backend,
    convergence, checkpoint/resume, unequal-token objectives or partial final windows.
    Deliberately halved gradients, disabled AMP and a changed second-rank weight
    each fail the intended assertion in the actual driver. Tolerance changes need numerical evidence,
    not a desire to make failures disappear.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The bug hidden by equal-length examples

    The [Hugging Face report](https://huggingface.co/blog/gradient_accumulation)
    describes a loss-normalization error during accumulation. Zach's
    [follow-up](https://muellerzr.github.io/blog/gradient_accumulation_part2.html)
    explicitly credits Marc and extends the discussion to distributed training.
    This is the strong match for their debugging story. It is not principally an
    AMP bug: it can occur in FP32.

    Suppose two microbatches contribute loss sums $S_1,S_2$ and valid-token counts
    $n_1,n_2$. The token-mean objective for the whole update is

    $$L = \frac{S_1+S_2}{n_1+n_2}.$$

    Averaging their individual means instead computes

    $$L_{\mathrm{naive}} = \frac{1}{2}\left(\frac{S_1}{n_1}+\frac{S_2}{n_2}\right).$$

    Equal counts make these agree. Unequal counts generally do not: the smaller
    microbatch gets too much weight. For loss sums 8 and 8 from four and one valid
    targets, the intended mean is 3.2; the mean of the two means is 5.0. This changes
    the objective whose gradient I compute, not just the logged number.

    In causal language modeling, count the labels that actually enter the loss
    **after the next-token shift and after masking**. `-100` labels are ignored.
    Equal tensor shapes or equal numbers of sequences do not imply equal numbers
    of valid targets; padding and response-only masks can change the denominator.

    ### Make that mistake visible in one update

    The next experiment uses a tiny linear classifier with shifted token targets.
    It isolates the loss arithmetic; it is not a Transformer or a DDP integration
    test. Each path starts from the same weights and sees the same two sequences.
    The first has four valid targets. Change the second sequence's count below.
    """)
    return


@app.cell
def _(F, torch):
    def compare_one_update(second_count):
        inputs = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 10 - 1
        labels = torch.tensor([[0, 0, 1, 2, 3], [0, 3, 2, 1, 0]])
        labels[1, second_count + 1 :] = -100
        targets = labels[:, 1:]  # logits at t predict the label at t + 1
        total_targets = (targets != -100).sum()
        initial = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 20
        results = {}

        for method in ("full batch", "token-weighted accumulation", "mean of means"):
            weight = torch.nn.Parameter(initial.clone())
            optimizer = torch.optim.SGD([weight], lr=0.1)
            optimizer.zero_grad(set_to_none=True)  # once per complete update
            chunks = [slice(0, 1), slice(1, 2)]
            if method == "full batch":
                chunks = [slice(None)]
            objective = 0.0
            for chunk in chunks:
                logits = inputs[chunk, :-1] @ weight.T
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, 4),
                    targets[chunk].reshape(-1),
                    reduction="sum",
                )
                if method == "mean of means":
                    loss = loss_sum / (targets[chunk] != -100).sum() / 2
                else:
                    loss = loss_sum / total_targets
                loss.backward()  # adds this contribution; no step between chunks
                objective += loss.detach().item()
            gradient = weight.grad.detach().clone()
            optimizer.step()
            results[method] = {
                "objective": objective,
                "gradient": gradient,
                "weight": weight.detach().clone(),
            }
        return results

    return (compare_one_update,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Run both cases:** the first sequence always has four valid targets. Compare
    one valid target in the second sequence with four. The function above rebuilds
    weights and optimizer each time, so rerunning this cell cannot keep training an
    old model.
    """)
    return


@app.cell
def _(compare_one_update):
    update_comparison = []
    for second_count in (1, 4):
        update_results = compare_one_update(second_count)
        reference_update = update_results["full batch"]
        for method, result in update_results.items():
            update_comparison.append(
                {
                    "target counts": f"4 + {second_count}",
                    "method": method,
                    "objective": result["objective"],
                    "max gradient error": (result["gradient"] - reference_update["gradient"]).abs().max().item(),
                    "max weight error": (result["weight"] - reference_update["weight"]).abs().max().item(),
                }
            )
    return (update_comparison,)


@app.cell(hide_code=True)
def _(mo, update_comparison):
    _rows = "\n".join(
        f"|{row['target counts']}|{row['method']}|{row['objective']:.6f}|"
        f"{row['max gradient error']:.3g}|{row['max weight error']:.3g}|"
        for row in update_comparison
    )
    mo.md(f"""
    Errors are relative to the corresponding full-batch update.

    | Valid targets | Method | Objective | Max gradient error | Max weight error |
    | --- | --- | ---: | ---: | ---: |
    {_rows}

    **Equal counts hide the mistake.** With counts **4 + 1**, the mean of means
    changes the weights incorrectly; token-weighted accumulation agrees with the
    full batch within rounding. Correct logging alone would not fix the wrong weights.
    This table reads detached snapshots from the visible computation; rendering it
    never trains the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What changes when the pieces live on different ranks?

    Let $N$ count valid targets over **all ranks and the whole accumulation window**.
    With $W$ ranks, ordinary DDP averages the rank gradients. If each rank backpropagates
    its local summed loss divided by $N$, that average introduces an extra factor
    $1/W$. For plain DDP with summed microbatch contributions, compensate with
    $W S_{r,j}/N$ before backward. Then the average produces the intended global gradient.

    Wrappers can introduce another factor: Accelerate's ordinary accumulation path
    divides the backward loss by the configured accumulation steps. Its documented
    token-sum recipe compensates for that as well. Trace the installed path before
    applying any multiplier; copying a world-size/accumulation factor into a path
    that already applies it produces a different bug. See the
    [current Accelerate recipe](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#gradient-accumulation-on-training-samples-of-variable-size).

    To test this, make valid-token counts unequal across ranks **and** microbatches,
    while keeping the reference's complete update identical. Count real targets in a
    partial final window. Use a scalar count reduction rather than gathering logits
    just to get a denominator. For communication efficiency, `no_sync` should cover
    forward and backward on nonfinal microbatches; the final backward synchronizes.
    Those communication details are separate from getting the objective right.

    **Why our integration fixture excludes it:** every rank has the same number of targets,
    so even its accumulated batches do not challenge unequal-token weighting. These
    simplifying assumptions define exactly which regression it cannot catch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Mixed precision: three decisions, not one dtype

    **What is stored?** In the ordinary PyTorch AMP recipe, parameters stay FP32.
    They hold the accumulated learning. Storing weights directly in a coarser format
    can round a small update away. No extra optimizer-owned master copy is required
    when the model parameters themselves remain FP32.

    **What arithmetic runs?** `autocast` selects dtypes for eligible operations.
    A matrix multiply can produce FP16 or BF16 output while the loss and parameter
    storage remain FP32. Leaving autocast before backward does not force every
    backward operation to FP32; backward follows its corresponding forward operation.

    **What scale survives backward?** FP16 has a narrower range than BF16, so scaling
    the loss can keep intermediate gradients from underflowing. Unscale before the
    optimizer uses them: $\nabla(SL)/S=\nabla L$ in exact arithmetic. This preserves
    the intended update; increasing the learning rate would change it. BF16 usually
    does not need scaling, but still rounds values and is not immune to failure.

    There are two different meanings of “scaling” here. **Token normalization** defines
    the objective. **AMP loss scaling** temporarily changes its numerical magnitude
    and is undone before the update. The second cannot repair an incorrect first.

    ### Watch one overflow skip and recovery

    This tiny **CPU FP16** example demonstrates the scaler's mechanism using the
    installed PyTorch. The second step deliberately injects an infinite gradient.
    Read both the scale and the weight movement: unchanged weights alone can also
    result from a zero gradient. CPU observations are not GPU correctness evidence.
    """)
    return


@app.cell
def _(F, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        _amp_model = torch.nn.Linear(3, 2)
    _amp_optimizer = torch.optim.SGD(_amp_model.parameters(), lr=0.1)
    _scaler = torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=100)
    _features = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.5]])
    _classes = torch.tensor([0, 1])
    amp_trace = []
    for _step in range(3):
        _amp_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.float16):
            _logits = _amp_model(_features)
            _loss = F.cross_entropy(_logits, _classes)
        _scaler.scale(_loss).backward()
        if _step == 1:
            _amp_model.weight.grad[0, 0] = float("inf")
        _before = _amp_model.weight.detach().clone()
        _scale_before = _scaler.get_scale()
        _scaler.step(_amp_optimizer)  # unscale, check, then step only if finite
        _scaler.update()
        amp_trace.append(
            {
                "step": _step + 1,
                "injected infinity": _step == 1,
                "scale before": _scale_before,
                "scale after": _scaler.get_scale(),
                "weights changed": not torch.equal(_before, _amp_model.weight),
            }
        )
    amp_dtypes = {
        "logits": str(_logits.dtype),
        "loss": str(_loss.dtype),
        "stored weight": str(_amp_model.weight.dtype),
        "stored gradient": str(_amp_model.weight.grad.dtype),
    }
    return amp_dtypes, amp_trace


@app.cell(hide_code=True)
def _(amp_dtypes, amp_trace, mo):
    _trace_rows = "\n".join(
        f"| {r['step']} | {r['injected infinity']} | "
        f"{r['scale before']:g} → {r['scale after']:g} | {r['weights changed']} |"
        for r in amp_trace
    )
    _dtype_rows = "\n".join(f"| {name} | {dtype} |" for name, dtype in amp_dtypes.items())
    mo.md(f"""
    | Step | Injected infinity | Scale before → after | Weights changed |
    | --- | --- | --- | --- |
    {_trace_rows}

    | Observed tensor | Dtype |
    | --- | --- |
    {_dtype_rows}

    The poisoned step leaves the weights unchanged and lowers the scale; the next
    finite step can update again. The scaler does not retry the skipped batch or
    repair an overflowed forward pass. Stored FP32 gradients do not imply that every
    intermediate backward value had FP32 precision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With accumulation, keep the AMP scale fixed throughout one effective batch.
    Unscale once at its boundary, then clip if needed, step and update the scaler.
    Mixing scaled and unscaled gradients combines different units. A scheduler meant
    to count successful optimizer updates must also respect skipped updates. The
    [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation)
    are the reference for these boundaries.

    For the AMP cases, compare equivalent layouts **using the same precision
    policy**. Separately compare against FP32 if that is the intended numerical-quality
    question. They need not share a tolerance. A test that loss is finite alone can
    miss skipped learning, incorrect normalization and a misplaced optimizer step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Keep the contribution useful and reviewable

    ### Learn from Marc's examples without copying their blind spots

    Both referenced PRs train real transformer models on GPUs, using a tiny Qwen2
    checkpoint. Loading a testing checkpoint does not establish useful pretrained
    capabilities. Our config-created current model removes asset loading from a test
    whose claim concerns training updates.

    - [TRL #4784](https://github.com/huggingface/trl/blob/512c3a96b951ddba40739b40caf03f0f580f5dcc/tests/distributed/test_distributed.py)
      runs SFT/DPO entry points with `trl-internal-testing/zen` on two GPUs. Its main
      assertion is successful execution. The streaming case is capped at three steps;
      not every case shares that cap. Author: Quentin Gallouédec; Marc linked it.
    - [Marc's Transformers #44338](https://github.com/huggingface/transformers/blob/27d935ee6e57491670c9523b5964ab47be8de80a/tests/trainer/distributed/test_trainer_distributed.py)
      checks training/AMP/accumulation completion, finite evaluation loss and resumed
      learning-rate logs. Common training uses32 synthetic examples. Resume scheduling
      is not full resumed-training equivalence.
    - Its [DDP token-averaging regression](https://github.com/huggingface/transformers/blob/27d935ee6e57491670c9523b5964ab47be8de80a/tests/trainer/distributed/test_trainer_distributed_ddp.py)
      uses50 Wikitext examples and ten updates, comparing logged losses under wrong
      and corrected normalization. It already protects the historical issue in Trainer.

    Successful execution is useful for a broad integration smoke test. Our narrower
    training contract justifies stronger observations of weights, precision, update
    boundaries and skip/recovery. That complements those examples; it does not make
    their tests worthless.

    ### Respect the component boundaries

    PyTorch owns tensor arithmetic, differentiation and DDP's underlying mechanism.
    Transformers owns its model and default loss. Accelerate owns how its API prepares
    and coordinates these components. We test an observable consequence of that
    composition. We do not reimplement or re-prove their internal mathematics.

    Accelerate already has regression-model synchronization/accumulation, plain versus
    prepared parameter checks, AMP overflow flags, scheduler tests and real BERT
    performance integration. Our real causal-LM comparison adds a specific workload
    and integration contract, not the first meaningful test in the repository.
    The toy normalization example in section3 is teaching. A caller's wrong loss is
    not automatically an Accelerate defect or a reason to create another PR.

    ### Fit the existing HF tooling and CI

    The file is already collected by Accelerate's core suite. Reuse its launch helper,
    local launch config, dependency/device guards and report conventions. A new GPU
    workflow is not required merely to register the test. The ordinary PR jobs run
    Python3.10 and generally skip a two-CUDA-device case; main/manual and nightly GPU
    jobs run `make test`. A green ordinary PR check is therefore not GPU evidence.
    The separate workflow that runs Transformers tests does not execute this file.
    See the pinned [Makefile](https://github.com/huggingface/accelerate/blob/b795b4838eb33bde60eb398649c4ab006874e3e9/Makefile),
    [PR workflow](https://github.com/huggingface/accelerate/blob/b795b4838eb33bde60eb398649c4ab006874e3e9/.github/workflows/test.yml)
    and [GPU workflow](https://github.com/huggingface/accelerate/blob/b795b4838eb33bde60eb398649c4ab006874e3e9/.github/workflows/run_merge_tests.yml).

    The repository declares Python>=3.10 and pins Ruff0.13.1 with a Python3.10 target.
    Its package minimum Torch is2.0, but its current minimum CI job installs2.5.1.
    The local untracked environment reads the repository's workflow and extras;
    it does not maintain another handwritten dependency list. The September 17 [multi-GPU nightly job](https://github.com/huggingface/accelerate/actions/runs/35173549475/job/105050178228)
    reported Torch 2.14.0 and Transformers 5.17.0, which includes Gemma 4. That checks
    model availability; this unsubmitted revision has not run on an HF runner. The
    existing nightly job failed elsewhere, so its status is not our test result.
    HF checkpoints/datasets are useful where loading/tokenization is part
    of the claim; they would add an unrelated network dependency to this test.

    ### What makes this easy to review?

    Keep the full required DDP/AMP/accumulation scope. One shared worker and one
    parametrized driver currently form a cohesive PR. Split only if each change has
    an independently useful architectural claim and the split lowers review cost.
    There is no target count. Current upstream integration is local; original branch
    and fork refs are preserved. Remote CI execution is separate from local preparation.

    The public PR explanation should stand on its own: state the behavior protected,
    fixture choice, assertions, actual validation and limits. Private discussions and
    local coordination history do not belong in that explanation. Keep the notebook
    and decision notes aligned with the tested revision when the code changes.

    Useful engineering principles, applied rather than attributed as endorsements:

    - **Parnas:** separate changeable launch/configuration details from the training
      behavior; assertions consume explicit results. No speculative backend framework.
      [Original modularization paper](https://www.cs.lafayette.edu/~gexia/cs301/resources/parnas.html).
    - **Fowler/Beck:** make tests useful bug detectors. Use a discriminating faulty path
      and the smallest scenario that protects the contract, rather than a matrix of
      every setting. [Self-Testing Code](https://martinfowler.com/bliki/SelfTestingCode.html).
    - **Challenge the reference:** an independent loop reduces shared-wrapper blind
      spots; it still shares the model and tensor library, whose correctness is not
      the responsibility this contribution claims.


    ### A reproducibility failure worth understanding

    An older Transformers version (5.14.1) registered Gemma 4's rotary-position buffers
    by iterating a Python set. Different processes can iterate that set in a different
    order. DDP broadcasts buffers in registration order, so equal-shaped buffers for
    full and sliding attention could silently exchange values on the second rank.
    Matching model weights alone would not detect this; our first-forward loss did.

    The [newer implementation sorts the order](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/gemma4/modeling_gemma4.py#L1088).
    Setting `PYTHONHASHSEED=0` **before launching workers** also makes the older fixture
    consistent. Setting it inside an already-running Python process is too late.
    Source inspection, CPU buffer-order checks and the corrected two-GPU run support
    this explanation. We control the fixture; we do not claim to fix a current
    Accelerate defect or weaken the assertions to accommodate the mismatch.

    ### What has actually been checked?

    All four cases passed on two RTX 4090s with Python 3.10 / Torch 2.5.1 /
    Transformers 5.17 and Python 3.14 / Torch 2.13 / Transformers 5.14.1.
    The suite took about 50–55 seconds per environment. Both ranks were observed.
    CPU and single-GPU execution skipped all four cases as intended, and repository
    quality checks passed. The three deliberate faults failed at their expected
    assertions. This is useful evidence for review, not universal numerical proof.

    The review lenses are concrete: **Parnas** asks whether responsibilities are clear;
    **Liskov** asks whether wrapping preserves the promised behavior; **Lamport** asks
    whether the statement covers every rank and the relevant ordering; **Beck** asks
    whether a small, understandable test catches a real mistake. These are applications
    of their ideas, not imagined endorsements. A review criterion earns its place by
    catching a defect or making the change easier to understand and maintain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sources and scope

    This companion condenses the original notebook's storage, autocast and scaler
    mechanisms, then connects them to distributed-update testing. Its two executable
    examples are teaching aids; the integration worker lives in Accelerate.

    - [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)
    - [HF: Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)
    - [Zach Mueller: Gradient accumulation, part 2](https://muellerzr.github.io/blog/gradient_accumulation_part2.html)
    - [Transformers distributed-test PR #44338](https://github.com/huggingface/transformers/pull/44338)
    - [TRL distributed-test PR #4784](https://github.com/huggingface/trl/pull/4784)

    Source and validation snapshot: September 17, 2026. Recheck implementation and CI
    versions when the contribution changes; the recorded evidence is dated rather than
    an assertion that external dependencies will stay unchanged.
    """)
    return


if __name__ == "__main__":
    app.run()
