# Training Fundamentals

*Part 1: the training loop. Part 2: mixed precision. Appendices: measurement, glossary.*

## Bird's-eye view

Training a neural network is a search. You have a model containing some quantity of adjustable numbers, a pile of examples, and a way of scoring how wrong the model currently is. You show it a handful of examples, measure the wrongness, work out which direction each adjustable number should move to make the wrongness smaller, and nudge every one of them a short distance in that direction. Then you do it again, a few hundred thousand times.

Everything below elaborates that sentence. The **forward pass** is *measure the wrongness*. The **backward pass** is *work out which direction*. The **optimizer** is *nudge them*. The memory arithmetic is bookkeeping for the fact that all those numbers have to physically live somewhere while it happens, and there are more of them than you would guess.

Every code block in this document shows **the same loop**, with modifications marked `# NEW`. That's the organising idea: there is one loop, and every technique you will learn — mixed precision, gradient accumulation, distributed training, sharding — is an intervention on one of its five lines.

---

# Part 1 — The Training Loop

## The loop

Every neural network training program, from a two-parameter toy to a frontier model, is the same eight lines:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch.inputs)          # forward
        loss = loss_fn(outputs, batch.labels)  # score
        loss.backward()                        # backward
        optimizer.step()                       # update
        optimizer.zero_grad()                  # reset
```

Everything else in modern training — mixed precision, gradient accumulation, **DDP** (Distributed Data Parallel), **FSDP** (Fully Sharded Data Parallel), **ZeRO** (Zero Redundancy Optimizer) — is a modification to one of those five lines. That's the useful frame: not "here is a pile of techniques" but "here is one loop, and here are the five places you can intervene."

Some vocabulary before we go further, because these words get used loosely and the looseness causes real confusion later:

A **parameter** (or *weight*) is a number the model learns. A 1-billion-parameter model has a billion of them. A **sample** is one training example. A **batch** (or *mini-batch*) is a group of samples processed together. A **step** (or *iteration*) is one execution of the loop body — one weight update. An **epoch** is one full pass over the dataset. Note that steps, not epochs, are the unit that matters mechanically; epochs are just bookkeeping.

**Convergence** means the loss has stopped meaningfully decreasing — the search has settled. "Does it converge?" is the weakest useful question you can ask of a training run, and correspondingly the most robust thing to assert in a test.

## The dataloader: where batches come from

The loop's first line hides a component that turns out to matter enormously, both for correctness and for everything distributed.

A **dataset** knows how to produce one sample by index. A **dataloader** wraps it and produces *batches*: it decides the order (shuffling), groups samples together, pads them to a common length and stacks them into a tensor (**collation**), and optionally does this on background worker processes so the GPU isn't waiting on the CPU.

Four properties of that innocuous-looking `for batch in dataloader` are worth knowing now, because each becomes a distributed-training problem later:

- **Order is a random variable.** Shuffling is seeded. Two runs agree only if the seed and the sampler agree — the foundation of every reproducibility test you'll write.
- **Position is state.** "Which batch am I on" lives inside the iterator, not on disk. A checkpoint that omits it will resume at the wrong place in the epoch and produce a subtly different loss curve.
- **Workers have their own randomness.** With `num_workers > 0`, each worker process has its own RNG (**random number generator**) seed. This is a common and confusing source of nondeterminism; `num_workers=0` removes it at some cost in speed.
- **Batches must be split across processes.** In distributed training each rank needs a *different* slice of the data, and the last batch of an epoch may not divide evenly among them. Handling that unevenness — pad, drop, or tolerate — is a real design decision with real correctness consequences.

That last point is why `dataloader` is one of the three things you hand to `accelerate`'s `prepare()`. The library replaces your loader with one that shards, seeds, and reassembles correctly. It is also, not coincidentally, the part of the library Marc described as the biggest rabbit hole.

## Forward: computing the prediction, and quietly recording how

The **forward pass** feeds the input through the network layer by layer and produces an output. For a language model: token IDs go in, a probability distribution over the vocabulary comes out for each position.

### What "layer by layer" actually means

Worth unpacking, because the specific operations determine almost everything that follows.

The token IDs first hit an **embedding table** — a large lookup matrix that maps each token ID to a vector of `hidden_size` numbers. From that point on, the data flowing through the network is a tensor of shape `(batch, sequence, hidden)`, and it keeps that shape all the way to the end.

That tensor then passes through a stack of identical **transformer blocks** — 32 of them in a 7B model, a couple in your test model. Each block does two things in sequence. Both are wrapped in a **residual connection** (add the block's input back to its output) and preceded by a **layer normalisation** (rescale the activations to roughly zero mean and unit variance, which keeps the numbers in a sane range across dozens of layers):

1. **Attention.** The input is multiplied by three learned weight matrices to produce **queries**, **keys** and **values**. Q is multiplied by Kᵀ to produce a `(sequence × sequence)` matrix of attention scores — how much each position should attend to each other position. **Softmax** turns each row of scores into weights summing to one. Those weights multiply V, and a fourth matrix projects the result back to `hidden_size`.

2. **MLP** (feed-forward). One matrix expands `hidden_size` by a factor of roughly four, an **activation function** is applied elementwise (GELU, SwiGLU — a nonlinearity, without which stacking layers would collapse into a single linear map), and a second matrix contracts back down.

After the final block, one more matrix — the **language-modelling head** — projects from `hidden_size` to vocabulary size, producing a score for every possible next token at every position.

Three things to take from that, each of which pays off later in this document:

**It is overwhelmingly matrix multiplies.** Count them: six large weight matrices per block — four in attention (Q, K, V and the output projection) and two in the MLP, or three if the MLP is *gated*, as SwiGLU is — plus two activation-times-activation matmuls inside attention itself (QKᵀ, and the weighted sum over V). Everything else is elementwise arithmetic and two normalisations. The matmuls dominate both the parameter count and the **FLOPs** (floating-point operations — the standard unit for counting arithmetic work). That is the entire reason mixed precision works: hardware that accelerates 16-bit matrix multiplication accelerates nearly all of the real work.

**The non-matmul operations are the numerically delicate ones.** Softmax and layer normalisation are *reductions*: they sum across a dimension. These are precisely the operations autocast keeps in fp32, and now the logic is visible — they're cheap enough that full precision costs almost nothing, and fragile enough that lowering them would hurt.

**Attention carries a term that scales with sequence *squared*.** That `(sequence × sequence)` score matrix exists per attention head, per layer. Everything else in the network scales linearly with sequence length; this one doesn't. It's why long-context training gets expensive so abruptly, and why the activation-memory estimate later needs a separate seq² term.

### The recording

That's the obvious job. The non-obvious job — and the one that dominates memory — is that PyTorch is simultaneously building a record of everything it did.

As each operation runs, PyTorch appends a node to a **computational graph**: a directed record of which tensors were produced by which operations from which inputs. Each output tensor carries a `grad_fn` attribute pointing at the operation that made it. This is **autograd** (automatic differentiation), and the graph is built dynamically, as the code executes, rather than declared in advance.

To later compute gradients, the graph must retain the intermediate tensors that the derivative formulas need. These retained intermediates are called **activations**. A matrix multiply `Y = X @ W`, for instance, must keep `X` around, because the gradient with respect to `W` depends on it.

This is the single most important memory fact in training: **activations are not a side effect, they are a stored dataset, and they scale with batch size and sequence length, not with model size.** It is why `torch.no_grad()` — which tells PyTorch not to build the graph — cuts inference memory so dramatically, and why gradient checkpointing (recomputing activations instead of storing them) exists at all.

## Loss: collapsing everything to one number

The **loss function** compares the model's output to the correct answer and returns a single scalar. For causal language modelling it's cross-entropy: roughly, "how surprised was the model by the token that actually came next," averaged over every position in the batch.

It must be a single number, because the gradient we're about to compute is *the derivative of that number with respect to every parameter*. Derivatives need a scalar output. This sounds like a technicality; it isn't. The fact that the loss is an *average* over tokens is precisely the detail that makes gradient accumulation subtly wrong when sequences have different lengths — the bug Zach Mueller and Marc Sun chased in `transformers`. Averages of averages aren't averages.

## Backward: the chain rule, run in reverse

`loss.backward()` walks the computational graph from the loss back to the inputs, applying the chain rule at each node. This is **backpropagation** — mathematically, *reverse-mode* automatic differentiation, meaning it starts from the single output and works back toward the many inputs, which is the efficient direction when you have one loss and billions of parameters.

The output is a **gradient** for every parameter: a number saying "if you nudge this weight up slightly, the loss changes by this much, in this direction." Gradients have exactly the same shape and count as the parameters. A billion parameters means a billion gradients.

### What backward actually computes

Backward does not interleave with forward. It is a separate, complete traversal that begins only once the loss scalar exists. What the forward pass does is *determine* it — building the graph decides what backward will do — but nothing executes until `loss.backward()` is called.

Every forward operation has a paired backward operation. The matmul case is the one worth carrying in your head, because it explains three separate things at once.

Forward computed `Y = X @ W`. Backward receives `dY` — the gradient of the loss with respect to that operation's output — and produces **two** results:

- `dX = dY @ Wᵀ` — passed further back, to become the previous layer's `dY`
- `dW = Xᵀ @ dY` — the gradient we actually want, written into `W.grad`

Two matrix multiplies out for every one in. From that:

**Cost.** Backward is roughly twice the arithmetic of forward, so a full training step costs about three forward passes. That ratio is the origin of the `6ND` compute rule, worked through in *Compute: where the time goes* below.

**Why activations must be stored — concretely.** Above, the claim was that the graph retains "the intermediates the derivative formulas need." `dW = Xᵀ @ dY` is the specific reason. The formula literally requires `X`, the forward input, and `X` is gone unless something kept it. That single equation is the whole of activation memory.

**Where peak memory occurs.** Activations are freed as backward consumes them, layer by layer. So the high-water mark sits at the forward/backward boundary — everything stored, nothing yet released. Worth knowing when interpreting `max_memory_allocated()`.

Two smaller points round out the picture. **Residual connections** matter here because addition passes the incoming gradient to *both* branches unchanged — that's the mechanism by which gradient reaches early layers without attenuating. And **softmax and layer normalisation** have backward passes that are genuinely more involved than their forwards (softmax's **Jacobian** — the matrix of every output's derivative with respect to every input — isn't diagonal, so each output's gradient depends on all the others), which is a second reason, beyond numerical fragility, that autocast leaves them in fp32.

### Where the gradients land

PyTorch writes each gradient into the parameter's `.grad` attribute. One detail with outsized consequences: it **accumulates** rather than overwrites — `.grad += new_gradient`, not `.grad = new_gradient`.

This default trips up everyone once, and it is also the entire mechanism behind gradient accumulation. If you run forward and backward several times before stepping the optimizer, the gradients sum, and you get the effect of a larger batch without the memory cost of one. The feature and the footgun are the same line of code.

## Zeroing: why you must clean up

Because gradients accumulate, they must be explicitly cleared, or step 2 would update using the sum of steps 1 and 2. `optimizer.zero_grad()` does this. Modern PyTorch defaults to `set_to_none=True`, which frees the gradient tensors entirely rather than filling them with zeros — slightly faster, and it releases memory between steps.

## The optimizer: turning gradients into an update

The **optimizer** decides how to change the weights given the gradients. The simplest rule, stochastic gradient descent, is one line: `param -= learning_rate * gradient`. The **learning rate** is the step size — the single most important **hyperparameter**, meaning a value you choose rather than one the model learns.

Almost nobody uses plain SGD for transformers. The standard is **AdamW**, which adapts the step size *per parameter* based on the recent history of that parameter's gradients. To do this it maintains, for every single parameter, two extra numbers:

- the **first moment** (`m`): an exponential moving average of the gradient — effectively momentum, smoothing out noise between batches.
- the **second moment** (`v`): an exponential moving average of the *squared* gradient — a running estimate of how volatile this parameter's gradient has been.

The update, stripped of detail, is:

```
m = β₁·m + (1-β₁)·grad           # smoothed gradient        (β₁ ≈ 0.9)
v = β₂·v + (1-β₂)·grad²          # smoothed squared gradient (β₂ ≈ 0.999)
param -= lr · m / (√v + ε)       # step, scaled per parameter
```

Read the last line as the whole idea: divide the smoothed gradient by its own recent magnitude. A parameter with consistently large gradients gets a proportionally *smaller* step; a parameter with tiny gradients gets a relatively larger one. The step size becomes roughly scale-invariant, which is why Adam works across the wildly different gradient magnitudes found in different layers of a transformer without per-layer tuning.

Two details in that formula that trip people up. **`ε`** (epsilon, typically 10⁻⁸) exists purely to stop division by zero when `v` is very small — but it also sets a floor on how large a step a tiny-gradient parameter can take, so it isn't quite as inert as it looks. And because `m` and `v` both start at zero, they are biased toward zero for the first several steps; real implementations apply a **bias correction** that divides them by `(1 - βⁿ)` at step `n`, which matters most in exactly the early iterations where training is most fragile.

The "W" in AdamW refers to *decoupled weight decay*. **Weight decay** is a regularisation term — a small pull of every weight toward zero, discouraging the model from relying too heavily on any single parameter. AdamW applies it directly to the weights rather than folding it into the gradient, which interacts better with the adaptive step. It matters for training quality but not for your mental model of the loop.

### Optimizer states are the hidden memory cost

`m` and `v` are the **optimizer states**, and they are the main reason training costs so much more memory than inference. Precisely: in fp32, parameters and gradients together cost 8 bytes per parameter, and Adam's two states add another 8 — so **Adam doubles the footprint**, or equivalently, the optimizer alone costs twice what the parameters do.

Different optimizers make different bargains, and this is the cheapest memory lever available:

| Optimizer | Optimizer state | fp32 bytes/param |
|---|---|---|
| SGD, no momentum | none | 0 |
| SGD with momentum | `m` | 4 |
| **AdamW** | `m`, `v` | **8** |
| 8-bit AdamW (bitsandbytes) | `m`, `v` quantised | 2 |

One practical consequence worth internalising, because it produces a genuinely confusing failure: **PyTorch allocates `m` and `v` lazily, on the first `optimizer.step()`.** They do not exist during the first forward and backward. This is the reason for the classic report *"step 1 ran fine, step 2 hit an out-of-memory error"* — the optimizer states materialised in between. If you profile only the first iteration, you will under-count memory by 8 bytes per parameter.

Two companions usually sit alongside the optimizer. A **learning-rate scheduler** changes the learning rate over the course of training — typically a short *warmup* from near zero (large early steps on a random model are destabilising) followed by a slow decay. And **gradient clipping** (`clip_grad_norm_`) rescales the whole gradient vector if its magnitude exceeds a threshold, which prevents one pathological batch from destroying the run.

With both companions in place, the loop is still the same loop:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch.inputs)                     # forward
        loss = loss_fn(outputs, batch.labels)             # score
        loss.backward()                                   # backward
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # NEW — clip
        optimizer.step()                                  # update
        scheduler.step()                                  # NEW — advance LR
        optimizer.zero_grad()                             # reset
```

## What the loop leaves out

Three things belong in your mental model even though they're not in the eight lines.

**Mode switching.** `model.train()` and `model.eval()` toggle layers that behave differently during training and inference — dropout is active in one and disabled in the other. Evaluation should additionally run inside `torch.no_grad()`, which skips graph construction entirely and therefore skips storing activations.

**Determinism.** Two identical runs will not produce identical numbers unless you seed every source of randomness (weight initialisation, dropout, data shuffling) and constrain nondeterministic GPU kernels. This is normally a nicety. For integration testing it's load-bearing: every test asserting that two runs agree depends on it.

**The full training state.** Newcomers think a checkpoint is the model weights. It is not. Resuming exactly where you left off requires the weights, the optimizer states (`m` and `v` — throw them away and the first steps after resume will be visibly wrong), the scheduler position, the RNG states, and the dataloader's position in the epoch. That list is precisely why checkpoint/resume is the most interesting thing to test in a distributed training library: it's the piece with the most ways to be quietly incomplete.

## Memory: where it all goes

For a model with **P** parameters trained in fp32 with AdamW, the per-parameter cost is fixed and easy to compute:

| Bucket | Bytes per parameter | For a 1B model |
|---|---|---|
| Parameters | 4 | 4 GB |
| Gradients | 4 | 4 GB |
| Adam first moment | 4 | 4 GB |
| Adam second moment | 4 | 4 GB |
| **Model states total** | **16** | **16 GB** |

That's the origin of the well-known rule of thumb: **16 bytes per parameter before you've stored a single activation.** It's the number to have memorised, because it tells you instantly that a 7B model cannot be trained on a 24 GB card without changing something structural — and every technique you'll be testing (ZeRO, FSDP) is a way of changing that something. They shard these four buckets across **ranks** (a *rank* is one process in a distributed job, usually one per GPU) so that no single GPU holds all of them.

On top of the model states sit **activations**, which behave completely differently. They scale with batch size × sequence length × model width × number of layers, with an additional term from attention that grows with the *square* of sequence length. Model states are fixed once you've chosen a model; activations are what you actually control at runtime, and they're usually what's moving when you hit an **OOM** (out-of-memory error).

Then there's a floor of overhead people forget: the CUDA context itself costs roughly 0.3–0.6 GB per process before you allocate anything, cuBLAS and cuDNN (NVIDIA's linear-algebra and neural-network kernel libraries) keep scratch workspaces, NCCL (NVIDIA's collective-communication library, used for multi-GPU synchronisation) allocates communication buffers in distributed runs, and the optimizer step creates transient temporaries.

### The same 16 bytes, arranged two different ways

The table above assumes pure fp32. Once mixed precision is involved, the total stays near 16 bytes per parameter but the *arrangement* changes — and the two dominant frameworks arrange it as near-mirror-images of each other:

| | PyTorch AMP | DeepSpeed ZeRO / Megatron |
|---|---|---|
| Parameters | fp32 (4) | half precision (2) |
| Gradients | fp32 (4) | half precision (2) |
| Optimizer | `m`, `v` in fp32 (8) | fp32 master copy + `m` + `v` (12) |
| Half-precision copies | transient, during forward | — (the model already is) |

PyTorch holds the model in fp32 and manufactures half-precision copies on the fly. DeepSpeed holds the model in half precision and keeps an fp32 master copy *inside the optimizer*. Same destination, opposite routes.

This matters more than it looks. The 12-byte optimizer chunk in the right-hand column is precisely what **ZeRO stage 1** shards across ranks — it's the largest single bucket, which is why it's the first thing the algorithm goes after, with gradients (stage 2) and parameters (stage 3) following. When you later compare memory across DDP, FSDP and DeepSpeed and the numbers don't line up neatly, this table is usually the reason.

### Memory over the course of a single step

The buckets above are a static budget. Within one iteration, usage moves in a characteristic shape worth recognising on a profile:

1. **Forward** — activations accumulate rapidly, layer by layer. Memory climbs.
2. **Backward** — gradients fill in while stored activations are progressively released as each layer's gradient is computed. The curve peaks near the start of backward and then declines.
3. **Optimizer step** — all gradients must be live simultaneously, and the optimizer states are read and written.

And the **first iteration does not look like the others**. Two reasons compound: the caching allocator is still learning your allocation pattern and doing extra bookkeeping, and — as above — `m` and `v` don't exist yet. This is why the honest answer to "how much memory does this configuration need" comes from a *steady-state* step, not the first one.

For how to check a prediction against reality without being misled by the allocator, see the [measurement appendix](#appendix--measuring-correctly).

## Compute: where the time goes

Memory answers "will it fit." The companion question is "how fast will it go," and it has an equally compact rule.

A forward pass costs roughly `2 · N · D` FLOPs, where **N** is the parameter count and **D** the number of tokens processed — the 2 being one multiply and one add per parameter per token. Backward costs about twice that. So a full training step is:

```
C ≈ 6 · N · D
```

That's the compute counterpart to `16 bytes/param`, and between the two you can size both halves of a training run from the parameter count alone.

The natural follow-up is: how much of the hardware's capability are you actually using? That's **MFU** (Model FLOPs Utilization) — the ratio of the FLOPs your model *theoretically requires* to the hardware's peak FLOPs over the same wall-clock time. Introduced in the PaLM paper for exactly this purpose: a hardware- and implementation-independent efficiency number that can be compared across systems. Well-tuned large-scale training lands somewhere in the 30–60% range; below that, something is wrong, and the something is usually data loading, communication, or too-small kernels rather than arithmetic.

Its sibling **HFU** (Hardware FLOPs Utilization) counts the FLOPs *actually executed*, including wasted or repeated work. The two diverge exactly when you use gradient checkpointing: recomputing the forward pass raises HFU (more real arithmetic) while leaving MFU unchanged (the model's theoretical requirement didn't move). MFU is the more honest number, because the goal is tokens per second, not keeping the arithmetic units busy.

You will not need MFU for the tiny test models in this project — at that scale you're dominated by Python and kernel-launch overhead, and MFU will be embarrassingly low for reasons that say nothing about the code. It's here so the number means something when you meet it in the playbook.

---

# Part 2 — Mixed Precision

## Where it plugs in

Mixed precision is an intervention on the forward pass — and, if you use fp16, on the backward and the update as well.

**bf16 — the whole change is two lines of indentation:**

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # NEW
            outputs = model(batch.inputs)          # forward
            loss = loss_fn(outputs, batch.labels)  # score
        loss.backward()                            # backward
        optimizer.step()                           # update
        optimizer.zero_grad()                      # reset
```

Note that `backward()` sits **outside** the autocast block. Autocast governs the forward pass; the backward pass automatically uses the dtypes recorded in the graph during forward.

The fp16 case needs three more moving parts, and understanding *why* is most of what there is to learn here.

## Why it was invented

The honest answer is: **for speed, with memory as a bonus.** The technique came out of NVIDIA and Baidu in 2017 (Micikevicius et al.), and it arrived at the same moment as the hardware that made it worthwhile.

A **Tensor Core** is a specialised hardware unit, introduced with the Volta generation, that performs a small matrix multiply-accumulate in one instruction: it takes 16-bit inputs, multiplies them, and accumulates the result in 32-bit. It's dramatically faster than doing the same work on the general-purpose CUDA cores — the peak-throughput ratio is several-fold, depending on generation.

So the pitch is: transformers are overwhelmingly matrix multiplies, there is hardware that does 16-bit matrix multiplies several times faster, therefore run the matrix multiplies in 16 bits. The memory saving on activations came along for the ride, and these days people often care about it more.

The catch is that 16 bits is not enough for everything, and the interesting engineering is in figuring out which parts break and how to prop them up.

## The formats: range versus precision

A floating-point number is stored as three fields. The **sign** bit. The **exponent**, which determines the *range* — how large and how small a number can be. And the **mantissa** (also called the significand), which determines the *precision* — how many distinct values you can express within that range.

| | Sign | Exponent | Mantissa | Max value | Smallest normal | Decimal digits |
|---|---|---|---|---|---|---|
| **fp32** | 1 | 8 | 23 | ~3.4 × 10³⁸ | ~1.2 × 10⁻³⁸ | ~7 |
| **fp16** | 1 | 5 | 10 | 65,504 | ~6.1 × 10⁻⁵ | ~3 |
| **bf16** | 1 | 8 | 7 | ~3.4 × 10³⁸ | ~1.2 × 10⁻³⁸ | ~2 |

Read that table as a story about two different bets on how to spend 16 bits.

**fp16** (half precision, an IEEE standard) keeps more mantissa and sacrifices exponent. It's more *precise* than bf16 but its range is tiny — it cannot represent anything above 65,504 or below about 6 × 10⁻⁵ without special handling.

**bf16** (bfloat16, "brain float," from Google) does the opposite: it keeps fp32's exponent field exactly and throws away mantissa bits. It is *less precise* than fp16 but has the identical range to fp32.

That single design choice — bf16 matching fp32's exponent — is why bf16 has essentially won for training. Converting fp32 → bf16 can never overflow or underflow, because the representable range is the same. It just gets rounder. Converting fp32 → fp16 can, and does, fall off both ends.

## Why "mixed" and not just "16-bit"

Some operations are numerically robust in 16 bits and some are not. The distinction is roughly: operations whose error stays bounded versus operations that *accumulate* error across many terms.

- **Safe in 16-bit:** matrix multiplies, convolutions, linear layers. These are also, conveniently, the expensive ones. Even here the hardware accumulates in fp32 internally.
- **Not safe:** reductions — sums, means, softmax, layer normalisation, loss functions. Adding thousands of small numbers in low precision loses the small ones entirely. Also exponentials and logarithms, which can leave the representable range.

`torch.autocast` implements this as a policy. It maintains a list of operations that get their inputs cast down, a list that is forced to fp32, and a list that promotes to the widest input type. You don't manage it; the context manager does. Within an autocast region it also caches weight casts, so a weight used twice is converted once.

Crucially, **autocast does not touch your parameters.** They remain fp32 in memory. What changes is the dtype flowing through eligible operations, and therefore the dtype of the activations that get saved for backward. Hold onto that — it's the key to the memory arithmetic below.

## The underflow problem, and loss scaling

Here is the failure mode that makes fp16 hard, and it's worth internalising because it explains an entire apparatus.

Gradients are small. In a converging transformer, a great many gradient values live around 10⁻⁷ to 10⁻⁹. fp16's smallest normal value is about 6 × 10⁻⁵. Below that it can represent **subnormal** numbers (a degraded-precision tail extending to ~6 × 10⁻⁸), and below *that*, zero.

So a large fraction of your gradients simply become zero. Not small — gone. The parameters they belong to stop learning. The model quietly fails to converge, with no error message.

The fix is elegant: **loss scaling.** Multiply the loss by a large constant *S* before calling backward. By the chain rule, every gradient in the network is then scaled by exactly *S* too, lifting the whole distribution up into fp16's representable range. Before the optimizer step, divide the gradients by *S* to recover the true values. The update is mathematically identical; the numbers just took a detour through a range where they survive.

Choosing *S* is the problem. Too small and you still underflow; too large and gradients overflow to infinity. And the right value drifts over training as gradient magnitudes change.

## GradScaler: dynamic loss scaling

PyTorch's `GradScaler` solves this adaptively. Same loop, four modified lines:

```python
scaler = torch.amp.GradScaler("cuda")   # NEW — created once, before the loop

for epoch in range(num_epochs):
    for batch in dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # NEW
            outputs = model(batch.inputs)          # forward
            loss = loss_fn(outputs, batch.labels)  # score
        scaler.scale(loss).backward()              # backward — loss × S, then backward
        scaler.step(optimizer)                     # update   — unscale, check, step or skip
        scaler.update()                            # NEW      — adjust S for next iteration
        optimizer.zero_grad()                      # reset
```

The algorithm is "start optimistic, back off on failure, creep back up":

- Begin at *S* = 65,536 (2¹⁶).
- After backward, inspect all gradients. If any is infinite or NaN, the scale was too high — **skip the optimizer step entirely** and multiply *S* by 0.5.
- If 2,000 consecutive iterations pass without overflow, multiply *S* by 2 and keep going.

Two consequences that matter for testing:

**Steps get skipped.** In the early iterations especially, `scaler.step()` may do nothing at all while the scaler hunts for a workable scale. This means *loop iterations no longer correspond one-to-one with optimizer updates.* Any test asserting a specific loss after N iterations, or comparing an fp16 run against an fp32 run step-for-step, has to account for this. It's a common source of confusing test failures.

**The scaler has state.** *S* and the overflow counter are part of your training state. Checkpoint them, or a resumed run restarts the search from 65,536 and behaves differently from an uninterrupted one — a real and easy-to-miss resume bug.

**bf16 needs none of this.** Its range is fp32's, so gradients don't underflow and loss scaling is unnecessary. `GradScaler` is simply not used. Fewer moving parts, no skipped steps, no extra state to checkpoint. This is a large part of why bf16 is the default recommendation on any hardware that supports it — including the RTX 4090.

One sharp edge worth naming: a model pretrained in bf16 often *cannot* be fine-tuned in fp16. Its weights and activations contain values outside fp16's range, so you get overflow rather than underflow, and the scaler drives *S* below 1 trying to compensate. If you see NaNs immediately under fp16, this is usually why.

## Master weights: the part frameworks disagree about

The original 2017 paper described keeping an **fp32 master copy** of the weights: compute in fp16, but store the authoritative parameters in fp32 and apply updates there.

The reason is a second, subtler precision failure. The update is `param -= lr * grad`. Late in training, `lr * grad` can be several orders of magnitude smaller than `param`. Adding a tiny number to a large one in low precision rounds straight back to the large one — the update is *swallowed*. fp32 master weights preserve those small accumulated updates.

**PyTorch AMP sidesteps this entirely by never storing weights in low precision.** Parameters are fp32, gradients accumulate into fp32 `.grad`, the optimizer works in fp32. Only the transient computation is 16-bit. There is no separate master copy because the working copy *is* the master copy.

**Other backends do it differently.** FSDP's mixed-precision policy has separate `param_dtype`, `reduce_dtype`, and `buffer_dtype` settings, and can genuinely hold sharded parameters in bf16 while keeping an fp32 copy for the update. DeepSpeed's fp16/bf16 modes maintain fp32 master weights inside the optimizer partition. So "mixed precision" denotes meaningfully different memory layouts and different numerics depending on which backend you're in.

That's worth flagging in a write-up, because it's precisely the kind of thing an integration test across DDP, FSDP and DeepSpeed exists to catch — and it's a good reason not to expect the three backends' loss curves to match to tight tolerance under mixed precision.

## Memory: what actually changes

Return to the Part 1 table, for a model with **P** parameters and AdamW, and mark what plain PyTorch AMP does to each row:

| Bucket | fp32 baseline | Under `autocast` |
|---|---|---|
| Parameters | 4 bytes/param | **4 bytes/param — unchanged** |
| Gradients | 4 bytes/param | **4 bytes/param — unchanged** |
| Adam first moment | 4 bytes/param | **unchanged** |
| Adam second moment | 4 bytes/param | **unchanged** |
| **Model states** | **16 bytes/param** | **16 bytes/param** |
| Activations | full width | **roughly halved** for autocast-eligible ops |
| Transient weight casts | — | **added** (16-bit copies during forward) |

So the honest summary is: **model states don't move, activations shrink, and a new transient cost appears.**

Whether total memory goes up or down therefore depends on which term dominates:

- **Activation-dominated** (large batch, long sequences, modest model) → memory goes **down**, sometimes a lot.
- **Parameter-dominated** (small batch, large model) → the halved activations don't offset the cast copies, and memory goes **up**.

The classic textbook framing is "up," and that's right for the parameter-dominated regime. For a tiny test model with a reasonable batch, you may well measure "down." Either result is a good result — the point of the exercise is being able to say *which regime you're in and why* before you run it.

## What to predict, and how to measure it

Three predictions, in descending order of confidence:

**Loss curve: essentially unchanged.** bf16's reduced mantissa introduces small drift, so expect close agreement rather than bitwise equality. If the loss diverges materially, something is wrong.

**Throughput: faster — but only if you're matmul-bound.** On a tiny test model the speedup may be small or absent, because you're dominated by kernel launch overhead and Python, not arithmetic. Tensor Core eligibility also depends on tensor dimensions being multiples of 8 (fp16) or 16; odd shapes silently fall back to the slow path. If your measured speedup is disappointing, check the shapes before concluding the theory is wrong.

**Memory: work out the regime first, then commit to a direction.** Compute the model-states figure from `16 × P`, estimate activations from batch × sequence × width × layers, see which dominates, and write down your prediction *before* running.

Measure with the method in the [appendix](#appendix--measuring-correctly) — the allocator will mislead you otherwise, and an unsynchronised timer will measure Python rather than the GPU.

---

## Coda: what accelerate does to the loop

In `accelerate`, all of the above collapses back into the canonical form:

```python
accelerator = Accelerator(mixed_precision="bf16")    # NEW — the only config
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch.inputs)          # forward
        loss = loss_fn(outputs, batch.labels)  # score
        accelerator.backward(loss)             # backward — was loss.backward()
        optimizer.step()                       # update
        optimizer.zero_grad()                  # reset
```

The library selects autocast, wires up a `GradScaler` if and only if you chose fp16, and handles the scaler's checkpoint state. One changed line in the loop body, one changed line at setup.

That is the whole value proposition of the library, and also the reason integration tests matter: the loop looks identical whether you are running fp32 on one GPU or bf16 across sharded ranks, so nothing in the loop's appearance tells you whether the machinery underneath is correct.

---

# Appendix — Measuring correctly

Applies to every experiment in this document and every topic after it. The naive measurement will mislead you in two different ways, one for memory and one for time.

## Memory

**PyTorch does not return freed memory to the driver.** It keeps a **caching allocator** that holds onto blocks for reuse, because asking the driver for memory is slow. So there are two distinct numbers:

- `torch.cuda.memory_allocated()` — bytes currently held by live tensors. This is what your model actually needs.
- `torch.cuda.memory_reserved()` — bytes the allocator has claimed from the GPU, including free blocks it is hanging onto for later.

`nvidia-smi` reports something closer to the second, plus the CUDA context. It will always overstate your true usage, sometimes by a wide margin. It is the right tool for "am I about to OOM" and the wrong tool for "does my prediction hold."

For predict-versus-measure work you want the **peak live-tensor high-water mark**, because that's the quantity the arithmetic predicts:

```python
torch.cuda.reset_peak_memory_stats()
# ... run some training steps ...
peak_bytes = torch.cuda.max_memory_allocated()
```

Reset before *each* run, or you'll carry the previous configuration's peak into the next measurement.

Two refinements worth knowing. Run a few warmup steps before resetting, so one-time allocations (cuDNN autotuning, the first optimizer step materialising Adam's `m` and `v`) don't contaminate the steady-state number. And `torch.cuda.memory_summary()` prints a full breakdown when a measurement is surprising and you need to find out why.

## Time

**CUDA kernels launch asynchronously.** (A *kernel* is a single function executed on the GPU — a matmul, a softmax, an elementwise add.) Python queues work on the GPU and returns immediately. An unsynchronised timer therefore measures how fast Python can enqueue, not how fast the GPU computes — which is why naive benchmarks often report implausibly fast results.

```python
torch.cuda.synchronize()
start = time.perf_counter()
# ... run some training steps ...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Both synchronisations are required. Warm up first — the first iterations include CUDA context setup, memory allocation, and kernel autotuning, and are not representative. Then time a fixed number of steps and report per-step time or tokens/second rather than total, so runs with different step counts stay comparable.

## In distributed runs

Each rank has its own allocator and its own peak. Record per-rank figures and report the maximum, not rank 0's — with an uneven shard split, rank 0 is not necessarily the worst case. And synchronise across ranks (`accelerator.wait_for_everyone()`, or `dist.barrier()`) before starting a timer, or you will measure stragglers rather than throughput.

---

# Appendix — Glossary

Terms are defined where they first appear; this is a lookup table, not a substitute for reading.

| Term | Definition |
|---|---|
| **Activations** | Intermediate tensors saved during forward because the backward pass needs them. Scale with batch and sequence length, not model size. |
| **AdamW** | The standard optimizer for transformers. Adaptive per-parameter step sizes, plus decoupled weight decay. |
| **autocast** | PyTorch's mixed-precision context manager. Casts eligible operations to 16-bit; leaves fragile ones in fp32. |
| **Autograd** | PyTorch's automatic differentiation engine. Records operations during forward, replays them in reverse during backward. |
| **bf16** (bfloat16) | 16-bit float with fp32's exponent range and reduced precision. The default choice for training. |
| **Caching allocator** | PyTorch's memory manager. Holds freed blocks for reuse rather than returning them to the driver. |
| **Collation** | Assembling individual samples into a batched tensor, including padding to a common length. |
| **Convergence** | The point at which the loss stops meaningfully decreasing. |
| **DDP** (Distributed Data Parallel) | Replicate the model on every GPU; average gradients across them each step. |
| **Epoch** | One full pass over the dataset. |
| **FLOPs** | Floating-point operations. The unit for counting arithmetic work. |
| **fp16** (half precision) | 16-bit float with reduced range. Requires loss scaling to train safely. |
| **FSDP** (Fully Sharded Data Parallel) | PyTorch's native sharding of parameters, gradients and optimizer states across ranks. |
| **Gradient** | The derivative of the loss with respect to a parameter. Same shape and count as the parameters. |
| **Gradient checkpointing** (activation recomputation) | Discard activations during forward and recompute them during backward. Trades roughly one extra forward pass for a large memory saving. |
| **GradScaler** | PyTorch's dynamic loss-scaling helper. Only needed for fp16. |
| **HFU** (Hardware FLOPs Utilization) | Fraction of peak FLOPs actually executed, including repeated work. |
| **Hyperparameter** | A value you choose (learning rate, batch size) rather than one the model learns. |
| **Kernel** | A single function executed on the GPU. |
| **Loss scaling** | Multiplying the loss by a constant before backward so small gradients survive fp16, then dividing it out before the update. |
| **Master weights** | An fp32 copy of the parameters kept for the optimizer update when the working copy is in half precision. |
| **MFU** (Model FLOPs Utilization) | Observed throughput relative to theoretical peak. The implementation-independent efficiency metric. |
| **NCCL** | NVIDIA's collective-communication library; the transport for multi-GPU synchronisation. |
| **OOM** | Out-of-memory error. |
| **Optimizer states** | The extra per-parameter values an optimizer maintains. For AdamW: `m` and `v`, 8 bytes/param in fp32. |
| **Parameter** (weight) | A number the model learns. |
| **Rank** | One process in a distributed job, usually one per GPU. |
| **Reduction** | An operation that sums or averages across a dimension (softmax, layer norm, mean). Numerically fragile in low precision. |
| **Residual connection** | Adding a block's input to its output. Passes gradient backward unchanged. |
| **Step** (iteration) | One execution of the loop body; one weight update. |
| **Tensor Core** | Specialised GPU hardware for 16-bit matrix multiply with 32-bit accumulate. |
| **Weight decay** | Regularisation that pulls weights toward zero. |
| **ZeRO** (Zero Redundancy Optimizer) | DeepSpeed's sharding scheme. Stage 1 shards optimizer states, stage 2 adds gradients, stage 3 adds parameters. |
