"""The theory side of the comparison: memory arithmetic done on paper.

The split here is deliberate.

**Exactly predictable** — parameters, gradients, optimizer state, and the autocast
weight cache. These follow from the config and a dtype policy, so the harness checks
them to the byte.

**Measured, not predicted** — activations, and transient peaks inside backward and
inside the optimizer step. Those depend on kernel implementations (`foreach` versus
`fused` AdamW, flash versus math attention) and on which tensors a library upcasts.
Predicting them analytically would turn a check into a curve fit. `l2c.harness.ledger`
measures them instead, and the interesting work is explaining the gap.

What is deliberately *not* here: deriving the parameter count from the config. That
derivation is the exercise. Write it by hand in the step's NOTES.md and prediction.toml,
and the harness checks it against `sum(p.numel())`.

In accelerate: there is no predictive counterpart. Mixed precision is configured
declaratively — `Accelerator(mixed_precision="bf16")` — and what that *costs* is left
implicit. The reason model states do not shrink is visible only in
`accelerate/accelerator.py` around line 1820, where `_prepare_model` wraps the
model's `forward` in an autocast context rather than converting any parameter:

    model.forward = convert_outputs_to_fp32(autocast_context(model_forward_func))

Parameters stay in fp32 and autocast casts op *inputs* on the fly. Predicting that
`mixed_precision="bf16"` halves your optimizer state is a prediction that should fail,
and watching it fail is the point of step 2.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

BYTES_PER_ELEMENT = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
}

# Extra tensors the optimizer keeps per parameter.
#   sgd            : 0
#   sgd + momentum : 1  (momentum_buffer)
#   adamw          : 2  (exp_avg, exp_avg_sq)
OPTIMIZER_STATES_PER_PARAM = {"sgd": 0, "sgd_momentum": 1, "adamw": 2}


@dataclass(frozen=True, slots=True)
class ModelStates:
    """The buckets that scale with parameter count and nothing else.

    Fixed once a model and optimizer are chosen — they do not move with batch size or
    sequence length, which is exactly why they can be predicted exactly while
    activations cannot.
    """

    num_params: int
    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes

    @property
    def bytes_per_param(self) -> float:
        return self.total_bytes / self.num_params


def model_states(
    num_params: int,
    *,
    param_dtype: torch.dtype = torch.float32,
    optimizer: str = "adamw",
    optimizer_state_dtype: torch.dtype = torch.float32,
) -> ModelStates:
    """fp32 params + fp32 grads + AdamW's two fp32 moments = 4 + 4 + 8 = 16 B/param.

    `torch.autocast` changes none of this. Parameters stay in `param_dtype`, gradients
    match them, and the optimizer's moments follow the parameters. What autocast adds
    is a separate cast cache — see `autocast_weight_cache`.
    """
    param_bytes = num_params * BYTES_PER_ELEMENT[param_dtype]
    grad_bytes = num_params * BYTES_PER_ELEMENT[param_dtype]
    n_states = OPTIMIZER_STATES_PER_PARAM[optimizer]
    optimizer_bytes = num_params * n_states * BYTES_PER_ELEMENT[optimizer_state_dtype]
    return ModelStates(num_params, param_bytes, grad_bytes, optimizer_bytes)


def autocast_eligible_weights(model: nn.Module) -> list[nn.Parameter]:
    """Every distinct `nn.Linear` weight, deduplicated by storage.

    autocast's dtype policy runs matmul-family ops in the reduced dtype and leaves
    normalizations and reductions in fp32. For a Llama-style decoder that means every
    projection weight is a cast target and every RMSNorm weight is not.

    Deduplication matters for a tied embedding. `lm_head` is an `nn.Linear` whose
    weight *is* the embedding table, so it must be counted exactly once: the
    `lm_head` matmul is a cast target, while the embedding *lookup* is not an
    autocast-eligible op at all. The same storage, used two ways, cast once.
    """
    seen: dict[int, nn.Parameter] = {}
    for module in model.modules():
        if isinstance(module, nn.Linear):
            seen.setdefault(module.weight.untyped_storage().data_ptr(), module.weight)
    return list(seen.values())


@dataclass(frozen=True, slots=True)
class WeightCache:
    """The reduced-precision copies autocast holds alongside the fp32 masters.

    Both the byte total and the tensor count are predicted. The count is what makes
    the ledger's classification self-checking: if the measured number of cast tensors
    equals the number of `nn.Linear` weights, the classification is almost certainly
    reading the right tensors, and the byte comparison means something.
    """

    num_params: int
    num_tensors: int
    total_bytes: int


def autocast_weight_cache(model: nn.Module, dtype: torch.dtype) -> WeightCache:
    """Bytes added by autocast — the term that makes model states *grow*.

    Inside an autocast region each eligible weight is cast once and the result is
    cached (`cache_enabled=True` by default), then held by the autograd graph through
    backward. So the bf16 copies are live *simultaneously* with the fp32 masters
    rather than replacing them.

    For a model whose parameters are essentially all in Linear layers, this is
    2 bytes added to 16, so peak model states rise by almost exactly **one eighth**.
    That is the sharpest prediction in step 2, and it is the one people get backwards:
    mixed precision is widely assumed to shrink model states, and it does the opposite.

    Whether *total* memory rises or falls then depends on the regime, because
    activations do shrink. Below roughly a few hundred tokens per step the cache
    dominates and peak memory goes up; above it the activation saving wins.
    """
    weights = autocast_eligible_weights(model)
    num_params = sum(w.numel() for w in weights)
    return WeightCache(
        num_params=num_params,
        num_tensors=len(weights),
        total_bytes=num_params * BYTES_PER_ELEMENT[dtype],
    )


def exact(num_params: int, vocab_size: int, *, optimizer: str = "adamw") -> dict[str, float]:
    """The bucket arithmetic, keyed as a report compares it.

    Deliberately narrow. These distribute a *measured* parameter count across the three
    buckets, so they check the 4/4/8 rule rather than anything a step asks for. Anything
    the exercise is to derive — the parameter count itself, bytes per parameter,
    activations, peak — is absent on purpose: a prediction the harness supplies would
    confirm itself. A hand-written `prediction.toml` overrides any key here.
    """
    states = model_states(num_params, optimizer=optimizer)
    return {
        "params_mib": states.param_bytes / 1024**2,
        "gradients_mib": states.grad_bytes / 1024**2,
        "optimizer_states_mib": states.optimizer_bytes / 1024**2,
        "model_states_mib": states.total_bytes / 1024**2,
        "initial_loss": expected_initial_loss(vocab_size),
    }


def expected_initial_loss(vocab_size: int) -> float:
    """ln(V): a randomly initialized LM is uniform over the vocabulary.

    The cheapest correctness check in the lab. Cross-entropy against a uniform
    distribution over V classes is ln(V), so a fresh model must start there — 10.8027
    for SmolLM2's 49,152-token vocabulary. If step 1 does not begin near it, the loss
    computation is wrong and every downstream number is noise.

    It must be read from a forward pass on *pristine* weights. One AdamW step moves
    every parameter by roughly the learning rate, which is small enough that a
    contaminated reading still looks plausible — the worst kind of broken check.
    """
    return math.log(vocab_size)
