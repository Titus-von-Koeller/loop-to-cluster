"""Model construction. Shared, because no lesson is about *building* a model.

The architecture mirrors SmolLM2-135M, but it is constructed from a config rather
than loaded from a checkpoint, for three reasons:

1. Knobs. `num_hidden_layers` can be swept, so the memory formula is checked against
   a line rather than a single point. Parameters are *linear* in depth, so the slope
   is the per-layer cost and the intercept is the embedding table — two predictions
   validated by one sweep.
2. Random init means the initial loss must be ``ln(vocab_size)``. That is a free
   correctness check, and pretrained weights would hide it.
3. Nothing large is downloaded. Only the tokenizer is fetched, once.

In accelerate: nothing in this file has a counterpart, because accelerate never
constructs models — `Accelerator.prepare` takes an already-built `nn.Module`. Its
model-side work happens in `_prepare_model` (accelerate/accelerator.py, around
line 1820): device placement, wrapping `forward` in an autocast context, and
wrapping the module in DDP/FSDP. The one place accelerate does participate in
construction is checkpoint-scale loading — `init_empty_weights` and
`load_checkpoint_and_dispatch` in accelerate/big_modeling.py — which is a different
problem from this one.
"""

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

#: Only the tokenizer is fetched from the Hub; weights are never downloaded.
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"

#: Positions cross-entropy skips. torch's default, and what transformers uses.
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class Preset:
    """Architecture knobs.

    Defaults mirror SmolLM2-135M's released config, and need verifying two ways,
    because a parameter count is blind to half of them. The shape fields reproduce
    134,515,008 parameters, and transformers derives ``head_dim = 576 // 9 = 64``.
    ``initializer_range`` and ``max_position_embeddings`` change no shape at all, so
    that count cannot see them: both are pinned here to the released values rather
    than inherited. Left to transformers, ``initializer_range`` falls back to 0.02
    against the released 0.041666..., which moves the initial loss by a third of a
    nat without moving a single tensor.

    `attn_implementation` is pinned rather than left to transformers' own default
    (which is `None`, meaning "pick one"). The choice changes *which tensors are
    saved for backward*, and therefore the activation number this whole lab exists
    to predict. Install flash-attention later and an unpinned baseline would move
    silently — the same class of bug as inheriting a `float32_matmul_precision`.
    """

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3  # GQA: K and V are a third of Q's width
    max_position_embeddings: int = 8192
    initializer_range: float = 0.041666666666666664  # 1/sqrt(576); transformers' is 0.02
    tie_word_embeddings: bool = True
    attn_implementation: str = "sdpa"


SMOLLM2_135M = Preset()


def build_model(preset: Preset = SMOLLM2_135M, *, seed: int) -> LlamaForCausalLM:
    """Construct a randomly initialized model.

    Seeds the global RNG, because transformers initializes weights through it and
    offers no way to pass a generator. This is the *only* place in the lab that
    touches the global seed; data order gets its own explicit generator (see
    `l2c.common.data.batch_generator`) so that changing the model cannot silently
    change which batches arrive in which order.

    accelerate's equivalent is `accelerate.utils.set_seed` (utils/random.py:40),
    which additionally seeds `random` and `numpy` and can offset the seed per
    process via `device_specific=True`.
    """
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=preset.vocab_size,
        hidden_size=preset.hidden_size,
        intermediate_size=preset.intermediate_size,
        num_hidden_layers=preset.num_hidden_layers,
        num_attention_heads=preset.num_attention_heads,
        num_key_value_heads=preset.num_key_value_heads,
        max_position_embeddings=preset.max_position_embeddings,
        initializer_range=preset.initializer_range,
        tie_word_embeddings=preset.tie_word_embeddings,
        attn_implementation=preset.attn_implementation,
        use_cache=False,  # a KV cache is for inference; in training it is dead weight
    )
    return LlamaForCausalLM(config)


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for next-token prediction, with the shift written out.

    A causal LM predicts token i+1 from everything up to i, so position i of the logits
    is scored against token i+1 of the input. `LlamaForCausalLM` does this internally
    when handed `labels=`, which fuses the forward and the scoring into one call. Doing
    it here keeps them two visible steps, which is the shape every later step's loop is
    a diff against.

    Shared rather than duplicated per step: no lesson in this lab is about the shift.

    The shift is applied to the labels, not to the logits, and that choice is worth a
    measurement. `logits` is contiguous, so flattening it is a view; slicing it to
    `[:, :-1]` first is not, and flattening the result copies a whole vocabulary-sized
    tensor — a few hundred MiB at this vocabulary, appearing in peak memory for no
    reason the lesson can explain. Padding the labels instead costs a few KiB of int64.
    This is also what `LlamaForCausalLM` does internally.
    """
    targets = nn.functional.pad(input_ids, (0, 1), value=IGNORE_INDEX)[:, 1:]
    return nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX
    )


def count_parameters(model: nn.Module) -> int:
    """The checker for the hand derivation. Do the derivation first.

    `parameters()` deduplicates by identity, so a tied `lm_head` is counted once —
    which is what makes the 16 bytes/parameter arithmetic hold. If it double-counted,
    AdamW would appear to need moments for the embedding table twice and the
    prediction would be out by 28.3M x 8 B = 226 MB.
    """
    return sum(p.numel() for p in model.parameters())


def replace_preset(preset: Preset, **changes: object) -> Preset:
    """A `Preset` with some fields changed. Sweep knobs go through here."""
    return replace(preset, **changes)


def preset_for(num_layers: int | None = None, *, base: Preset = SMOLLM2_135M) -> Preset:
    """The preset a run uses, with the depth sweep applied if it was asked for.

    Resolving this in one place is what lets a result be keyed by the model it
    measured rather than by the flags that happened to describe it.
    """
    return base if num_layers is None else replace_preset(base, num_hidden_layers=num_layers)


def preset_dict(preset: Preset) -> dict[str, object]:
    return asdict(preset)
