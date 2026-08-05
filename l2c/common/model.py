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


@dataclass(frozen=True, slots=True)
class Preset:
    """Architecture knobs.

    Defaults mirror SmolLM2-135M. Verified against the released config: this yields
    134,515,008 parameters, and transformers derives ``head_dim = 576 // 9 = 64``.

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
    max_position_embeddings: int = 2048
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
        tie_word_embeddings=preset.tie_word_embeddings,
        attn_implementation=preset.attn_implementation,
        use_cache=False,  # a KV cache is for inference; in training it is dead weight
    )
    return LlamaForCausalLM(config)


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


def preset_dict(preset: Preset) -> dict[str, object]:
    return asdict(preset)
