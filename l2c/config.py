"""Run configuration.

One `RunConfig` fully determines a run: same config plus same seed must give the same
loss curve. Everything a variant changes (precision today, grad accumulation and
parallelism later) is a field here, so a variant is a diff you can print.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import torch

# Precision modes. `act_dtype` is the dtype low-precision ops produce under autocast;
# it is float32 for the modes that do not use autocast at all.
PRECISIONS = {
    # IEEE fp32 matmuls: no tensor cores, no reduced-precision storage.
    "fp32": dict(autocast=None, tf32=False, scaler=False),
    # Tensor cores with fp32 storage. Isolates "AMP is faster because of tensor
    # cores" from "AMP is faster because it moves half the bytes".
    "tf32": dict(autocast=None, tf32=True, scaler=False),
    "amp_bf16": dict(autocast=torch.bfloat16, tf32=True, scaler=False),
    # fp16 has no exponent headroom for gradients, so it needs loss scaling.
    "amp_fp16": dict(autocast=torch.float16, tf32=True, scaler=True),
}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048
    max_seq: int = 1024

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 8
    seq_len: int = 512
    # Drawn once, up front, so every precision mode sees byte-identical batches in the
    # same order.
    #
    # Small on purpose: 64Ki tokens is a handful of batches, which the model can memorise,
    # so the loss falls from ln(V) to a few nats over a couple of hundred steps. A corpus
    # large enough to be unmemorisable gives a nearly flat curve, and "the two precisions
    # agree" is a vacuous claim about two flat lines. A moving curve also exercises fp16
    # loss scaling on gradients that actually vary in magnitude.
    corpus_tokens: int = 1 << 16
    data_seed: int = 1234


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # AdamW kernel shape. "foreach" is torch's default on CUDA and allocates
    # param-sized temporaries during the step; "fused" allocates none. This is a
    # peak-memory term that has nothing to do with precision, so it is pinned
    # explicitly rather than left to a version-dependent default.
    impl: str = "foreach"


@dataclass(frozen=True)
class RunConfig:
    precision: str = "fp32"
    seed: int = 0
    steps: int = 50
    warmup: int = 10
    device: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    # Optional extra passes, off by default so they cannot perturb the timed run.
    anatomy: bool = True
    snapshot: str | None = None

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.data.seq_len > self.model.max_seq:
            raise ValueError("seq_len exceeds model.max_seq")

    @property
    def spec(self) -> dict[str, Any]:
        return PRECISIONS[self.precision]

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return self.spec["autocast"]

    @property
    def act_dtype(self) -> torch.dtype:
        """Dtype of tensors produced by autocast-eligible ops."""
        return self.autocast_dtype or torch.float32

    @property
    def tokens_per_step(self) -> int:
        return self.data.batch_size * self.data.seq_len

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def config_from_args(args: Any) -> RunConfig:
    """Build a RunConfig from argparse output."""
    return RunConfig(
        precision=args.precision,
        seed=args.seed,
        steps=args.steps,
        warmup=args.warmup,
        device=args.device,
        model=ModelConfig(
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            d_ff=args.d_ff or 4 * args.d_model,
        ),
        data=DataConfig(batch_size=args.batch_size, seq_len=args.seq_len),
        optim=OptimConfig(impl=args.optim_impl, grad_clip=args.grad_clip),
        anatomy=not args.no_anatomy,
        snapshot=args.snapshot,
    )
