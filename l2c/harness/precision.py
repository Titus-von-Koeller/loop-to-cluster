"""The precision axis, as four named arms rather than a boolean.

"Mixed precision on or off" hides the two effects it bundles together, and they are
separable:

    fp32   full 24-bit significand, matmuls on the CUDA cores
    tf32   fp32 tensors, but matmuls truncated to 11 significand bits on the tensor
           cores. Same dtypes, same memory, different kernels.
    bf16   autocast to bfloat16: 8 significand bits, fp32's exponent range, so no
           gradient scaling is needed
    fp16   autocast to float16: 11 significand bits but only 5 exponent bits, so
           gradients underflow and a `GradScaler` is required

`tf32` is the arm that makes the other comparisons interpretable. Going fp32 -> tf32
holds memory constant and changes only the kernels, isolating "tensor cores are faster".
Going tf32 -> bf16 holds the tensor cores constant and changes only the dtype, isolating
"smaller tensors use less memory". Comparing fp32 with bf16 alone conflates the two, and
that conflation is why mixed-precision speedups are so often quoted against a baseline
that had already been silently accelerated.

In accelerate this axis is `Accelerator(mixed_precision="no"|"fp16"|"bf16"|"fp8")`,
resolved in `AcceleratorState` and applied in `_prepare_model` by wrapping the model's
forward (accelerate/accelerator.py, near line 1820). Two consequences of that design are
worth knowing:

- accelerate wraps `model.forward` in the autocast context rather than asking the call
  site to. It also wraps the result in `convert_outputs_to_fp32`, which is why model
  outputs come back fp32 under accelerate but bf16 here.
- `mixed_precision` has no `tf32` value, but TF32 is not left to the caller either: it is
  coupled to torch.compile. `AcceleratorState` sets
  `torch.backends.cuda.matmul.allow_tf32 = True` exactly when a dynamo backend is
  requested *and* `mixed_precision == "no"` (accelerate/state.py:1023). So requesting
  compilation silently changes what the fp32 arm computes and how fast it runs, which
  entangles two comparisons that look independent: compiled-versus-eager, and
  fp32-versus-bf16. transformers couples them the same way, gated on `torch_compile`
  (training_args.py:1604), though it at least exposes an explicit `tf32` flag to override.
"""

from enum import StrEnum

import torch


class Precision(StrEnum):
    FP32 = "fp32"
    TF32 = "tf32"
    BF16 = "bf16"
    FP16 = "fp16"

    @property
    def matmul_precision(self) -> str:
        """What an fp32 matmul is permitted to do, for `set_float32_matmul_precision`."""
        return "high" if self is Precision.TF32 else "highest"

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """The dtype autocast casts op inputs to, or None when autocast is not used."""
        match self:
            case Precision.BF16:
                return torch.bfloat16
            case Precision.FP16:
                return torch.float16
            case _:
                return None

    @property
    def uses_autocast(self) -> bool:
        return self.autocast_dtype is not None

    @property
    def needs_scaler(self) -> bool:
        """Only fp16 does.

        bf16 keeps fp32's 8-bit exponent, so the smallest normal value is about 1.2e-38
        and gradients simply do not underflow. fp16's 5-bit exponent puts that floor at
        6.1e-5, well inside the range real gradients occupy, so they must be scaled up
        before backward and unscaled before the update.
        """
        return self is Precision.FP16

    def apply(self) -> None:
        """Pin the matmul precision for this arm. Never inherit it from a global default."""
        torch.set_float32_matmul_precision(self.matmul_precision)
