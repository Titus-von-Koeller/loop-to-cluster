"""Turn the measured objects into the flat record a run is stored as.

One place decides what a measurement is called, so a key means the same thing in every
step, in `runs/`, and on the axis of a fit. A step assembling this itself is a step
where the loop competes for attention with twenty lines of dictionary.

Nothing here measures. Every argument is already-collected state, and the only work is
naming and unit conversion.
"""

from l2c.harness.ledger import ACTIVATIONS, LOGITS, WEIGHT_CASTS, Ledger
from l2c.harness.measure import Phases, Snapshot, StateInventory, Timings, mib


def run(
    *,
    num_params: int,
    tokens_per_step: int,
    phases: Phases,
    model_states: StateInventory,
    timings: Timings,
    peak: Snapshot,
    saved: Ledger,
    losses: list[float],
) -> dict[str, object]:
    """Everything every step measures, whatever its lesson is about."""
    return {
        "num_params": num_params,
        "bytes_per_param": model_states.total_bytes / num_params,
        "initial_loss": phases.initial_loss,
        "final_loss": losses[-1],
        "params_mib": mib(model_states.param_bytes),
        "gradients_mib": mib(model_states.grad_bytes),
        "optimizer_states_mib": mib(model_states.optimizer_bytes),
        "model_states_mib": mib(model_states.total_bytes),
        "model_states_allocated_mib": mib(phases.after_step),
        "block_padding_mib": mib(phases.after_step - model_states.total_bytes),
        "activations_mib": mib(phases.activations),
        "peak_mib": mib(peak.peak_allocated),
        "peak_reserved_mib": mib(peak.peak_reserved),
        "allocator_overhead_mib": mib(peak.allocator_overhead),
        "median_step_ms": timings.median_ms,
        "p10_step_ms": timings.p10_ms,
        "p90_step_ms": timings.p90_ms,
        "steps_per_second": timings.steps_per_second,
        "tokens_per_second": timings.tokens_per_second(tokens_per_step),
        "saved_tensors": saved.summary(),
        "loss_curve": losses,
    }


def mixed_precision(
    *,
    precision: object,
    phases: Phases,
    saved: Ledger,
    grad_scale: float,
    steps_attempted: int,
    steps_applied: int,
) -> dict[str, object]:
    """What step 2 adds: where the bytes went once autocast is involved.

    `activations_excl_cache_mib` exists because the raw forward-pass delta bundles the
    weight cache — autocast allocates those copies during the forward. Subtracting it
    separates what the graph holds in order to differentiate from what mixed precision
    added in order to hold the weights twice.
    """
    cache_bytes = saved.bytes_in(WEIGHT_CASTS)
    return {
        "precision": str(precision),
        "weight_cache_mib": mib(cache_bytes),
        "weight_cache_tensors": saved.count_in(WEIGHT_CASTS),
        "activations_excl_cache_mib": mib(phases.activations - cache_bytes),
        "saved_activations_mib": mib(saved.bytes_in(ACTIVATIONS)),
        "saved_logits_mib": mib(saved.bytes_in(LOGITS)),
        "saved_still_fp32_mib": mib(saved.by_dtype().get("float32", 0)),
        "final_grad_scale": grad_scale,
        "steps_attempted": steps_attempted,
        "steps_applied": steps_applied,
        "steps_skipped": steps_attempted - steps_applied,
    }
