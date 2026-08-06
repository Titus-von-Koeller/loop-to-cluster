"""Which quantities a step puts on the table, and how exactly each one must match.

A row names a key and looks it up on both sides: in the predictions and in the
measurements. A key absent from the predictions renders as a measurement with nothing
to compare, which is how a quantity gets reported without being claimed.

`tolerance_pct` is the interesting field. It answers "how close would count as
confirmation?" before the number is known, which is the difference between a prediction
and a description. Bucket arithmetic is exact and gets a tight bound; an activation
estimate is a first guess and gets a loose one. A step's `prediction.toml` may override
any of these under a `[tolerance]` table.

The label, the format and the ordering are presentation and live here. The predicted
value and how exact it must be belong to whoever wrote the prediction.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RowSpec:
    label: str
    key: str
    fmt: str = ",.1f"
    tolerance_pct: float = 1.0
    #: Set when the measurement is not the same quantity the prediction names.
    actual_key: str | None = None

    def measured_key(self) -> str:
        return self.actual_key or self.key


#: Step 1 — the buckets, then the two quantities that have to be measured to be known.
TRAINING_LOOP = (
    RowSpec("parameters", "num_params", ",.0f", 0.0),
    RowSpec("bytes/param", "bytes_per_param", ",.2f"),
    RowSpec("initial loss", "initial_loss", ",.4f"),
    RowSpec("params (MiB)", "params_mib", ",.1f", 0.1),
    RowSpec("gradients (MiB)", "gradients_mib", ",.1f", 0.1),
    RowSpec("optim states (MiB)", "optimizer_states_mib", ",.1f", 0.1),
    RowSpec("model states (MiB)", "model_states_mib", ",.1f", 0.1),
    RowSpec("block padding (MiB)", "block_padding_mib"),
    RowSpec("activations (MiB)", "activations_mib", ",.1f", 10.0),
    RowSpec("peak allocated (MiB)", "peak_mib", ",.1f", 10.0),
    RowSpec("steps/sec", "steps_per_second", ",.2f", 20.0),
)


def mixed_precision(arm: str) -> tuple[RowSpec, ...]:
    """Step 2 — per-arm keys, because the point of the step is the difference between arms.

    The measured activation figure excludes the weight cache while the prediction is of
    activations alone, so the two sides name different keys on purpose.
    """
    return (
        RowSpec("model states (MiB)", "model_states_mib", ",.1f", 0.1),
        RowSpec(
            "weight cache (MiB)", f"weight_cache_mib_{arm}", ",.1f", 0.1, "weight_cache_mib"
        ),
        RowSpec(
            "weight cache tensors",
            f"weight_cache_tensors_{arm}",
            ",.0f",
            0.0,
            "weight_cache_tensors",
        ),
        RowSpec("initial loss", "initial_loss", ",.4f"),
        RowSpec(
            "activations (MiB)",
            f"activations_mib_{arm}",
            ",.1f",
            15.0,
            "activations_excl_cache_mib",
        ),
        RowSpec("peak allocated (MiB)", f"peak_mib_{arm}", ",.1f", 10.0, "peak_mib"),
        RowSpec(
            "median step (ms)", f"median_step_ms_{arm}", ",.1f", 20.0, "median_step_ms"
        ),
    )
