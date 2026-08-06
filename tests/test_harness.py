"""Tests for the harness logic that every measurement depends on.

Scope is deliberate: this covers the arithmetic and bookkeeping, not the measurements
themselves. Whether a 4090 takes 56 ms for a step is not a testable proposition, but
whether `requested_bytes` deduplicates a tied weight is — and a silent failure there
would move every memory number in the lab at once.

The two real bugs found while building steps 1 and 2 were both caught by a prediction
sharp enough to decompose its own error. That worked, but it is luck rather than method:
it required a quantity to be predictable to the byte *and* to be wrong. These tests cover
the pieces where nothing would have been suspicious.
"""

import math

import pytest
import torch
from torch import nn

from l2c.harness import ledger, measure, predict, report
from l2c.harness.fit import Fit, least_squares, pluck
from l2c.harness.precision import Precision

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="measures CUDA allocator state"
)


# --------------------------------------------------------------------------------
# predict — the arithmetic the whole lab is checked against
# --------------------------------------------------------------------------------


def test_model_states_is_sixteen_bytes_per_param_for_adamw():
    states = predict.model_states(1_000_000)
    assert states.param_bytes == 4_000_000
    assert states.grad_bytes == 4_000_000
    assert states.optimizer_bytes == 8_000_000
    assert states.bytes_per_param == 16.0


@pytest.mark.parametrize(
    ("optimizer", "expected"),
    [("sgd", 8.0), ("sgd_momentum", 12.0), ("adamw", 16.0)],
)
def test_optimizer_choice_moves_only_the_optimizer_bucket(optimizer, expected):
    states = predict.model_states(1_000, optimizer=optimizer)
    assert states.bytes_per_param == expected


def test_bf16_params_would_halve_every_bucket():
    """Not how autocast works — autocast leaves parameters fp32 — but the arithmetic
    for a genuinely 16-bit model should still be right, since step 5 needs it."""
    states = predict.model_states(1_000, param_dtype=torch.bfloat16)
    assert states.param_bytes == 2_000
    assert states.grad_bytes == 2_000


def test_expected_initial_loss_is_log_vocab():
    assert predict.expected_initial_loss(49152) == pytest.approx(math.log(49152))
    assert predict.expected_initial_loss(49152) == pytest.approx(10.8027, abs=1e-4)


def test_autocast_eligible_weights_dedupes_a_tied_head():
    """A tied lm_head shares the embedding table, and the cast must be counted once."""
    embedding = nn.Embedding(100, 8)
    head = nn.Linear(8, 100, bias=False)
    head.weight = embedding.weight
    model = nn.Sequential(embedding, nn.Linear(8, 8, bias=False), head)

    weights = predict.autocast_eligible_weights(model)
    cache = predict.autocast_weight_cache(model, torch.bfloat16)

    assert len(weights) == 2, "the tied head and the hidden projection, not three tensors"
    assert cache.num_tensors == 2
    assert cache.num_params == 100 * 8 + 8 * 8
    assert cache.total_bytes == cache.num_params * 2


def test_weight_cache_is_one_eighth_of_model_states():
    """The headline step-2 prediction: 2 bytes added to 16 for an all-Linear model."""
    model = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Linear(64, 64, bias=False))
    num_params = sum(p.numel() for p in model.parameters())
    states = predict.model_states(num_params)
    cache = predict.autocast_weight_cache(model, torch.bfloat16)
    assert cache.total_bytes / states.total_bytes == pytest.approx(0.125)


# --------------------------------------------------------------------------------
# measure — byte accounting
# --------------------------------------------------------------------------------


@requires_cuda
def test_requested_bytes_counts_a_shared_storage_once():
    base = torch.zeros(256, device="cuda")
    view = base.view(16, 16)
    assert measure.requested_bytes([base]) == 1024
    assert measure.requested_bytes([base, view]) == 1024, "a view is not a second allocation"


@requires_cuda
def test_requested_bytes_skips_cpu_and_none():
    on_device = torch.zeros(256, device="cuda")
    on_host = torch.zeros(4096)
    assert measure.requested_bytes([on_device, on_host, None]) == 1024


@requires_cuda
def test_state_inventory_refuses_to_report_a_missing_bucket():
    model = nn.Linear(8, 8).cuda()
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(RuntimeError, match="optimizer state is empty"):
        measure.state_inventory(model, optimizer)

    model(torch.randn(2, 8, device="cuda")).sum().backward()
    optimizer.step()
    inventory = measure.state_inventory(model, optimizer)
    assert inventory.grad_bytes == inventory.param_bytes
    assert inventory.optimizer_bytes == 2 * inventory.param_bytes

    optimizer.zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError, match="no parameter has a gradient"):
        measure.state_inventory(model, optimizer)


def test_timings_reports_median_not_mean():
    """One slow step must not move the headline number."""
    timings = measure.Timings(device_ms=[10.0] * 9 + [1000.0], wall_seconds=1.09)
    assert timings.median_ms == 10.0
    assert timings.steps_per_second == pytest.approx(10 / 1.09)
    assert timings.tokens_per_second(2048) == pytest.approx(2048 * 10 / 1.09)


def test_snapshot_overhead_is_reserved_minus_allocated():
    snapshot = measure.Snapshot(allocated=1, reserved=2, peak_allocated=100, peak_reserved=175)
    assert snapshot.allocator_overhead == 75


# --------------------------------------------------------------------------------
# ledger — categorization
# --------------------------------------------------------------------------------


def _entry(category: str, nbytes: int, dtype: str = "float32") -> ledger.SavedTensor:
    return ledger.SavedTensor(category=category, shape=(1,), dtype=dtype, nbytes=nbytes)


def test_ledger_totals_by_category_and_dtype():
    inventory = ledger.Ledger(
        entries=[
            _entry(ledger.ACTIVATIONS, 100),
            _entry(ledger.ACTIVATIONS, 50, "bfloat16"),
            _entry(ledger.LOGITS, 400),
            _entry(ledger.WEIGHT_CASTS, 20, "bfloat16"),
        ]
    )
    assert inventory.total_bytes == 570
    assert inventory.bytes_in(ledger.ACTIVATIONS) == 150
    assert inventory.bytes_in(ledger.ACTIVATIONS, ledger.LOGITS) == 550
    assert inventory.count_in(ledger.WEIGHT_CASTS) == 1
    assert inventory.by_dtype() == {"bfloat16": 70, "float32": 500}
    assert inventory.by_category()[ledger.PARAMETERS] == 0


def test_ledger_bytes_in_with_no_argument_covers_everything():
    inventory = ledger.Ledger(
        entries=[_entry(ledger.ACTIVATIONS, 7), _entry(ledger.LOGITS, 3)]
    )
    assert inventory.bytes_in() == inventory.total_bytes == 10


@requires_cuda
def test_ledger_finds_every_weight_cast_at_a_colliding_shape():
    """A (576, 576) activation has the same shape as a square projection weight.

    An earlier classification matched on shape and would file that activation as a cast.
    Walking the backward graph identifies casts by provenance instead, so the count is
    exact even when shapes collide.
    """
    width = 576
    model = nn.Sequential(
        nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)
    ).cuda()
    # Same shape as a projection weight, and requires grad so both layers need their
    # weight for the input gradient — otherwise the first cast is never saved at all.
    activations_shaped_like_a_weight = torch.randn(
        width, width, device="cuda", requires_grad=True
    )

    with (
        ledger.record(model, vocab_size=-1) as inventory,
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        out = model(activations_shaped_like_a_weight)
    del out

    assert inventory.count_in(ledger.WEIGHT_CASTS) == 2
    assert inventory.bytes_in(ledger.WEIGHT_CASTS) == 2 * width * width * 2
    # The colliding activation is still an activation, on provenance rather than shape.
    assert inventory.bytes_in(ledger.ACTIVATIONS) > 0


@requires_cuda
def test_ledger_omits_a_weight_the_forward_never_needed():
    """Why the count invariant is `<=` and not `==`.

    A first layer whose input does not require grad never needs its weight for backward,
    so no cast of it is saved. Fewer casts than eligible weights is legitimate; more is a
    misclassification.
    """
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)).cuda()
    with (
        ledger.record(model, vocab_size=-1) as inventory,
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        out = model(torch.randn(4, 8, device="cuda"))  # no requires_grad
    del out
    assert inventory.count_in(ledger.WEIGHT_CASTS) == 1


@requires_cuda
def test_ledger_records_no_casts_without_autocast():
    """The fp32 control: zero weight casts is what makes the bf16 delta measurable."""
    model = nn.Linear(32, 32, bias=False).cuda()
    with ledger.record(model, vocab_size=-1) as inventory:
        out = model(torch.randn(4, 32, device="cuda", requires_grad=True))
    del out
    assert inventory.count_in(ledger.WEIGHT_CASTS) == 0


# --------------------------------------------------------------------------------
# precision — the four arms
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm", "matmul", "dtype", "scaler"),
    [
        (Precision.FP32, "highest", None, False),
        (Precision.TF32, "high", None, False),
        (Precision.BF16, "highest", torch.bfloat16, False),
        (Precision.FP16, "highest", torch.float16, True),
    ],
)
def test_precision_arms(arm, matmul, dtype, scaler):
    assert arm.matmul_precision == matmul
    assert arm.autocast_dtype is dtype
    assert arm.uses_autocast is (dtype is not None)
    assert arm.needs_scaler is scaler


def test_only_tf32_relaxes_fp32_matmuls():
    relaxed = [arm for arm in Precision if arm.matmul_precision != "highest"]
    assert relaxed == [Precision.TF32]


def test_precision_is_usable_as_a_cli_value():
    assert Precision("bf16") is Precision.BF16
    assert str(Precision.BF16) == "bf16"


# --------------------------------------------------------------------------------
# fit — the sweep
# --------------------------------------------------------------------------------


def test_least_squares_recovers_an_exact_line():
    slope, intercept = 3_540_096, 28_312_128
    xs = [5, 10, 15, 20, 25, 30]
    ys = [slope * x + intercept for x in xs]
    fit = least_squares(xs, ys)
    assert fit.slope == pytest.approx(slope)
    assert fit.intercept == pytest.approx(intercept)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.num_points == 6
    assert fit.predict(7) == pytest.approx(slope * 7 + intercept)


def test_least_squares_reports_a_poor_fit_as_poor():
    fit = least_squares([1, 2, 3, 4], [1.0, 9.0, 2.0, 8.0])
    assert fit.r_squared < 0.5


def test_least_squares_needs_two_points():
    with pytest.raises(ValueError, match="at least two points"):
        least_squares([1], [1.0])


def test_least_squares_handles_a_flat_line():
    """Zero variance in y would divide by zero in the R^2 denominator."""
    fit = least_squares([1, 2, 3], [5.0, 5.0, 5.0])
    assert fit.slope == pytest.approx(0.0)
    assert fit.r_squared == 1.0


def test_pluck_searches_config_before_measurements():
    entry = {
        "preset": {"num_hidden_layers": 30},
        "run": {"batch_size": 4},
        "actual": {"peak_mib": 6621.9, "num_hidden_layers": 999},
    }
    assert pluck(entry, "num_hidden_layers") == 30.0, "preset outranks actual"
    assert pluck(entry, "batch_size") == 4.0
    assert pluck(entry, "peak_mib") == pytest.approx(6621.9)
    assert pluck(entry, "absent") is None


def test_pluck_ignores_non_numeric_values():
    assert pluck({"run": {"precision": "bf16"}}, "precision") is None


def test_pluck_tolerates_a_missing_section():
    assert pluck({}, "anything") is None


# --------------------------------------------------------------------------------
# report — the table
# --------------------------------------------------------------------------------


def test_table_marks_a_hit_and_a_miss_by_tolerance():
    text = report.table(
        [
            report.Row("exact", 100.0, 100.0, ",.1f", 0.1),
            report.Row("just inside", 100.0, 100.05, ",.1f", 0.1),
            report.Row("outside", 100.0, 110.0, ",.1f", 0.1),
        ]
    )
    lines = {line.split("  ")[0].strip(): line for line in text.splitlines()}
    assert lines["exact"].endswith("ok")
    assert lines["just inside"].endswith("ok")
    assert lines["outside"].endswith("OFF")
    assert "+10.00%" in lines["outside"]


def test_table_reports_a_measurement_with_no_prediction():
    text = report.table([report.Row("block padding (MiB)", None, 90.3, ",.1f")])
    assert "90.3" in text
    assert "OFF" not in text and "ok" not in text


def test_table_does_not_divide_by_a_zero_prediction():
    text = report.table([report.Row("weight cache", 0.0, 0.0, ",.1f")])
    assert "n/a" in text


def test_load_prediction_is_empty_when_absent(tmp_path):
    assert report.load_prediction(tmp_path) == {}


def test_load_prediction_reads_toml(tmp_path):
    (tmp_path / "prediction.toml").write_text(
        "num_params = 134515008\nbytes_per_param = 16.0\n"
    )
    prediction = report.load_prediction(tmp_path)
    assert prediction["num_params"] == 134515008
    assert prediction["bytes_per_param"] == 16.0
    assert prediction.get("activations_mib") is None, "an omitted key is just absent"


def test_fit_dataclass_is_immutable():
    fit = Fit(1.0, 2.0, 1.0, 3)
    with pytest.raises(AttributeError):
        fit.slope = 5.0
