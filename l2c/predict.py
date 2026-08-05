"""The analytic memory model — the theory side of the comparison.

Written from the forward pass and the autocast dtype policy, then checked against
`instrument.py`'s measurements. The split is deliberate:

* **Exactly predictable**: parameters, gradients, optimiser state, the autocast weight
  cache, and the set of tensors backward holds. These follow from the config, and the
  harness asserts them to the byte.
* **Measured, not predicted**: transient peaks inside backward and inside the optimiser
  step. Those depend on kernel implementations — `foreach` versus `fused` AdamW, flash
  versus mem-efficient attention — and predicting them would turn a check into a curve
  fit.

Two `torch.amp` facts the model has to encode, both verified on this build rather than
assumed:

* `layer_norm` is autocast-promoted to fp32, so it saves an *fp32* copy of the residual
  stream. Those tensors do not shrink under AMP.
* `cross_entropy` is promoted too, and `log_softmax` saves its output, so the (B*S*V)
  fp32 term is identical in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import ModelConfig, RunConfig

MIB = 1024 * 1024


@dataclass(frozen=True)
class Term:
    label: str
    bucket: str  # module path the measured ledger will attribute this to
    n: int  # multiplicity: n_layer for per-block terms, else 1
    elems: int  # elements per instance
    itemsize: int
    note: str = ""

    @property
    def bytes(self) -> int:
        return self.n * self.elems * self.itemsize

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "bytes": self.bytes}


def param_counts(m: ModelConfig) -> dict[str, int]:
    D, F, V, L = m.d_model, m.d_ff, m.vocab_size, m.n_layer
    per_block_linear = 4 * D * D + 2 * F * D  # qkv (3D x D), proj, fc1, fc2
    linear = L * per_block_linear + V * D  # + lm_head
    embedding = V * D + m.max_seq * D
    layernorm = L * 4 * D + 2 * D  # two LayerNorms per block, weight+bias, + ln_f
    return {
        "linear": linear,
        "embedding": embedding,
        "layernorm": layernorm,
        "total": linear + embedding + layernorm,
    }


def static_terms(cfg: RunConfig) -> list[Term]:
    """Parameters, gradients and optimiser state. Precision-independent by design.

    This is where the textbook mixed-precision recipe and what `autocast` actually does
    diverge. The textbook keeps a *persistent* fp16 copy of the weights and accumulates
    fp16 gradients, costing 2P for weights plus 2P for gradients on top of the fp32
    master copies. `torch.amp` keeps weights and gradients in fp32 and casts on the fly,
    so none of these four lines change with precision.
    """
    p = param_counts(cfg.model)["total"]
    terms = [
        Term("parameters (fp32 master)", "-", 1, p, 4),
        Term("gradients (fp32)", "-", 1, p, 4),
        Term("AdamW exp_avg", "-", 1, p, 4),
        Term("AdamW exp_avg_sq", "-", 1, p, 4),
    ]
    if cfg.spec["scaler"]:
        terms.append(Term("GradScaler scale + growth tracker", "-", 2, 1, 4, "one element each"))
    return terms


def weight_cache_terms(cfg: RunConfig) -> list[Term]:
    """Autocast's cached low-precision copies of the Linear weights.

    Each `nn.Linear` weight is cast once per autocast region and the cast is cached, so
    the cost is one low-precision copy of the *Linear* weights — not of all parameters.
    Embeddings and LayerNorm are not autocast-eligible. Backward needs those same casts
    for the input-gradient matmul, so they survive until backward is done.

    This is AMP's only fixed memory cost, and it scales with parameters, not with batch.
    """
    if cfg.autocast_dtype is None:
        return []
    itemsize = torch.finfo(cfg.autocast_dtype).bits // 8
    return [
        Term(
            f"autocast weight cache ({str(cfg.autocast_dtype).removeprefix('torch.')})",
            "weight_cast",
            1,
            param_counts(cfg.model)["linear"],
            itemsize,
            "Linear weights only",
        )
    ]


def default_sdpa_backend(cfg: RunConfig) -> str:
    """Which attention kernel a given precision gets on an Ada/Ampere GPU.

    Flash attention rejects fp32 inputs, so the fp32 modes fall back to the mem-efficient
    kernel. `train.py` measures this rather than assuming it, and passes the measured value
    in; this default exists so the predictor is usable without a GPU.
    """
    return "mem_efficient" if cfg.act_dtype == torch.float32 else "flash"


def activation_terms(cfg: RunConfig, sdpa_backend: str | None = None) -> list[Term]:
    """Tensors backward holds after a forward pass, term by term.

    Derived by walking the forward pass and asking of each op what its backward needs:
    matmul needs its input, GELU needs its input, LayerNorm needs its input plus mean and
    rstd, SDPA needs q, k, v, its output and the log-sum-exp.

    Views are free, and two of them are load-bearing here. `qkv(x).split(...)` followed by
    `view`/`transpose` produces views of one (B, S, 3D) allocation, charged once. And
    SDPA returns its output strided so that `transpose(1, 2).reshape(B, S, D)` *is* a
    view — verified for both the flash and the mem-efficient kernel — so the attention
    projection's input costs nothing beyond the SDPA output already charged. Writing
    `.contiguous()` there would add one B*S*D tensor per layer.

    Note which terms carry `4` rather than `a`: the LayerNorm inputs stay fp32 in both
    modes, which is why AMP shrinks activations by distinctly less than half.
    """
    m, d = cfg.model, cfg.data
    B, S, D, F, V, L, H = (
        d.batch_size,
        d.seq_len,
        m.d_model,
        m.d_ff,
        m.vocab_size,
        m.n_layer,
        m.n_head,
    )
    a = torch.finfo(cfg.act_dtype).bits // 8  # activation itemsize
    BSD, BSF, BS = B * S * D, B * S * F, B * S
    backend = sdpa_backend or default_sdpa_backend(cfg)

    return [
        # -- per block, x n_layer -----------------------------------------------------
        Term("ln1 saved input", "blocks.*.ln1", L, BSD, 4, "fp32: layer_norm is promoted"),
        Term("ln1 mean+rstd", "blocks.*.ln1", L, 2 * BS, 4),
        Term("qkv input", "blocks.*.attn.qkv", L, BSD, a),
        Term("qkv output (q|k|v)", "blocks.*.attn", L, 3 * BSD, a, "one alloc, three views"),
        Term("sdpa output", "blocks.*.attn", L, BSD, a),
        Term("sdpa logsumexp", "blocks.*.attn", L, B * H * S, 4),
        # Flash keeps the philox seed and offset so its backward can regenerate the same
        # dropout mask. Saved unconditionally, even at dropout_p=0. The mem-efficient
        # kernel does not save them.
        *(
            [Term("flash philox seed+offset", "blocks.*.attn", L, 3, 8, "uint64, even at p=0")]
            if backend == "flash"
            else []
        ),
        Term("ln2 saved input", "blocks.*.ln2", L, BSD, 4, "fp32: layer_norm is promoted"),
        Term("ln2 mean+rstd", "blocks.*.ln2", L, 2 * BS, 4),
        Term("fc1 input", "blocks.*.mlp.fc1", L, BSD, a),
        Term("gelu input (fc1 output)", "blocks.*.mlp", L, BSF, a),
        Term("fc2 input (gelu output)", "blocks.*.mlp.fc2", L, BSF, a),
        # -- head ---------------------------------------------------------------------
        Term("ln_f saved input", "ln_f", 1, BSD, 4),
        Term("ln_f mean+rstd", "ln_f", 1, 2 * BS, 4),
        Term("lm_head input", "lm_head", 1, BSD, a),
        # The loss head is where AMP costs memory rather than saving it. log_softmax runs
        # in the autocast dtype and saves its own output; nll_loss is fp32-promoted, so it
        # saves an fp32 *cast* of that output. Under AMP those are two distinct tensors
        # (B*S*V at 2 bytes plus B*S*V at 4); in fp32 they are the same tensor, charged
        # once. Chunking the loss or fusing it is what removes this term.
        Term("log_softmax output", "<root>", 1, B * S * V, a),
        *(
            [Term("nll_loss input (fp32 cast of log-probs)", "<root>", 1, B * S * V, 4)]
            if cfg.autocast_dtype is not None
            else []
        ),
        Term("cross_entropy target", "<root>", 1, BS, 8, "int64"),
        Term("loss scalar", "<root>", 1, 1, 4, "saved by nll_loss backward"),
        Term("token indices", "tok_emb", 1, BS, 8, "int64"),
        Term("position indices", "pos_emb", 1, S, 8, "int64"),
    ]


def bytes_of(terms: list[Term]) -> int:
    return sum(t.bytes for t in terms)


def bucket_bytes(terms: list[Term]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in terms:
        out[t.bucket] = out.get(t.bucket, 0) + t.bytes
    return out


def predict(cfg: RunConfig, sdpa_backend: str | None = None) -> dict[str, Any]:
    static = static_terms(cfg)
    cache = weight_cache_terms(cfg)
    acts = activation_terms(cfg, sdpa_backend)
    return {
        "param_counts": param_counts(cfg.model),
        "terms": {
            "static": [t.to_dict() for t in static],
            "weight_cache": [t.to_dict() for t in cache],
            "activations": [t.to_dict() for t in acts],
        },
        "activation_buckets": bucket_bytes(acts),
        "static_bytes": bytes_of(static),
        "weight_cache_bytes": bytes_of(cache),
        "activation_bytes": bytes_of(acts),
        "persistent_bytes": bytes_of(static) + bytes_of(cache) + bytes_of(acts),
    }


def activation_bytes_per_token(cfg: RunConfig) -> float:
    return bytes_of(activation_terms(cfg)) / cfg.tokens_per_step


def persistent_bytes(cfg: RunConfig, sdpa_backend: str | None = None) -> int:
    return (
        bytes_of(static_terms(cfg))
        + bytes_of(weight_cache_terms(cfg))
        + bytes_of(activation_terms(cfg, sdpa_backend))
    )


def crossover_batch_tokens(base: RunConfig, amp: RunConfig) -> float:
    """The B*S at which AMP stops costing memory and starts saving it.

    AMP pays a fixed cost — the weight cache, proportional to parameters — and earns a
    saving proportional to activations, hence to B*S. Below the crossover the fixed cost
    dominates and peak memory goes *up*; above it AMP is a memory optimisation. This is
    the sharpest prediction the model makes, because it is a number the theory produces
    that no single run reveals.
    """
    saving_per_token = activation_bytes_per_token(base) - activation_bytes_per_token(amp)
    if saving_per_token <= 0:
        return float("inf")
    return bytes_of(weight_cache_terms(amp)) / saving_per_token
