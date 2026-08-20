"""Ground-truth every claim the wiki makes about the model, its optimizer and its memory.

Run: cd /home/titus/src/loop-to-cluster && pixi run python figures/verify_facts.py
"""

import glob
import inspect
import json
import math
import os

import torch
from torch import nn

from l2c.common.model import SMOLLM2_135M, build_model, causal_lm_loss, preset_dict

print("=" * 62)
print(f"torch {torch.__version__}")
import transformers  # noqa: E402 -- after the torch banner, so versions print together

print(f"transformers {transformers.__version__}")
print("=" * 62)

model = build_model(SMOLLM2_135M, seed=0)

# ---------------------------------------------------------------- norms
print("\n### Normalization layer type")
norm_types = {type(m).__name__ for n, m in model.named_modules() if "norm" in n.lower()}
print(f"norm module types: {sorted(norm_types)}")
has_bias = {
    n: (m.bias is not None) for n, m in model.named_modules() if isinstance(m, nn.Linear)
}
print(f"nn.Linear count: {len(has_bias)}")
print(f"any Linear with bias: {any(has_bias.values())}")

# ------------------------------------------------------- released config
# The parameter count below is blind to any field that changes no shape, which is
# exactly where a preset drifts from the model it claims to be. Compare field by
# field instead.
print("\n### Preset vs the released config")
_snapshots = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*/config.json"
    )
)
if not _snapshots:
    print("  released config not in the local HF cache — skipped")
else:
    with open(_snapshots[0]) as fh:
        released = json.load(fh)
    for field, ours in sorted(preset_dict(SMOLLM2_135M).items()):
        if field not in released:
            print(f"  {field:<28}{ours!s:>24}  (not in the released config)")
            continue
        theirs = released[field]
        verdict = "match" if ours == theirs else f"DIFFERS — released is {theirs}"
        print(f"  {field:<28}{ours!s:>24}  {verdict}")

# ---------------------------------------------------------------- init
print("\n### Initialization")
print(f"config.initializer_range: {model.config.initializer_range}")
for name in (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.29.mlp.down_proj.weight",
):
    t = dict(model.named_parameters())[name]
    print(f"  {name:<44} std={t.std().item():.5f} mean={t.mean().item():+.6f}")
norm_w = dict(model.named_parameters())["model.layers.0.input_layernorm.weight"]
print(f"  RMSNorm weight: all ones? {bool((norm_w == 1).all())}")

# ------------------------------------------------------- ln(V) empirical
print("\n### The ln(V) check, measured")
V = SMOLLM2_135M.vocab_size
torch.manual_seed(0)
ids = torch.randint(0, V, (2, 128))
with torch.no_grad():
    logits = model(input_ids=ids).logits
loss = causal_lm_loss(logits, ids)
print(f"  ln(V) = ln({V}) = {math.log(V):.4f}")
print(f"  measured initial loss = {loss.item():.4f}")
print(f"  absolute error        = {abs(loss.item() - math.log(V)):.4f}")

# ------------------------------------------------------ param groups
print("\n### Parameter groups (the standard no-decay split)")
decay, no_decay = [], []
for n, p in model.named_parameters():
    (no_decay if p.ndim < 2 else decay).append((n, p.numel()))
print(f"  decay    : {len(decay):>4} tensors, {sum(c for _, c in decay):>12,} params")
print(f"  no_decay : {len(no_decay):>4} tensors, {sum(c for _, c in no_decay):>12,} params")
print(
    f"  no_decay share: {sum(c for _, c in no_decay) / sum(p.numel() for p in model.parameters()):.4%}"
)
print(f"  no_decay examples: {[n for n, _ in no_decay[:3]]}")

# ------------------------------------------------- optimizer defaults
print("\n### AdamW defaults (signature)")
sig = inspect.signature(torch.optim.AdamW.__init__)
for k in ("lr", "betas", "eps", "weight_decay", "foreach", "fused", "capturable"):
    if k in sig.parameters:
        print(f"  {k:<14} default={sig.parameters[k].default}")

# --------------------------------------------- state_dict structure
print("\n### optimizer.state_dict() structure")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(
    f"  before any step -> state entries: {len(opt.state_dict()['state'])}  (lazy allocation)"
)
loss2 = causal_lm_loss(model(input_ids=ids).logits, ids)
loss2.backward()
opt.step()
sd = opt.state_dict()
print(f"  after one step  -> state entries: {len(sd['state'])}")
print(f"  top-level keys  : {list(sd.keys())}")
print(f"  state key type  : {type(next(iter(sd['state']))).__name__}  (index, NOT name)")
first = sd["state"][next(iter(sd["state"]))]
print(f"  per-param keys  : {list(first.keys())}")
for k, v in first.items():
    print(
        f"      {k:<12} {type(v).__name__:<8} {tuple(v.shape) if hasattr(v, 'shape') else v}"
    )
pg = sd["param_groups"][0]
print(f"  param_group keys: {sorted(pg.keys())}")
print(f"  params field    : list of {len(pg['params'])} int indices")

# ------------------------------------------------- the logits reference
# Cross-entropy's backward needs its internal log-softmax output, not the logits
# passed in, so a Python name is the only thing keeping the largest tensor in the
# step alive across backward(). Needs a CUDA device to measure.
print("\n### Holding a reference to the logits through backward()")
if not torch.cuda.is_available():
    print("  no CUDA device — skipped")
else:
    B, S, MIB = 4, 1024, 1024**2
    cuda_model = build_model(SMOLLM2_135M, seed=0).cuda()
    cuda_ids = torch.randint(0, V, (B, S), device="cuda")

    def _peak(*, hold: bool) -> float:
        cuda_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = cuda_model(input_ids=cuda_ids)
        step_loss = causal_lm_loss(out.logits, cuda_ids)
        if not hold:
            del out
        step_loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / MIB

    _peak(hold=True)  # settle the allocator before measuring
    held, freed = _peak(hold=True), _peak(hold=False)
    nominal = B * S * V * 4 / MIB
    print(f"  logits {B} x {S} x {V} x 4 B   = {nominal:>9,.0f} MiB")
    print(f"  peak, reference held        = {held:>9,.1f} MiB")
    print(f"  peak, reference dropped     = {freed:>9,.1f} MiB")
    print(
        f"  cost of the reference       = {held - freed:>9,.1f} MiB"
        f"  ({(held - freed) / nominal:.2f} x the tensor)"
    )

print("=" * 62)
