"""Verify the analytic parameter count against the constructed model.

Run: cd /home/titus/src/loop-to-cluster && pixi run python figures/verify_params.py

Nothing here is a step prediction: step 1 is a completed worked example, and its
parameter count is already stated in l2c/common/model.py.
"""

from collections import defaultdict

from l2c.common.model import SMOLLM2_135M, build_model, count_parameters

p = SMOLLM2_135M
d, V, L = p.hidden_size, p.vocab_size, p.num_hidden_layers
ffn = p.intermediate_size
head_dim = d // p.num_attention_heads
kv_width = p.num_key_value_heads * head_dim
q_width = p.num_attention_heads * head_dim

embed = V * d
attn = d * q_width + d * kv_width * 2 + q_width * d
mlp = 3 * d * ffn          # gated: gate, up, down
norms = 2 * d              # two RMSNorm weights per block
per_layer = attn + mlp + norms
analytic = embed + L * per_layer + d   # + final norm; lm_head tied -> 0

# The textbook heuristic, for comparison.
textbook = 12 * L * d**2 + V * d

model = build_model(p, seed=0)
measured = count_parameters(model)

buckets: dict[str, int] = defaultdict(int)
seen: set[int] = set()
for name, t in model.named_parameters():
    if id(t) in seen:
        continue
    seen.add(id(t))
    if "embed" in name:
        key = "embedding"
    elif "norm" in name:
        key = "norms"
    elif any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
        key = "attention"
    elif any(k in name for k in ("gate_proj", "up_proj", "down_proj")):
        key = "mlp"
    else:
        key = "other:" + name
    buckets[key] += t.numel()

print(f"head_dim            {head_dim}")
print(f"q_width / kv_width  {q_width} / {kv_width}")
print(f"attn  per layer     {attn:>12,}")
print(f"mlp   per layer     {mlp:>12,}")
print(f"norms per layer     {norms:>12,}")
print(f"TOTAL per layer     {per_layer:>12,}")
print(f"embedding           {embed:>12,}")
print(f"final norm          {d:>12,}")
print("-" * 40)
print(f"analytic            {analytic:>12,}")
print(f"measured            {measured:>12,}")
print(f"MATCH               {analytic == measured}")
print("-" * 40)
print(f"textbook 12Ld^2+Vd  {textbook:>12,}")
print(f"error vs truth      {(textbook - measured) / measured:+.2%}")
print("-" * 40)
for k in sorted(buckets, key=lambda k: -buckets[k]):
    print(f"{k:<20}{buckets[k]:>12,}  {buckets[k] / measured:>6.1%}")
print("-" * 40)
print(f"tied lm_head        {model.config.tie_word_embeddings}")
print(f"untied would add    {embed:>12,}  ({embed / measured:.1%} more)")
