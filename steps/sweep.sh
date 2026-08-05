#!/usr/bin/env bash
# Validate the memory formula against a LINE, not a single point.
# One matching number is a coincidence; six on a line is a formula.
#
# Depth is the axis to sweep. Every quantity of interest is linear in it:
#
#     num_params = embedding + final_norm + num_layers * per_layer_params
#
# so the fitted slope is the per-layer cost and the intercept is the embedding table.
# Sweeping hidden_size instead would be quadratic in parameters, and would also require
# every value to stay divisible by num_attention_heads — transformers derives
# head_dim = hidden_size // num_attention_heads, so an indivisible width silently
# truncates it and the projections stop being square.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for layers in 5 10 15 20 25 30; do
  echo "=== num_layers=$layers ==="
  python steps/step1_training_loop/train.py --num-layers "$layers" --num-steps 10
done

python -m l2c.harness.fit \
  --step step1_training_loop \
  --x num_hidden_layers \
  --y num_params model_states_mib activations_mib peak_mib
