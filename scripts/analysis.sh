#!/bin/bash
# Diagnostic: measure, for every generated token, the distance from the
# PCA-projected hidden state to the feasible region of the fitted SVC
# constraints, then correlate that distance with the response length. Produces
# the distance-vs-length plots and a statistics JSON under results/analysis/.
#
# Note: --param_file is resolved relative to the params/ directory by
# projection_analysis.py, so do not repeat the "params/" prefix here.
#
# Usage: bash scripts/analysis.sh <gpu_id> [model_name]
set -e

gpu=${1:-0}
MODEL=${2:-"DeepSeek-R1-Distill-Qwen-1.5B"}

CUDA_VISIBLE_DEVICES=$gpu python projection_analysis.py \
    --model_name_or_path ~/model/$MODEL \
    --save_dir results/analysis \
    --param_file online/$MODEL/qp_params_dim10_90_layer20_cluster8_noscaler.json \
    --dataset_dir results/COT/$MODEL/Projection/0_400_gen_3/thinking_results.json \
    --save_preprocessed \
    --batch_size 4 \
    --projection_layer 20 \
    --max_examples 400
