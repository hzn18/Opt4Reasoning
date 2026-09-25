#!/bin/bash
# Offline constrained-projection steering (`--method projection-offline`).
# The multi-parametric QP is solved ahead of time by qp_offline.py; inference
# only does a GPU point-location lookup plus an affine evaluation.
#
# Usage: bash scripts/run_offline.sh <gpu_id> [model_name]
set -e

gpu=${1:-0}
MODEL=${2:-"DeepSeek-R1-Distill-Qwen-1.5B"}
DATASETS=("AMC23" "AIME2025" "GSM8K" "MATH500" "GPQA")

STEER_FILE="params/offline/$MODEL/qp_params_dim12_90_layer20_cluster8_vllm_ablation.pt"

echo "🚀 Offline projection steering: $MODEL on GPU $gpu"
echo "   Steering file: $STEER_FILE"

for DATASET in "${DATASETS[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python projection_vllm.py \
        --model_name_or_path ~/model/$MODEL \
        --save_dir results/$DATASET/$MODEL/projection_offline \
        --max_tokens 16384 \
        --dataset $DATASET \
        --steer_file_path $STEER_FILE \
        --step_size 1.0 \
        --remove_bos \
        --trigger \
        --method projection-offline \
        --steering_layer 20 \
        --num_gen 3
done
