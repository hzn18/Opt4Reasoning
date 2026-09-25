#!/bin/bash
# Online constrained-projection steering (`--method projection`).
# The QP is re-solved on the fly with cvxpylayers using the per-cluster SVC
# constraints stored in the JSON parameter file produced by hidden_analysis.py.
#
# Usage: bash scripts/run.sh <gpu_id> [model_name]
set -e

gpu=${1:-0}
MODEL=${2:-"DeepSeek-R1-Distill-Qwen-1.5B"}
DATASETS=("AMC23" "AIME2025" "GSM8K" "MATH500" "GPQA")

STEER_FILE="params/online/$MODEL/qp_params_dim10_90_layer20_cluster8_noscaler.json"

echo "🚀 Online projection steering: $MODEL on GPU $gpu"
echo "   Steering file: $STEER_FILE"

for DATASET in "${DATASETS[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python projection_vllm.py \
        --model_name_or_path ~/model/$MODEL \
        --save_dir results/$DATASET/$MODEL/projection \
        --max_tokens 16384 \
        --dataset $DATASET \
        --steer_file_path $STEER_FILE \
        --step_size 1.0 \
        --remove_bos \
        --trigger \
        --method projection \
        --steering_layer 20 \
        --num_gen 3
done
