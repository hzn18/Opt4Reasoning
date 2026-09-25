#!/bin/bash
# Unsteered baseline: no --method and no --steer_file_path, so no intervention
# is applied. Used as the reference point for every comparison.
#
# Usage: bash scripts/run_vanilla.sh <gpu_id> [model_name] [dataset]
set -e

gpu=${1:-0}
MODEL=${2:-"DeepSeek-R1-Distill-Qwen-1.5B"}
DATASET=${3:-"MATH500"}

echo "🚀 Vanilla (unsteered) generation: $MODEL on GPU $gpu, dataset $DATASET"

CUDA_VISIBLE_DEVICES=$gpu python projection_vllm.py \
    --model_name_or_path ~/model/$MODEL \
    --save_dir results/$DATASET/$MODEL/Vanilla \
    --max_tokens 16384 \
    --dataset $DATASET \
    --remove_bos \
    --trigger \
    --steering_layer 20 \
    --num_gen 3
