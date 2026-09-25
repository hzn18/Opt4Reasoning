#!/bin/bash
# Stage 1 + 2 of the pipeline: generate the paired CoT dataset, then extract
# hidden states and fit PCA / KMeans / per-cluster linear SVCs to obtain the QP
# steering parameters.
#
# Skip this script entirely if you only want to reproduce the results: the
# released parameters live in params/online/ and params/offline/.
#
# Usage: bash scripts/preprocess.sh <gpu_id> [model_name]
set -e

gpu=${1:-0}
MODEL=${2:-"DeepSeek-R1-Distill-Qwen-1.5B"}

DIM=10
CLUSTER=8
LAYER=20
COVER=90

COT_DIR="results/COT/$MODEL/Projection/0_400_gen_3"

echo "🚀 Building steering parameters for $MODEL on GPU $gpu"

# ---------------------------------------------------------------------------
# 1. Generate paired reasoning samples on the MATH training split.
#    Writes $COT_DIR/thinking_results.json with "vanilla" and "direct" rollouts,
#    each tagged by correctness.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$gpu python generate_dataset_vllm.py \
    --model_name_or_path ~/model/$MODEL \
    --save_dir results/COT/$MODEL/Projection \
    --max_examples 400 \
    --num_gen 3 \
    --max_tokens 20000

# ---------------------------------------------------------------------------
# 2. Extract hidden states at --target_layer, reduce with PCA to --latent_dim
#    components, cluster the correct rollouts into --n_clusters groups, and fit
#    one linear SVC per cluster. --coverage sets the percentile used to adjust
#    each SVC intercept so the feasible region covers the requested fraction of
#    samples.
#
#    Output: a JSON file holding svc_0..svc_{n-1} (coef/intercept), the PCA
#    components, and the scaler mean.
#
#    The released params/online/.../qp_params_dim10_90_layer20_cluster8_noscaler.json
#    was produced with exactly these settings; the file written here uses a
#    different name so it does not overwrite it.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$gpu python hidden_analysis.py \
    --model_name_or_path ~/model/$MODEL \
    --save_path params/online/$MODEL/qp_params_dim${DIM}_${COVER}_layer${LAYER}_cluster${CLUSTER}_vllm.json \
    --dataset $COT_DIR/thinking_results.json \
    --num_samples 400 \
    --batch_size 4 \
    --coverage $COVER \
    --latent_dim $DIM \
    --eval \
    --target_layer $LAYER \
    --n_clusters $CLUSTER

echo "✅ Wrote params/online/$MODEL/qp_params_dim${DIM}_${COVER}_layer${LAYER}_cluster${CLUSTER}_vllm.json"
