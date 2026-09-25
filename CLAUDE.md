# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Official code for **Efficient Reasoning via Constrained Optimization in Latent
Space** (NeurIPS 2026). The method steers a model's reasoning by constraining
its hidden state to a feasible region in a PCA-reduced latent space, where the
region is defined by per-cluster linear SVMs. The intervention is a quadratic
program solved either online per token or offline as a multi-parametric QP.

Pipeline:

1. **Dataset Generation**: Sample paired "direct" vs "vanilla" chain-of-thought
   traces on MATH train (`generate_dataset_vllm.py`).
2. **Hidden State Analysis**: Extract hidden states at a target layer, PCA,
   K-means clustering, and one linear SVC per cluster; write the QP parameters
   (`hidden_analysis.py`).
3. **Offline Conversion** (optional): Solve the projection QP as a
   multi-parametric QP and store its critical regions (`qp_offline.py`).
4. **Steering**: Apply the projection at inference with vLLM's
   `SteerVectorRequest` (`projection_vllm.py`).
5. **Evaluation**: Accuracy on MATH500, GSM8K, AIME, AMC23, GPQA
   (`evaluator.py`).

## Common Commands

### Environment Setup

The steering path needs a **fork of vLLM** that implements steer vectors. It is
not vendored — it is shipped as a patch. Apply it before installing:

```bash
git clone https://github.com/ZJU-REAL/EasySteer
cd EasySteer

# The patch adds the two constrained-projection algorithms under the
# vllm-steer submodule, hence the vllm-steer/ prefix in its paths.
git apply -p1 /path/to/Opt4Reasoning/patches/easysteer-projection-algorithms.patch

# Install per EasySteer's own instructions, in editable mode
pip install -e .

cd /path/to/Opt4Reasoning
pip install -r requirements.txt
```

Verify registration:

```bash
python -c "
from vllm.steer_vectors.algorithms import ProjectionAlgorithm, ProjectionOFFAlgorithm
from vllm.steer_vectors.algorithms.factory import ALGORITHM_REGISTRY
print('projection' in ALGORITHM_REGISTRY, 'projection-offline' in ALGORITHM_REGISTRY)
"
# -> should print: True True
```

### Pipeline Execution

**1 + 2. Build steering parameters from scratch** (slow; skip to reproduce results):

```bash
bash scripts/preprocess.sh 0 DeepSeek-R1-Distill-Qwen-1.5B
```

**3. Convert online JSON to offline `.pt`:**

```bash
python qp_offline.py -i params/online/<MODEL>/qp_params_*.json -o params/offline/<MODEL>/qp_params_*.pt
```

**4. Run steering experiments** (scripts take `<gpu_id> [model_name]`):

```bash
bash scripts/run_vanilla.sh 0 DeepSeek-R1-Distill-Qwen-1.5B   # no steering
bash scripts/run.sh 0 DeepSeek-R1-Distill-Qwen-1.5B           # --method projection
bash scripts/run_offline.sh 0 DeepSeek-R1-Distill-Qwen-1.5B   # --method projection-offline
```

**5. Evaluate:**

```bash
python evaluator.py --file_path results/MATH500/<MODEL>/projection/... --type MATH --save
```

`--type` is `MATH` or `GPQA`, auto-detected from the path when omitted.

## Architecture

### Key Files

- `hidden_analysis.py`: Extracts hidden states, fits PCA + KMeans + per-cluster
  SVCs, writes JSON with `svc_0..svc_{n-1}`, PCA components, and scaler mean.
  This is the **only** producer of the online JSON; it never writes `.pt`.
- `qp_offline.py`: Consumes the online JSON and emits a `.pt` with tensors
  `Ath`, `bth`, `K`, `k`, `pca_comp`, `s_mean` (the critical regions of the
  multi-parametric QP, via `pdaqp.MPQP`).
- `projection_vllm.py`: Applies steering via vLLM's `SteerVectorRequest`.
  Supports `projection` and `projection-offline`. Handles MATH500, MATH*,
  GSM8K, AIME2025/2026, AMC23, GPQA.
- `evaluator.py`: Answer extraction and verification.
- `projection_analysis.py`: Correlates latent-space distance to the feasible
  region with response length. Note it joins `"params"` onto `--param_file`
  internally — do not pass a `params/` prefix.

### Data Flow

1. **Input**: `data/{MATH,MATH500,gsm,AIME2025,AIME2026,AMC23,GPQA}/` — tracked
   in git. `data/livecodebench/` is deliberately gitignored and unused.
2. **Intermediate**: `results/COT/{MODEL}/Projection/0_N_gen_M/thinking_results.json`
   (`{"vanilla": [...], "direct": [...]}`, each item tagged `isCorrect`).
3. **Parameters**: `params/online/{MODEL}/*.json`, `params/offline/{MODEL}/*.pt`.
4. **Outputs**: `results/{DATASET}/{MODEL}/{method}/{steer_file}/.../predictions.jsonl`.

### Important Parameters

- `--target_layer` / `--steering_layer`: layer to intervene (20 for the
  released DeepSeek-R1-Distill parameters).
- `--latent_dim`: PCA dimension. `--n_clusters`: K-means clusters.
- `--coverage`: percentile for the SVC intercept adjustment.
- `--step_size`: steering strength.
- `--trigger`: use newline trigger tokens. (Only affects the output directory
  name; the trigger token list is always passed to `SteerVectorRequest`.)
- `--remove_bos`: declared but unused.

## Directory Structure

```
data/                   # Benchmark splits used in the paper (tracked)
patches/                # The steering algorithms added to the vLLM fork
params/online/          # JSON QP parameters (solved per token)
params/offline/         # Precomputed critical regions (lookup per token)
scripts/                # Shell entry points for each pipeline stage
eval_math_rule/         # Math answer parsing and verification (tracked, imported by evaluator.py)
results/                # Experiment outputs (gitignored)
archive/                # Deprecated code, ignore during analysis
EasySteer/              # (External, gitignored) upstream fork for local development
code_evaluation/        # LiveCodeBench harness — unused, gitignored
```

## Development Notes

- **GPU Management**: Scripts take the GPU id as the first argument.
- **Model Paths**: `~/model/{MODEL_NAME}`. `evaluator.py` infers the tokenizer
  from the model name in the output path and expands `~`.
- **`archive/`** contains deprecated code and should be ignored during analysis.
- **Adding a dataset**: add loading in `projection_vllm.py` and evaluation in
  `evaluator.py` (plus a `--type` branch if it is not math-like).
