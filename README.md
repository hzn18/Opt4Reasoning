# Efficient Reasoning via Constrained Optimization in Latent Space

Official code for the NeurIPS 2026 paper *Efficient Reasoning via Constrained
Optimization in Latent Space*.

We steer a language model's reasoning behaviour by constraining its hidden
state to a feasible region in a PCA-reduced latent space. The region is
defined by a bank of linear SVMs, one per cluster of reasoning traces, and the
intervention at each decoding step is the *projection* of the current hidden
state onto that region. Because the projection is a small quadratic program,
it can either be solved online per token, or solved once offline as a
multi-parametric QP whose critical regions are evaluated by a lookup at
inference time.

## Method

The pipeline has four stages:

```
data/MATH/train.jsonl
        │
        │  1. generate_dataset_vllm.py
        ▼
paired CoT rollouts  (results/COT/<model>/.../thinking_results.json)
        │
        │  2. hidden_analysis.py
        ▼
QP steering parameters  (params/online/<model>/*.json)
        │
        ├──────── 3a. qp_offline.py ────────►  params/offline/<model>/*.pt
        │                                      (precomputed critical regions)
        │  4. projection_vllm.py
        ▼
steered generations  (results/<dataset>/<model>/.../predictions.jsonl)
        │
        │  5. evaluator.py
        ▼
accuracy
```

1. **Dataset generation.** For each MATH training problem we sample several
   rollouts and keep those that reach the correct answer, labelling them by
   *reasoning style* — the "vanilla" prompt and the "direct" prompt elicit
   long reflective chains and short direct chains respectively.

2. **Hidden-state analysis.** We extract the hidden state at a chosen
   transformer layer on the newline trigger tokens that separate reasoning
   steps, reduce it with PCA to a low-dimensional latent space, cluster the
   correct traces with K-means, and fit one linear SVC per cluster to separate
   the two styles. The SVC hyperplanes, the PCA components, and the scaler
   mean are written to a JSON file.

3. **Steering parameters.** The same JSON can be converted into an offline
   form: `qp_offline.py` solves the projection QP as a multi-parametric QP over
   the latent coordinate and stores every critical region together with its
   affine solution.

4. **Steering at inference.** `projection_vllm.py` runs vLLM with a
   `SteerVectorRequest`. On every trigger token the hidden state is projected
   into PCA space, moved onto the feasible region defined by the SVC
   constraints, and mapped back — scaled by `--step_size`. Two algorithms are
   available:

   | `--method` | Steering file | How the projection is computed |
   | --- | --- | --- |
   | `projection` | `.json` | A `cvxpylayers` QP layer solves the projection for every token at inference time. |
   | `projection-offline` | `.pt` | The QP was solved ahead of time by `qp_offline.py`; inference does a vectorised GPU point-location lookup plus an affine evaluation `z = K·θ + k`. |

## Installation

Steering runs through [EasySteer](https://github.com/ZJU-REAL/EasySteer), which
bundles a vLLM fork with steer-vector support. It is **not vendored here**:
clone EasySteer, apply the patch in [`patches/`](patches/) that adds the two
constrained-projection algorithms, and install it in editable mode. Python 3.10
is what the released parameters were produced with.

```bash
conda create -n Opt4Reasoning python=3.10 -y
conda activate Opt4Reasoning

# 1. This repository's dependencies
git clone https://github.com/hzn18/Opt4Reasoning.git
cd Opt4Reasoning
pip install -r requirements.txt

# 2. EasySteer, with our patch applied
cd ..
git clone https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer
git checkout 83f866d8db701472938a781892702f31fe969e89

# The vLLM fork lives in the vllm-steer submodule, so fetch it first:
# without this the patch has no files to apply to.
git submodule update --init --recursive
cd vllm-steer

# Install with pre-compiled version (recommended)
export VLLM_PRECOMPILED_WHEEL_COMMIT=72506c98349d6bcd32b4e33eec7b5513453c1502
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Install EasySteer
cd ..
pip install --editable .

# Apply from the EasySteer root; the patch paths are prefixed with vllm-steer/
git apply -p1 /path/to/Opt4Reasoning/patches/easysteer-projection-algorithms.patch
```

One caveat on the verification stack: `latex2sympy2` pins
`antlr4-python3-runtime==4.7.2`, while `math-verify` needs `>=4.9.3` and the
generated parser shipped in `eval_math_rule/` was built with `4.9.3`. If your
resolver refuses the combination, install the pieces in the order listed in
`requirements.txt` and finish with:

```bash
pip install --no-deps antlr4-python3-runtime==4.9.3
```

Verify the two algorithms registered:

```bash
python -c "
from vllm.steer_vectors.algorithms import ProjectionAlgorithm, ProjectionOFFAlgorithm
from vllm.steer_vectors.algorithms.factory import ALGORITHM_REGISTRY
print('projection' in ALGORITHM_REGISTRY, 'projection-offline' in ALGORITHM_REGISTRY)
"
# -> should print: True True
```

See [`patches/README.md`](patches/README.md) for what the patch contains and
why it is shipped as a patch rather than a fork.

## Data

The benchmarks used in the paper ship with this repository under `data/`, and
`projection_vllm.py` reads them from fixed paths:

| Path | Contents |
| --- | --- |
| `data/MATH/train.jsonl` | MATH training split, used to fit the steering parameters |
| `data/MATH/test.jsonl` | MATH test split |
| `data/MATH500/test.jsonl` | MATH500 |
| `data/gsm/test.jsonl` | GSM8K |
| `data/AIME2025/test.jsonl` | AIME 2025 |
| `data/AIME2026/test.jsonl` | AIME 2026 |
| `data/AMC23/test.jsonl` | AMC 2023 |
| `data/GPQA/test.jsonl` | GPQA |

Models are expected under `~/model/<model_name>`; every script takes the model
name as an argument and resolves the path itself.

## Quick Start

To reproduce the reported numbers you do **not** need stage 1–3: the released
steering parameters in `params/` can be used directly.

```bash
# Unsteered baseline
bash scripts/run_vanilla.sh 0 DeepSeek-R1-Distill-Qwen-1.5B

# Online constrained projection (solves a QP per token)
bash scripts/run.sh 0 DeepSeek-R1-Distill-Qwen-1.5B

# Offline constrained projection (precomputed critical regions)
bash scripts/run_offline.sh 0 DeepSeek-R1-Distill-Qwen-1.5B

# Score the generations
python evaluator.py --file_path results/MATH500/DeepSeek-R1-Distill-Qwen-1.5B/projection/... --type MATH --save
```

To build steering parameters from scratch instead:

```bash
# 1 + 2: generate rollouts and fit PCA / KMeans / SVCs
bash scripts/preprocess.sh 0 DeepSeek-R1-Distill-Qwen-1.5B

# 3: (optional) precompute the offline multi-parametric QP
python qp_offline.py \
    -i params/online/DeepSeek-R1-Distill-Qwen-1.5B/qp_params_dim10_90_layer20_cluster8_noscaler.json \
    -o params/offline/DeepSeek-R1-Distill-Qwen-1.5B/qp_params_dim12_90_layer20_cluster8_vllm_ablation.pt

# 4: diagnose how far the hidden states sit from the feasible region
bash scripts/analysis.sh 0 DeepSeek-R1-Distill-Qwen-1.5B
```

Individual commands and their flags:

- `generate_dataset_vllm.py` — sample paired reasoning traces on MATH train.
- `hidden_analysis.py` — extract hidden states, fit PCA + KMeans + per-cluster
  SVC, write the QP parameters (`--latent_dim`, `--n_clusters`, `--coverage`,
  `--target_layer`).
- `qp_offline.py` — convert the online JSON into the offline `.pt`.
- `projection_vllm.py` — run inference with steering (`--method`, `--step_size`,
  `--steering_layer`, `--dataset`).
- `evaluator.py` — compute accuracy (`--type MATH` or `--type GPQA`; inferred
  from the path when omitted).
- `projection_analysis.py` — correlate latent-space distance with response
  length.

## Released Steering Parameters

`params/online/` holds the JSON files consumed by `--method projection`;
`params/offline/` holds the `.pt` files consumed by `--method projection-offline`.

| Model | Layer | PCA dim | Clusters | Online (`.json`) | Offline (`.pt`) |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-1.5B | 20 | 10 | 8 | `qp_params_dim10_90_layer20_cluster8_noscaler.json` | `qp_params_dim12_90_layer20_cluster8_vllm_ablation.pt` (dim 12) |
| DeepSeek-R1-Distill-Qwen-7B | 20 | 16 | 16 | `qp_params_dim16_90_layer20_cluster16_sample400_target.json` | `qp_params_dim16_90_layer20_cluster8_sample400_target.pt` (8 clusters) |

The offline file is not a byte-for-byte restatement of the online one — the
dimension and cluster count differ per row above, and the `.pt` files were
produced by `qp_offline.py` runs whose exact online inputs are not all shipped
here. Use them with the `--steering_layer` / `--step_size` in the scripts.

## Repository Structure

```
├── data/                     # benchmark splits (see Data)
├── patches/                  # the steering algorithms added to the vLLM fork
├── params/
│   ├── online/               # JSON QP parameters, solved per token
│   └── offline/              # precomputed critical regions, lookup per token
├── scripts/
│   ├── preprocess.sh         # stage 1 + 2
│   ├── run.sh                # stage 4, --method projection
│   ├── run_offline.sh        # stage 4, --method projection-offline
│   ├── run_vanilla.sh        # stage 4, no steering
│   └── analysis.sh           # latent-space distance diagnostics
├── eval_math_rule/           # math answer parsing and verification
├── evaluator.py              # stage 5
├── generate_dataset_vllm.py  # stage 1
├── hidden_analysis.py        # stage 2
├── qp_offline.py             # stage 3
├── projection_analysis.py    # diagnostics
└── projection_vllm.py        # stage 4
```

## Citation

```bibtex
@inproceedings{opt4reasoning2026efficient,
  title     = {Efficient Reasoning via Constrained Optimization in Latent Space},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
