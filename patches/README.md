# EasySteer patch

This directory contains the only change this repository makes to its inference
backend, [EasySteer](https://github.com/ZJU-REAL/EasySteer) — a toolkit built on
a fork of vLLM with steer-vector support. The toolkit provides the steering
infrastructure; the patch below adds the two **constrained-projection**
algorithms that this paper actually uses.

## Contents

`easysteer-projection-algorithms.patch` adds three things under
`vllm-steer/vllm/steer_vectors/algorithms/`, i.e. inside the `vllm-steer`
submodule of the EasySteer repository:

| File | Class | `--method` value | Steering file format |
| --- | --- | --- | --- |
| `projection.py` | `ProjectionAlgorithm` | `projection` | JSON |
| `projection_light.py` | `ProjectionOFFAlgorithm` | `projection-offline` | `torch.load` (`.pt`) |
| `__init__.py` | — | — | registers the two algorithms above |

Both algorithms project a hidden state onto the feasible set of a
cluster-wise linear SVM in a PCA-reduced latent space, and apply the resulting
displacement scaled by `--step_size`. They differ only in how the constrained
problem is solved:

- **`projection`** (online) builds a `cvxpylayers` layer from the SVC
  coefficients at load time and solves the QP for every token at inference.
  It reads the `.json` files produced by `hidden_analysis.py`.
- **`projection-offline`** reads a `.pt` file produced by `qp_offline.py`, which
  solves the multi-parametric QP ahead of time and stores the critical regions.
  At inference a vectorised GPU point-location lookup plus an affine evaluation
  (`z = K·θ + k`) replaces the per-token solve.

## Applying

Apply the patch **from the root of the EasySteer repository**. The paths in the
patch are prefixed with `vllm-steer/` because the vLLM fork lives in the
`vllm-steer` submodule of that repository, so `git apply` must run one directory
above it.

```bash
git clone https://github.com/ZJU-REAL/EasySteer
cd EasySteer

# The vLLM fork lives in the vllm-steer submodule, so fetch it first
git submodule update --init --recursive

git apply -p1 /path/to/Opt4Reasoning/patches/easysteer-projection-algorithms.patch
```

The patch was generated against commit
`3f8f537fe895b61f734628740632347960fd6fa7` of the `vllm-steer` submodule, and
touches only the three files listed above. If your `vllm-steer` checkout has
moved past that commit, `git apply` will tell you which hunk failed rather than
silently mis-applying — resolve it by hand or use `git apply -3`.

Verify the algorithms registered:

```bash
python -c "
from vllm.steer_vectors.algorithms import ProjectionAlgorithm, ProjectionOFFAlgorithm
from vllm.steer_vectors.algorithms.factory import ALGORITHM_REGISTRY
print('projection' in ALGORITHM_REGISTRY, 'projection-offline' in ALGORITHM_REGISTRY)
"
# -> should print: True True
```

## Note on `__init__.py`

The patched `__init__.py` imports only the two algorithms listed above, which is
sufficient for everything in this repository. If you maintain additional
experimental algorithms in your own working copy (`projection-qpth`,
`rebalance`, …), keep their `from .<module> import ...` lines in `__init__.py` —
otherwise the registry will not see them.
