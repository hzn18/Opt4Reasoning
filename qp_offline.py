import json
import torch
import numpy as np
from pdaqp import MPQP
import os
import argparse

# pdaqp runs through JuliaCall; leave signal handling to the host process.
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"

def precompute_and_save(json_path, output_pt_path, big_m=10000.0):
    print(f"Loading configuration from: {json_path}")
    with open(json_path, 'rb') as f:
        qp_params = json.load(f)
    
    A_list = []
    b_list = []
    for k, v in qp_params.items():
        if "svc" in k:
            # Match the CVXPY constraint: coef @ x + intercept <= 0
            #                          =>  coef @ x <= -intercept
            A_list.append(np.array(v["coef"]).flatten())
            b_list.append(-v["intercept"])  # note the sign flip
            
    A = np.array(A_list)
    b = np.array(b_list)

    c_dim, n_dim = A.shape
    
    # Match the CVXPY objective: sum_squares(x - target_pca).
    # theta is the parameter (the projected hidden state) and x the variable, so
    # F must be the negative identity for x to track theta.
    H = np.eye(n_dim)
    f = np.zeros((n_dim, 1))
    F = -np.eye(n_dim)
    B = np.zeros((c_dim, n_dim))

    # The parameter space must be bounded; a large box stands in for R^n.
    thmin = -big_m * np.ones(n_dim)
    thmax = big_m * np.ones(n_dim)

    print(f"Solving mpQP offline with BIG_M={big_m}. This may take a moment...")
    mpQP = MPQP(H, f, F, A, b, B, thmin, thmax)  # b already has the sign flipped
    mpQP.solve()
    
    CRs = mpQP.CRs
    n_regions = len(CRs)
    print(f"mpQP solved. Total critical regions found: {n_regions}")

    # Critical regions have different numbers of facets; pad to the widest one
    # so they can be stacked into a single tensor for the GPU lookup.
    max_constraints = max(cr.Ath.shape[0] for cr in CRs)

    A_tensor = torch.zeros((n_regions, max_constraints, n_dim), dtype=torch.float32)
    b_tensor = torch.full((n_regions, max_constraints), float('inf'), dtype=torch.float32)
    K_tensor = torch.zeros((n_regions, n_dim, n_dim), dtype=torch.float32)
    k_tensor = torch.zeros((n_regions, n_dim), dtype=torch.float32)

    for i, cr in enumerate(CRs):
        c_i = cr.Ath.shape[0]
        A_tensor[i, :c_i, :] = torch.from_numpy(cr.Ath).float()
        b_tensor[i, :c_i] = torch.from_numpy(cr.bth).float()
        
        K_tensor[i] = torch.from_numpy(cr.z[:, :-1]).float()
        k_tensor[i] = torch.from_numpy(cr.z[:, -1]).float()
    
    torch.save({
        "Ath": A_tensor.cpu(),
        "bth": b_tensor.cpu(),
        "K": K_tensor.cpu(),
        "k": k_tensor.cpu(),
        "pca_comp": torch.tensor(qp_params["pca"]["components"]).float().cpu(),
        "s_mean": torch.tensor(qp_params["scaler"]["mean"]).float().cpu()
    }, output_pt_path)
    print(f"Saved successfully to {output_pt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute mpQP explicit solution and save as PyTorch tensors.")

    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True, 
        help="Path to the input JSON configuration file (e.g., params/.../qp_params.json)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="mpqp_cache.pt", 
        help="Path to save the output PyTorch tensor file (.pt). Default is 'mpqp_cache.pt'"
    )
    parser.add_argument(
        "--big_m", 
        type=float, 
        default=10000.0, 
        help="Bounding box size for the parameter space. Default is 10000.0"
    )

    args = parser.parse_args()

    precompute_and_save(args.input, args.output, args.big_m)