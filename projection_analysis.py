import json
import numpy as np
import pandas as pd
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange, tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cvxpy as cp
import math
from scipy import stats

def main(args):

    with open(os.path.join("params", args.param_file), "r", encoding="utf-8") as f:
        qp_params = json.load(f)

    with open(args.dataset_dir, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    vanilla_cot = loaded_data["vanilla"]

    if args.max_examples:
        test_data = vanilla_cot[:args.max_examples]
    else:
        test_data = vanilla_cot

    if not args.dataset_preprocessed:

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # split_str = ["\n\n", "\n\n\n", ".\n\n", ")\n\n", " \n\n", "!\n\n", "?\n\n", "]\n\n", ").\n\n"]
        # target_token_ids = tokenizer(split_str, add_special_tokens=False)["input_ids"]
        # target_token_ids = [item[0] for item in target_token_ids]
        vocab = tokenizer.get_vocab()
        target_token_ids = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]
        
        think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[-1]
        all_hidden_states = []
        source_outputs = []
        source_token_counts = []


        temp_outputs = [res["response"] for res in test_data]

        print(f"Start to compute hidden states in Layer {args.projection_layer}, Batch Size: {args.batch_size} from {len(temp_outputs)} samples")

        for i in tqdm(range(0, len(temp_outputs), args.batch_size), desc="Extracting Features"):
            batch_texts = temp_outputs[i : i + args.batch_size]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            with torch.no_grad():
                output = model(**inputs, output_hidden_states=True)
                layer_states = output.hidden_states[args.projection_layer]

            for b in range(input_ids.shape[0]):
                valid_ids = input_ids[b][attention_mask[b] == 1]
                think_indices = (valid_ids == think_token_id).nonzero(as_tuple=True)[0]
                if think_indices.numel() > 0:
                    first_think_pos = think_indices[0].item()
                    count_after_think = len(valid_ids) - (first_think_pos + 1)
                    source_token_counts.append(count_after_think)
                else:
                    source_token_counts.append(0)

                indices = [idx for idx, token_id in enumerate(input_ids[b]) if token_id.item() in target_token_ids]

                for index, indice in enumerate(indices):
                    vec = layer_states[b, indice, :].detach().cpu()
                    all_hidden_states.append(vec)
                    source_outputs.append(i + b)

            del inputs, input_ids, output, layer_states
            torch.cuda.empty_cache()

        if all_hidden_states:
            final_features = torch.stack(all_hidden_states)
            print(f"hidden state shape: {final_features.shape}")
    
        X_latent = final_features.numpy()

        if args.save_preprocessed:
            with open(os.path.join(os.path.dirname(args.dataset_dir), f"thoughts_layer{args.projection_layer}_sample{args.max_examples}.json"), "w", encoding="utf-8") as f:
                json.dump({"X": X_latent.tolist(), "source_outputs": source_outputs, "source_token_counts": source_token_counts}, f, ensure_ascii=False, indent=4)
            print(f"save thoughts dataset in {os.path.join(os.path.dirname(args.dataset_dir), f'thoughts_layer{args.projection_layer}_sample{args.max_examples}.json')}")

    else:
        with open(args.dataset_preprocessed, "r") as f:
            dataset_preprocessed= json.load(f)
        X_latent = np.array(dataset_preprocessed["X"])
        source_outputs = dataset_preprocessed["source_outputs"]
        source_token_counts = dataset_preprocessed["source_token_counts"]
    
    X_latent_pca = (X_latent - qp_params["scaler"]["mean"]) @ np.array(qp_params["pca"]["components"]).T

    index2dis = {}
    index2num = {}

    x_dim = X_latent_pca.shape[1]
    x_var = cp.Variable(x_dim)
    target_param = cp.Parameter(x_dim) 

    constraints = []
    for key, value in qp_params.items():
        if "svc" in key:
            constraints.append(value["coef"] @ x_var + value["intercept"] <= 0)

    prob_obj = cp.Minimize(cp.sum_squares(x_var - target_param))
    prob = cp.Problem(prob_obj, constraints)
    for i in tqdm(range(X_latent_pca.shape[0])):
        target_param.value = X_latent_pca[i, :] 
        prob.solve(solver=cp.OSQP, warm_start=True) 
        index2dis[source_outputs[i]] = index2dis.get(source_outputs[i], 0) + math.sqrt(prob.value)
        index2num[source_outputs[i]] = index2num.get(source_outputs[i], 0) + 1

    index2avgdis = {}
    for k, v in index2dis.items():
        index2avgdis[k] = v / index2num[k]

    x = np.array(list(index2avgdis.values()))
    y1 = np.array([len(res["response"].split("<think>")[-1]) for res in test_data])
    y2 = np.array(source_token_counts)

    res1 = stats.spearmanr(x, y1)
    res2 = stats.spearmanr(x, y2)

    print(f"Spearman coefficient in distance - response length: {res1.correlation:.4f}, P-value: {res1.pvalue:.4e}")
    print(f"Spearman coefficient in distance - source token count: {res2.correlation:.4f}, P-value: {res2.pvalue:.4e}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.save_dir:
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y1, alpha=0.6, s=15, color='tab:blue')
        plt.xlabel("Average Distance")
        plt.ylabel("Response Length")
        save_path1 = os.path.join(args.save_dir, "distance_vs_response_length.png")
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"Saved Figure distance_vs_response_length.png in {save_path1}")
        plt.close() 

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y2, alpha=0.6, s=15, color='tab:orange')
        plt.xlabel("Average Distance")
        plt.ylabel("Source Token Count")
        save_path2 = os.path.join(args.save_dir, "distance_vs_token_count.png")
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"Saved Figure distance_vs_token_count.png in {save_path2}")
        plt.close()

        save_file_path = os.path.join(args.save_dir, "statistic_analysis.json")
        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump({
                "spearman_distance_vs_response_length": res1.correlation,
                "spearman_distance_vs_token_count": res2.correlation,
                "dataset_dir": args.dataset_dir,
                "projection_layer": args.projection_layer,
                "num_samples": len(test_data)
            }, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/projection"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--param_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_preprocessed",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_preprocessed",
        action="store_true"
    )
    parser.add_argument(
        "--projection_layer",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    path_suffix = f"{args.param_file.removesuffix('.json')}"
    args.save_dir = os.path.join(args.save_dir, path_suffix)

    args.save_dir = os.path.join(args.save_dir, f"layer{args.projection_layer}_samples{args.max_examples}")


    print(args.save_dir)

    main(args)
