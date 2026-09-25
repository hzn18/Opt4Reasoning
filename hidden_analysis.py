import json
import numpy as np
import torch
import argparse
import os
# from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans


def filter_dataset(direct_dataset, vanilla_dataset, num_samples):
    direct_slices = []
    vanilla_slices = []
    for i in range(num_samples):
        if 3 * i + 2 < len(direct_dataset):
            temp = [direct_dataset[3 * i], direct_dataset[3 * i + 1], direct_dataset[3 * i + 2]]
            temp = [t for t in temp if t["isCorrect"] and len(t["response"]) < 12000]
            if temp:
                direct_slices.append(temp[0])
        else:
            break
    for i in range(num_samples):
        if 3 * i + 2 < len(vanilla_dataset):
            temp = [vanilla_dataset[3 * i], vanilla_dataset[3 * i + 1], vanilla_dataset[3 * i + 2]]
            temp = [t for t in temp if t["isCorrect"] and len(t["response"]) < 40000]
            if temp:
                vanilla_slices.append(temp[0])
    if num_samples:
        direct_slices = direct_slices[:num_samples]
        vanilla_slices = vanilla_slices[:num_samples]
    return direct_slices, vanilla_slices

# def filter_dataset(direct_dataset, vanilla_dataset, num_samples):
#     direct_slices = []
#     vanilla_slices = []

#     direct_slices = [cot for cot in direct_dataset if cot['isCorrect'] and len(cot['response']) < 10000]
#     vanilla_slices = [cot for cot in vanilla_dataset if cot['isCorrect'] and len(cot['response']) < 30000]

#     if num_samples:
#         direct_slices = direct_slices[:num_samples]
#         vanilla_slices = vanilla_slices[:num_samples]
#     return direct_slices, vanilla_slices

# def filter_dataset(direct_dataset, vanilla_dataset, num_samples):
#     direct_slices = direct_dataset
#     vanilla_slices = vanilla_dataset
#     if num_samples:
#         direct_slices = direct_slices[:num_samples]
#         vanilla_slices = vanilla_slices[:num_samples]
#     return direct_slices, vanilla_slices


# def filter_dataset(direct_dataset, vanilla_dataset, num_samples):
#     direct_slices = []
#     vanilla_slices = []
#     for i in range(len(direct_dataset) // 3):
#         if 3 * i + 2 < len(direct_dataset):
#             temp = [direct_dataset[3 * i], direct_dataset[3 * i + 1], direct_dataset[3 * i + 2]]
#             temp = [t for t in temp if t["isCorrect"] and len(t["response"]) < 10000]
#             if temp:
#                 direct_slices.append(temp[0])
#         else:
#             break

#     # direct_slices = direct_dataset
#     for i in range(len(vanilla_dataset) // 3):
#         if 3 * i + 2 < len(vanilla_dataset):
#             temp = [vanilla_dataset[3 * i], vanilla_dataset[3 * i + 1], vanilla_dataset[3 * i + 2]]
#             temp = [t for t in temp if t["isCorrect"]]
#             if temp:
#                 vanilla_slices.append(temp[0])
#     if num_samples:
#         direct_slices = direct_slices[:num_samples*2]
#         vanilla_slices = vanilla_slices[:num_samples]
#     return direct_slices, vanilla_slices


def main(args):

    if not args.dataset_hidden_features:

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map = "auto", dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)


        with open(args.dataset, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        direct_cot = loaded_data["direct"]
        vanilla_cot = loaded_data["vanilla"]

        split_str = ["\n\n", "\n\n\n", ".\n\n", ")\n\n", " \n\n", "!\n\n", "?\n\n", "]\n\n", ").\n\n"] # "):\n\n", ":\n\n"] # , "</think>"]
        target_token_ids = tokenizer(split_str, add_special_tokens=False)["input_ids"]
        target_token_ids = [item[0] for item in target_token_ids]
        
        # vocab = tokenizer.get_vocab()
        # target_token_ids = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]
        
        all_hidden_states = []
        valid_labels = []

        direct_slices, vanilla_slices = filter_dataset(direct_cot, vanilla_cot, args.num_samples)

        temp_outputs = [res["response"] for res in (direct_slices + vanilla_slices)]
        temp_labels = [0] * len(direct_slices) + [1] * len(vanilla_slices)# [1] * len(direct_slice) + [0] * len(vanilla_slice)
        
        tokenizer.padding_side = "left" 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Start Exacting, Batch Size: {args.batch_size} from {len(temp_outputs)} samples")

        for i in tqdm(range(0, len(temp_outputs), args.batch_size), desc="Extracting Features"):
            batch_texts = temp_outputs[i : i + args.batch_size]
            batch_labels = temp_labels[i : i + args.batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                output = model(**inputs, output_hidden_states=True)
                layer_states = output.hidden_states[args.target_layer]
            for b in range(input_ids.shape[0]):
                indices = [idx for idx, token_id in enumerate(input_ids[b]) if token_id.item() in target_token_ids]
                for index, indice in enumerate(indices):
                    vec = layer_states[b, indice, :].detach().cpu()
                    all_hidden_states.append(vec)
                    valid_labels.append(batch_labels[b])
            del inputs, input_ids, output, layer_states
            torch.cuda.empty_cache()

        if all_hidden_states:
            final_features = torch.stack(all_hidden_states)
            final_labels = torch.tensor(valid_labels)
            print(f"feature shape: {final_features.shape}, label shape: {final_labels.shape}")


        X = final_features.numpy()
        y = final_labels.numpy()

        if args.save_dataset:
            with open(os.path.join(os.path.dirname(args.dataset), f"hidden_features_layer{args.target_layer}_sample{args.num_samples}.json"), "w", encoding="utf-8") as f:
                json.dump({"X": X.tolist(), "y": y.tolist()}, f, ensure_ascii=False, indent=4)
            print(f"save hidden features in {os.path.join(os.path.dirname(args.dataset), f'hidden_features_layer{args.target_layer}_sample{args.num_samples}.json')}")
    else:
        with open(args.dataset_hidden_features, "r") as f:
            all_hidden_states = json.load(f)
        X = np.array(all_hidden_states["X"])
        y = np.array(all_hidden_states["y"])
        # X = np.concatenate([X[y==1][:args.vanilla_num], X[y==0][:args.direct_num]])
        # y = np.concatenate([y[y==1][:args.vanilla_num], y[y==0][:args.direct_num]])


    print(f"Direct thought {(y == 0).sum()}, Vanilla thought {(y == 1).sum()}")

    if args.scaler:
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(X) 
    else:
        # X_mean = np.mean(X[y == 1], axis=0)
        X_mean = np.mean(X, axis=0)
        X_transformed = X - X_mean

    # pca = PCA(n_components=args.latent_dim)
    # pca.fit(X_transformed[y == 1])
    # X_pca = pca.transform(X_transformed)
    # X_raw_pca = X_pca[y == 1]

    pca = PCA(n_components=args.latent_dim)
    X_pca = pca.fit_transform(X_transformed)
    X_raw_pca = X_pca[y == 1]

    # pca = PCA(n_components=args.latent_dim)
    # pca.fit(X_transformed[y == 0])
    # X_pca = pca.transform(X_transformed)
    # X_raw_pca = X_pca[y == 0]
    
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_raw_pca)

    svcs = []
    for i in range(args.n_clusters):
        X_sub = np.concatenate((X_pca[y == 0], X_raw_pca[cluster_labels == i]), axis=0)
        y_sub = [0] * sum(y == 0) + [1] * sum(cluster_labels == i)
        # X_sub = np.concatenate((X_pca[y == 1], X_raw_pca[cluster_labels == i]), axis=0)
        # y_sub = [0] * sum(y == 1) + [1] * sum(cluster_labels == i)
        svc = SVC(kernel='linear', C=args.svc_C, random_state=42) 
        svc.fit(X_sub, y_sub)
        svcs.append(svc)

    intercept_modified = []
    for i, svc in enumerate(svcs):
        intercept_bound1 = np.percentile(X_pca[y == 0] @ svc.coef_[0], args.coverage)
        intercept_bound2 = svc.coef_[0] @ np.mean(X_pca[y == 0], axis = 0)
        # intercept_bound1 = np.percentile(X_pca[y == 1] @ svc.coef_[0], args.coverage)
        # intercept_bound2 = svc.coef_[0] @ np.mean(X_pca[y == 1], axis = 0)
        print(svc.intercept_, intercept_bound1, intercept_bound2)
        intercept_modified.append(-float(max(intercept_bound1, intercept_bound2)))

    if args.eval:
        out_region = 0
        X_direct_pca = X_pca[y == 0]
        # X_direct_pca = X_pca[y == 1]
        for i in range(X_direct_pca.shape[0]):
            for j, svc in enumerate(svcs):
                decision_values = X_direct_pca[i, :] @ svcs[j].coef_[0] + intercept_modified[j]
                if decision_values > 0.0:
                    out_region += 1
                    break
        print(f"Out region ratio: {out_region / X_direct_pca.shape[0]:.4f} ({out_region} vs {X_direct_pca.shape[0]})")

    data_to_save = {}

    for i, svc in enumerate(svcs):
        key = f"svc_{i}"
        data_to_save[key] = {
            "coef": svc.coef_[0].tolist(),
            "intercept": float(intercept_modified[i])
        }

    data_to_save["pca"] = {
        "components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
    }

    if args.scaler:
        data_to_save["scaler"] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }
    else:
        data_to_save["scaler"] = {
            "mean": X_mean.tolist()
        }

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"Save qp params in {args.save_path}!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/gsm"
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
        "--dataset",
        type=str,
        default="results/Projection/thinking_results.json",
    )
    parser.add_argument(
        "--dataset_hidden_features",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--scaler",
        action="store_true"
    )
    parser.add_argument(
        "--eval",
        action="store_true"
    )
    parser.add_argument(
        "--svc_C",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--direct_num",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--vanilla_num",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--save_dataset",
        action="store_true"
    )

    args = parser.parse_args()
    main(args)