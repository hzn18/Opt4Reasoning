import os, re, json
import torch
from tqdm import tqdm
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import argparse

def main(args):
    test_data = []
    if args.dataset == "MATH500":
        with open("data/MATH500/test.jsonl") as fin:
            for line in fin:
                example = json.loads(line)
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt": example["answer"],
                })
    elif  "MATH500" in args.dataset:
        with open(f"data/{args.dataset}/test.json") as fin:
            examples = json.load(fin)
            print(examples[0])
            for example in examples:    
                test_data.append({
                        "question": example["problem"],
                        "answer": example["answer"],
                        "gt": example["answer"],
                    })
    elif args.dataset == "GSM8K":
        with open("data/gsm/test.jsonl") as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer":example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    elif args.dataset in ["AIME2025", "AIME2026", "AMC23"]:
        with open(f"data/{args.dataset}/test.jsonl") as fin:
            for line in fin:
                example = json.loads(line)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"],
                    "gt": example["answer"],
                })
    elif args.dataset in ["GPQA"]:
        with open(f"data/{args.dataset}/test.jsonl") as fin:
            for line in fin:
                example = json.loads(line)
                test_data.append({
                    "question": example["problem"],
                    "answer": example["answer"],
                    "gt": example["answer"],
                })
    else:
        raise ValueError("Unsupported dataset")

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    llm = LLM(model=args.model_name_or_path, enable_steer_vector=True, enforce_eager=True, tensor_parallel_size=torch.cuda.device_count(), enable_chunked_prefill=False)

    tokenizer = llm.get_tokenizer()
    tokenizer.padding_side = "left"

    
    prompts = []
    for i, example in enumerate(test_data):
        prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
        messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)

    trigger_list = ["\n\n", "\n\n\n", ".\n\n", ")\n\n", " \n\n", "!\n\n", "?\n\n", "]\n\n", ").\n\n"] 
    trigger_token_list = tokenizer(trigger_list, add_special_tokens=False).input_ids
    trigger_token_list = [token[0] for token in trigger_token_list if len(token) == 1]
    print(f"Trigger tokens: {trigger_token_list}")

    if args.method == "projection":
        reason_request = SteerVectorRequest("efficient", 1, steer_vector_local_path=args.steer_file_path, scale=args.step_size, target_layers=[args.steering_layer - 1], prefill_trigger_positions=[-1], generate_trigger_tokens=trigger_token_list, algorithm='projection')
        # reason_request = SteerVectorRequest("efficient", 1, steer_vector_local_path=args.steer_file_path, scale=args.step_size, target_layers=[args.steering_layer], prefill_trigger_tokens=trigger_token_list, generate_trigger_tokens=trigger_token_list, algorithm='projection')
    elif args.method == "projection-offline":
        reason_request = SteerVectorRequest("efficient", 1, steer_vector_local_path=args.steer_file_path, scale=args.step_size, target_layers=[args.steering_layer - 1], prefill_trigger_positions=[-1], generate_trigger_tokens=trigger_token_list, algorithm='projection-offline')

    print(f"Load steering params from {args.steer_file_path}")

    if args.deterministic:
        sampling_params = SamplingParams(
            n=1,             
            temperature=0.0,               
            max_tokens=args.max_tokens  
        )
    else:
        if "Qwen3.5" in args.model_name_or_path:
            sampling_params = SamplingParams(
                n=args.num_gen,             
                temperature=1.0,            
                top_p=0.95,                 
                top_k=20,                   
                presence_penalty=1.05,       
                max_tokens=args.max_tokens  
            )
        else:
            sampling_params = SamplingParams(
                n=args.num_gen,
                temperature=0.6,   
                top_p=0.95,     
                max_tokens=args.max_tokens
            )

    if args.method in ["projection", "projection-offline"]:
        outputs = llm.generate(prompts=prompts, steer_vector_request=reason_request, sampling_params=sampling_params)
    else:
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        result.append(attempts)

    os.makedirs(args.save_dir, exist_ok=True)

    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution":  example["answer"],
        "model_generation": output,
    } for example, output, prompt in zip(test_data, result, prompts)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    print(f"Save results in {os.path.join(args.save_dir, 'predictions.jsonl')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH500",
    )
    parser.add_argument(
        "--steer_file_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )   
    parser.add_argument(
        "--trigger",
        action="store_true",
    )  
    parser.add_argument(
        "--method",
        type=str,
        default=None,
    )  
    parser.add_argument(
        "--num_gen",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    if args.steer_file_path:
        path_suffix, ext = os.path.splitext(os.path.basename(args.steer_file_path))
    else:
        path_suffix = "vanilla"

    if args.max_examples:
        args.save_dir = os.path.join(args.save_dir, path_suffix, f"{0}_{args.max_examples}")
    else:
        args.save_dir = os.path.join(args.save_dir, path_suffix, "total")

    args.save_dir += f"_{args.step_size}"

    if args.trigger:
        args.save_dir += "_trigger"

    if args.deterministic:
        args.save_dir += "_deterministic"

    if not args.method:
        args.save_dir += "_notproj"
    else:
        args.save_dir += f"_{args.method}"

    if args.steering_layer:
        args.save_dir += f"_layer{args.steering_layer}"
    else:
        args.save_dir += "_layerall"

    args.save_dir += f"_{args.num_gen}_{args.max_tokens}"

    if os.path.exists(args.save_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"{args.save_dir.rstrip('/')}_{timestamp}"

    print(args.save_dir)

    main(args)