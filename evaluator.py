import os
import json
import argparse
import numpy as np
from transformers import AutoTokenizer
from eval_math_rule.evaluation.parser import extract_answer
from math_verify import parse, verify
from tqdm import tqdm
import regex as re

def _clean_extracted_text(text: str) -> str:
    """
    Strip LaTeX formatting, units, and digit-group separators from an answer
    string so that it can be compared numerically.
    """
    # 1. LaTeX spacing and formatting commands: \, \! \enspace \quad \qquad ...
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # 2. Stray backslashes, e.g. a model writing \129,200
    text = text.replace("\\", "")
    # 3. Everything that is not a digit, separator, or sign: $, %, emoji, ...
    text = re.sub(r"[^\d.,\-+]", "", text)
    return text


def _extract_latex_braced_command_args(text: str, command: str) -> list[str]:
    """
    Extract braced arguments for a LaTeX command, handling nested braces.

    Example: r"\\boxed{\\frac{1}{2}}" -> ["\\frac{1}{2}"]
    """
    needle = "\\" + command
    out: list[str] = []
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        while k < len(text) and text[k].isspace():
            k += 1
        if k >= len(text) or text[k] != "{":
            i = j + len(needle)
            continue
        depth = 0
        start = None
        end = None
        for t in range(k, len(text)):
            ch = text[t]
            if ch == "{":
                depth += 1
                if depth == 1:
                    start = t + 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    end = t
                    break
        if start is not None and end is not None and end >= start:
            out.append(text[start:end])
            i = end + 1
        else:
            i = k + 1
    return out

_RE_HASH_ANSWER_NUM = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE | re.MULTILINE)
_RE_LAST_NUMBER = re.compile(r"([-+]?\d[\d,]*\.?\d*)")

def extract_GSM8K_answer(model_output):
    if not model_output:
        return None
    boxed = _extract_latex_braced_command_args(model_output, "boxed")
    if boxed:
        model_output = _clean_extracted_text(boxed[-1])
        # return boxed[-1]
    match = _RE_HASH_ANSWER_NUM.search(model_output)
    if match:
        return match.group(1).replace(",", "")
    matches = _RE_LAST_NUMBER.findall(model_output)
    if matches:
        return matches[-1].replace(",", "")
    return None

def extract_gpqa_answer(model_output):
    if not model_output:
        return None
    boxed = _extract_latex_braced_command_args(model_output, "boxed")
    if boxed:
        return boxed[-1]
    markers = [
        r"ANSWER:\s*([A-D])", 
        r"Final\s*Answer:\s*([A-D])",
        r"correct\s*option\s*is\s*([A-D])",
        r"correct\s*choice\s*is\s*([A-D])"
    ]
    for pattern in markers:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    bracket_pattern = r"[\(\[\{]([A-D])[\)\]\}]"
    matches = re.findall(bracket_pattern, model_output)
    if matches:
        return matches[-1].upper()  # the last one is the final answer
    last_paragraph = model_output.strip().split('\n')[-1]
    last_letter_match = re.search(r"\b([A-D])\b\.?$", last_paragraph.strip())
    if last_letter_match:
        return last_letter_match.group(1).upper()
    return None

LABEL_PATTERNS = [
    re.compile(r"answer:\s*(yes|no)", re.IGNORECASE),
    re.compile(r"final\s*answer:\s*(yes|no)", re.IGNORECASE),
    re.compile(r"conclusion:\s*(yes|no)", re.IGNORECASE),
    re.compile(r"is\s*it\s*.*\?\s*(yes|no)", re.IGNORECASE)
]

GLOBAL_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
STRICT_END_PATTERN = re.compile(r"(yes|no)[\.\!\?]*$", re.IGNORECASE)

def extract_strategyqa_answer(model_output):
    if not model_output:
        return None
    text = model_output.strip()
    for pattern in LABEL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            last_val = matches[-1].lower()
            return "Yes" if last_val == "yes" else "No"
    all_matches = GLOBAL_PATTERN.findall(text)
    if all_matches:
        last_val = all_matches[-1].lower()
        return "Yes" if last_val == "yes" else "No"
    end_match = STRICT_END_PATTERN.search(text)
    if end_match:
        return "Yes" if end_match.group(1).lower() == "yes" else "No"
    return None


def eval_math(result_dir, tokenizer_name_or_path, max_examples=None, save=False, output_dir=None):
    results = []
    with open(result_dir, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    if max_examples:
        results = results[:max_examples]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    corrects = []
    tokens = []
    lengths = []

    for i, example in enumerate(tqdm(results, desc="Evaluate reasoning results")):
        if "GSM8K" in result_dir:
            all_pred = [extract_GSM8K_answer(p.split("</think>")[-1]) for p in example["model_generation"]]
        else:
            all_pred = [extract_answer(p.split("</think>")[-1], data_name="omni-math") for p in example["model_generation"]]
        all_eval = [verify(parse(f"${pred}$"), parse(f"${example['answer']}$")) for pred in all_pred]
        corrects.append(all_eval)
        token_ids = tokenizer(example["model_generation"], add_special_tokens=False)["input_ids"]
        token_lens = [len(token_id) for token_id in token_ids]
        tokens.append(token_lens)
        lengths.append([len(response) for response in example["model_generation"]])

    mean_acc = np.mean(corrects, axis = 0).tolist()
    mean_tokens = np.mean(tokens, axis = 0).tolist()
    mean_lens = np.mean(lengths, axis = 0).tolist()

    print(f"Accuracy: {mean_acc}, Token Length: {mean_tokens}, Response Length: {mean_lens}")
    
    print(f"Mean Accuracy: {float(np.mean(mean_acc))}, Mean Token Length: {float(np.mean(mean_tokens))}, Mean Response Length: {float(np.mean(mean_lens))}")

    if save:
        metric_file = os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": mean_acc, "tokens": mean_tokens, "lengths": mean_lens}, f)

def eval_gpqa(result_dir, tokenizer_name_or_path, max_examples=None, save=False, output_dir=None):
    results = []
    with open(result_dir, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    if max_examples:
        results = results[:max_examples]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    corrects = []
    tokens = []
    lengths = []

    for i, example in enumerate(tqdm(results, desc="Evaluate reasoning results")):
        if "gpqa" in result_dir.lower():
            all_pred = [extract_gpqa_answer(p.split("</think>")[-1]) for p in example["model_generation"]]
        else:
            all_pred = [extract_strategyqa_answer(p.split("</think>")[-1]) for p in example["model_generation"]]
        all_eval = [(pred == example['answer']) for pred in all_pred]
        # print(all_pred, example['answer'])
        corrects.append(all_eval)
        token_ids = tokenizer(example["model_generation"], add_special_tokens=False)["input_ids"]
        token_lens = [len(token_id) for token_id in token_ids]
        tokens.append(token_lens)
        lengths.append([len(response) for response in example["model_generation"]])

    mean_acc = np.mean(corrects, axis = 0).tolist()
    mean_tokens = np.mean(tokens, axis = 0).tolist()
    mean_lens = np.mean(lengths, axis = 0).tolist()

    print(f"Accuracy: {mean_acc}, Token Length: {mean_tokens}, Response Length: {mean_lens}")

    print(f"Mean Accuracy: {float(np.mean(mean_acc))}, Mean Token Length: {float(np.mean(mean_tokens))}, Mean Response Length: {float(np.mean(mean_lens))}")

    if save:
        metric_file = os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": mean_acc, "tokens": mean_tokens, "lengths": mean_lens}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-i",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help=(
            "Tokenizer used to count response tokens. When omitted it is "
            "inferred from the model name in --file_path, falling back to "
            "~/model/DeepSeek-R1-Distill-Qwen-1.5B."
        ),
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None
    )
    args = parser.parse_args()
    # Determine result_path and output_dir based on file_path
    
    if os.path.isdir(args.file_path):
        result_path = os.path.join(args.file_path, 'predictions.jsonl')
        output_dir = args.file_path
    else:
        result_path = args.file_path
        output_dir = os.path.dirname(args.file_path)

    print(args.file_path)

    if not args.tokenizer_name_or_path:
        tokenizer_name_list = ["Qwen3.5-4B", "Qwen3-4B", "DeepSeek-R1-Distill-Qwen-1.5B"]
        for tokenizer_name in tokenizer_name_list:
            if tokenizer_name in args.file_path:
                args.tokenizer_name_or_path = os.path.expanduser(f"~/model/{tokenizer_name}")
                break
        else:
            args.tokenizer_name_or_path = os.path.expanduser(
                "~/model/DeepSeek-R1-Distill-Qwen-1.5B"
            )

    if not args.type:
        math_key = ["MATH", "AMC", "AIME", "GSM8K"]
        qp_key = ["GPQA"]
        if any(key.lower() in args.file_path.lower() for key in math_key):
            args.type = "MATH"
        elif any(key.lower() in args.file_path.lower() for key in qp_key):
            args.type = "GPQA"

    if args.type == "GPQA":
        eval_gpqa(result_path, args.tokenizer_name_or_path, args.max_examples, args.save, output_dir)
    else:
        eval_math(result_path, args.tokenizer_name_or_path, args.max_examples, args.save, output_dir)