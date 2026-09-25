import argparse
import os
import json
import numpy as np
from vllm import LLM, SamplingParams
from eval_math_rule.evaluation.parser import extract_answer
from math_verify import parse, verify

def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a

def main(args):
    test_data = []
    with open("data/MATH/train.jsonl") as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt": extract_box(example["solution"]),
            })
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]


    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
        trust_remote_code=True,
        max_model_len=args.max_tokens+2000,
        dtype="bfloat16" 
    )

    if "Qwen3.5" in args.model_name_or_path:
        sampling_params = SamplingParams(
            n=args.num_gen,             
            temperature=1.0,            
            top_p=0.95,                 
            top_k=20,                   
            presence_penalty=1.5,       
            max_tokens=args.max_tokens  
        )
    else:
        sampling_params = SamplingParams(
            n=args.num_gen,
            temperature=0.6,   
            top_p=0.95,     
            max_tokens=args.max_tokens
        )

    vanilla_cot_prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    # direct_cot_prefix = "Given a question, please tell me how to get this answer step by step."
    direct_cot_format = """
        Given a question and its answer, provide a detailed derivation.
        ### Requirements
        1. DO NOT use headers like "Step-by-Step" or other subtitles(e.g. "Step 1").
        2. DO NOT use lists or bullets.
        3. ONLY provide the sequential logic and the necessary math steps in a cohesive flow.
        4. Ensure the derivation is compact and direct.
        5. The result must conclude with the final answer in \boxed{}.
        ### Examples
        \nThe math club has 6 boys and 8 girls, making a total of 14 students. We need to select a team of 6 people without any restrictions. \n\nFirst, we recognize that this is a combination problem because the order in which we select the team members does not matter. The formula for combinations is given by:\n\n\[ C(n, k) = \\frac{n!}{k!(n - k)!} \\]\n\nHere, \( n = 14 \\) and \( k = 6 \\). Plugging these values into the formula, we get:\n\n\[ C(14, 6) = \\frac{14!}{6!(14 - 6)!} = \\frac{14!}{6! \\cdot 8!} \\]\n\nWe simplify the factorials:\n\n\[ C(14, 6) = \\frac{14 \\times 13 \\times 12 \\times 11 \\times 10 \\times 9 \\times 8!}{6! \\times 8!} \\]\n\nThe \( 8! \\) terms cancel out, leaving:\n\n\[ C(14, 6) = \\frac{14 \\times 13 \\times 12 \\times 11 \\times 10 \\times 9}{6!} \\]\n\nCalculating \( 6! \\):\n\n\[ 6! = 720 \\]\n\nNext, we compute the numerator:\n\n\[ 14 \\times 13 \\times 12 \\times 11 \\times 10 \\times 9 = 2,162,160 \\]\n\nDividing the numerator by the denominator:\n\n\[ \\frac{2,162,160}{720} = 3,003 \\]\n\nThus, the number of ways to select the team is:\n\n\[\n\\boxed{3003}\n\\]
        ### Task
        Question: [QUESTION]
        Answer: [ANSWER]
    """ 

    # direct_cot_prefix = "Given a question, please tell me how to get this answer."
    # direct_cot_postfix = "\nOnly return a detailed step-by-step solution, containing only 'Step-by-Step Solution' and 'Final Answer'(within \\boxed{}). The detailed step-by-step solution is:"
    direct_cot_postfix = "\n only return a coherent thought process, without explicit step-by-step or point-by-point breakdown."

    vanilla_cot_prompts = []
    direct_cot_prompts = []
    
    tokenizer = llm.get_tokenizer()
    for example in test_data:
        vanilla_msg = [{"role": "user", "content": vanilla_cot_prefix + "Question: " + example["question"].strip()}]
        direct_msg = [{"role": "user", "content": direct_cot_format.replace("[QUESTION]", example["question"].strip()).replace("[ANSWER]", example['gt'])}]
        
        # direct_msg = [{"role": "user", "content": direct_cot_prefix + "Question: " + example["question"].strip() + "\nAnswer: " + example['gt'] + direct_cot_postfix}]
        
        vanilla_p = tokenizer.apply_chat_template(vanilla_msg, tokenize=False, add_generation_prompt=True)
        direct_p = tokenizer.apply_chat_template(direct_msg, tokenize=False, add_generation_prompt=True)
        
        if tokenizer.bos_token and vanilla_p.startswith(tokenizer.bos_token):
            vanilla_p = vanilla_p[len(tokenizer.bos_token):]
        if tokenizer.bos_token and direct_p.startswith(tokenizer.bos_token):
            direct_p = direct_p[len(tokenizer.bos_token):]
            
        vanilla_cot_prompts.append(vanilla_p)
        direct_cot_prompts.append(direct_p)

    print(f"Generating vanilla CoT for {len(vanilla_cot_prompts)} examples...")
    vanilla_raw_outputs = llm.generate(vanilla_cot_prompts, sampling_params)
    
    print(f"Generating direct CoT for {len(direct_cot_prompts)} examples...")
    direct_raw_outputs = llm.generate(direct_cot_prompts, sampling_params)

    vanilla_outputs = []
    direct_outputs = []

    for i, request_output in enumerate(vanilla_raw_outputs):
        for completion in request_output.outputs:
            response_text = completion.text
            thinking_content = response_text.split('</think>')[0] + '</think>'
            full_thinking_path = request_output.prompt + thinking_content            
            pred_answer = extract_answer(response_text.split('</think>')[-1], data_name="omni-math")
            gt_answer = test_data[i]['gt']
            
            vanilla_outputs.append({
                "index": i,
                "response": full_thinking_path,
                "pred": pred_answer,
                "gt": gt_answer,
                "isCorrect": verify(parse(f"${gt_answer}$"), parse(f"${pred_answer}$"))
            })

    for i, request_output in enumerate(direct_raw_outputs):
        for completion in request_output.outputs:
            response_text = completion.text
            thinking_content = response_text.split('</think>')[-1] + '</think>'
            if "<think>" not in vanilla_cot_prompts[i]:
                full_thinking_path = vanilla_cot_prompts[i] + '<think>' + thinking_content
            else:
                full_thinking_path = vanilla_cot_prompts[i] + thinking_content
            
            pred_answer = extract_answer(response_text.split('</think>')[-1], data_name="omni-math")
            gt_answer = test_data[i]['gt']

            direct_outputs.append({
                "index": i,
                "response": full_thinking_path,
                "pred": pred_answer,
                "gt": gt_answer,
                "isCorrect": verify(parse(f"${gt_answer}$"), parse(f"${pred_answer}$"))
            })

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_file_path = os.path.join(args.save_dir, "thinking_results.json")
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump({
            "vanilla": vanilla_outputs,
            "direct": direct_outputs 
        }, f, ensure_ascii=False, indent=4)

    print(f"Successfully saved results to {save_file_path}")

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
        default="results/Projection_temp"
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
        default="MATH",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_gen",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    if args.max_examples:
        args.save_dir = os.path.join(args.save_dir, f"{0}_{args.max_examples}_gen_{args.num_gen}")

    print(args.save_dir)

    main(args)