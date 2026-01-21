#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 rollout jsonl 文件生成伪造的格式化提示词
适配字段: question, response (而非 token_2048)
"""
import json
import os

def generate_data():
    # 输入文件路径
    input_jsonl_path = "/data/chenhaotian/verl/experiments/data_generation/outputs/qwen3_4b_1000query_16sample/rollout_20260107_134649.jsonl"
    
    # 输出文件路径
    output_dir = "/data/chenhaotian/verl/experiments/token_2048_best_thinking/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, "fake_formatted_prompts_from_rollout.json")
    
    # 伪造模板 - 使用 Qwen3 的 chat 格式
    fake_template = "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n{truncated_response}"
    
    print(f"Loading data from {input_jsonl_path}")
    with open(input_jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} records")
    
    formatted_prompts = []
    for item in data:
        query = item["question"]
        # 使用 response 字段代替 token_2048
        truncated_response = item["response"]
        formatted_prompt = fake_template.format(query=query, truncated_response=truncated_response)
        formatted_prompts.append(formatted_prompt)

    with open(output_json_path, "w") as f:
        json.dump(formatted_prompts, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Formatted {len(formatted_prompts)} continuation prompts")
    print(f"✓ Output saved to {output_json_path}")


if __name__ == "__main__":
    generate_data()
