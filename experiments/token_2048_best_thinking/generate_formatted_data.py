#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json


def generate_data():
    input_jsonl_path = "/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/stage1_output.jsonl"
    model_path = "/datacenter/models/Qwen/Qwen3-4B-Instruct-2507"
    fake_template = "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n{truncated_response}"
    
    print(f"Loading data from {input_jsonl_path}")
    with open(input_jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    formatted_prompts = []
    for item in data:
        query = item["question"]
        truncated_response = item["token_2048"]
        formatted_prompt = fake_template.format(query=query, truncated_response=truncated_response)
        formatted_prompts.append(formatted_prompt)

    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/fake_formatted_prompts.json", "w") as f:
        json.dump(formatted_prompts, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Formatted {len(formatted_prompts)} continuation prompts")


if __name__ == "__main__":
    generate_data()