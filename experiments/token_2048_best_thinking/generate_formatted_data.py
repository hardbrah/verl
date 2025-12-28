#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from configs.config import PathConfig

def generate_data():
    input_jsonl_path = PathConfig.STAGE1_OUTPUT_JSONL
    model_path = PathConfig.MODEL_PATH
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

    with open(PathConfig.FAKE_FORMATTED_PROMPTS_JSON, "w") as f:
        json.dump(formatted_prompts, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Formatted {len(formatted_prompts)} continuation prompts")


if __name__ == "__main__":
    generate_data()