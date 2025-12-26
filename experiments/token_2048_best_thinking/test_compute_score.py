#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from verl.utils.reward_score.math_dapo import compute_score
import json
from transformers import AutoTokenizer
from collections import Counter

if __name__ == "__main__":
    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/final_complete_rollouts_51200.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    results = []
    model_path = "/datacenter/models/Qwen/Qwen3-4B-Instruct-2507" # 你的模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    num_tokens_list = []
    for item in data:
        solution_str = item["complete_rollout"]
        ground_truth = item["gt_answer"]
        result = compute_score(solution_str, ground_truth)
        results.append(result)
        # 计算 solution_str 的平均token数
        token_ids = tokenizer.encode(solution_str, add_special_tokens=False)
        num_tokens = len(token_ids)
        num_tokens_list.append(num_tokens)

    total = 0
    correct = 0
    for result in results:
        total += 1
        if result["score"] == 1:
            correct += 1
    print(f"正确率: {correct / total}")
    print(f"平均token数: {sum(num_tokens_list) / len(num_tokens_list)}")
    print(f"最大token数: {max(num_tokens_list)}")
    print(f"最小token数: {min(num_tokens_list)}")
    print(f"token数分布: {Counter(num_tokens_list)}")
    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/results_analysis_dapo_200query_8token2048_32sample.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.write(f"\n平均token数: {sum(num_tokens_list) / len(num_tokens_list)}\n")
        f.write(f"最大token数: {max(num_tokens_list)}\n")
        f.write(f"最小token数: {min(num_tokens_list)}\n")
        f.write(f"token数分布: {Counter(num_tokens_list)}\n")
        f.write(f"正确率: {correct / total}\n")