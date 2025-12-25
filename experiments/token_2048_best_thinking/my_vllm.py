#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 vllm 的生成脚本，用于高效生成数据（Rollout）
"""

import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def generate_responses(
    model_path: str,
    # input_parquet: str,
    # output_parquet: str,
    n_samples: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    gpu_memory_utilization: float = 0.95, # 新增: vllm显存占用比例
    tensor_parallel_size: int = 2,       # 新增: 多卡并行数量
    max_model_len:int = 32768,           # 新增: 模型最大长度
):
    """
    使用 vllm 高效生成响应
    
    Args:
        model_path: 模型路径
        input_parquet: 输入文件路径
        output_parquet: 输出文件路径
        n_samples: 每个问题生成多少个响应 (Best-of-N / Rollout)
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: nucleus sampling参数
        gpu_memory_utilization: KV cache 预留显存比例
        tensor_parallel_size: 使用多少张卡进行张量并行
        max_model_len: 模型最大长度
    """
    # 1.读取 formatted_prompts.json
    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/fake_formatted_prompts.json", "r") as f:
        formatted_prompts = json.load(f)
    formatted_prompts = formatted_prompts[:1600]
    print("Total number of prompts: ", len(formatted_prompts))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 初始化 vllm 引擎
    print(f"Initializing vLLM engine with model {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16", # 推荐在 Ampere+ 架构显卡上使用
        max_model_len=max_model_len
    )

    # 3. 设置采样参数
    # vllm 的 n 参数可以直接为一个 prompt 生成多个 output，比 for 循环快得多
    sampling_params = SamplingParams(
        n=n_samples,              # 关键：一次生成 n 个样本
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id] if tokenizer.eos_token_id else None,
        repetition_penalty=1.05,
        ignore_eos=False,
    )

    # 4. 执行生成
    print(f"Generating {n_samples} responses for {len(formatted_prompts)} prompts...")
    # vllm 会自动处理 batching，不需要手动分 batch
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 5. 解析结果
    all_responses = []
    
    # 6. 打开 /mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/stage2_continuation_prompts.jsonl
    # key:token_id q_id question gt_answer token_2048
    # outputs是使用上述每个条目再继续sample 100 次的结果
    # 将结果保存到 /mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/final_complete_rollouts_100.jsonl
    for output in outputs:
        # output.outputs 是一个 list，长度为 n_samples
        batch_responses = [o.text for o in output.outputs]
        all_responses.append(batch_responses)
    print("Stage3 rollouts generated")
    print("="*100)
    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/stage3_rollouts_51200.json", "w") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=4)
    print("Stage3 rollouts saved")
    print("="*100)

    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/final_complete_rollouts_51200.jsonl", "a") as f:
        with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/stage2_continuation_prompts.jsonl", "r") as f_1:
            for line,responses in zip(f_1,all_responses):
                data = json.loads(line)
                token_id = data["token_id"]
                q_id = data["q_id"]
                question = data["question"]
                gt_answer = data["gt_answer"]
                token_2048 = data["token_2048"]
                for response in responses:
                    f.write(json.dumps({
                        "token_id": token_id,
                        "q_id": q_id,
                        "question": question,
                        "gt_answer": gt_answer,
                        "token_2048": token_2048,
                        "complete_rollout": token_2048+response,
                    }) + "\n")





if __name__ == "__main__":
    print("="*100)
    print("Starting generation...")
    print("="*100)
    generate_responses(
            model_path="/datacenter/models/Qwen/Qwen3-4B-Instruct-2507",
            n_samples=32,
            max_new_tokens=16384,
            max_model_len=20480,
        )
    print("="*100)
    print("Generation completed")
    print("="*100)