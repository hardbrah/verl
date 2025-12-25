#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 vllm 的生成脚本，用于高效生成数据（Rollout）
"""
# 修复 CUDA 多进程问题：在导入任何 CUDA 相关模块之前设置
# 必须在初始化 CUDA 之前调用，避免 "Cannot re-initialize CUDA in forked subprocess" 错误
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

import os
# 强制使用 vLLM v0，避免 v1 的 CUDA 多进程问题
# os.environ["VLLM_USE_V1"] = "0"  # 强制覆盖，不使用 setdefault 
import sys
# import multiprocessing
from pathlib import Path
import pandas as pd
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


# 添加verl到Python路径 (保留原逻辑)
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

def generate_responses_stage3(
    model_path: str,
    input_parquet: str,
    output_parquet: str,
    n_samples: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    gpu_memory_utilization: float = 0.9, # 新增: vllm显存占用比例
    tensor_parallel_size: int = 2,       # 新增: 多卡并行数量
    max_model_len:int = 32768,           # 新增: 模型最大长度
):
    """
    使用 vllm 高效生成响应 - Stage3 续写版本
    
    Stage3 特点：
    - 输入是对话历史列表（包含 user + 未完成的 assistant 回答）
    - 需要让模型继续之前的思考过程
    
    Args:
        model_path: 模型路径
        input_parquet: 输入文件路径（包含对话历史）
        output_parquet: 输出文件路径
        n_samples: 每个问题生成多少个响应 (Best-of-N / Rollout)
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: nucleus sampling参数
        gpu_memory_utilization: KV cache 预留显存比例
        tensor_parallel_size: 使用多少张卡进行张量并行
        max_model_len: 模型最大长度
    """
    
    print(f"Loading data from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    
    if 'question' not in df.columns:
        raise ValueError("Input data must have a 'question' column")

    # 1. 预处理 Prompts：使用 Tokenizer 应用 Chat Template
    # Stage3 特殊处理：question 列包含完整的对话历史（列表格式）
    print(f"Processing continuation prompts using tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    formatted_prompts = []
    for conversation in tqdm(df['question'], desc="Applying Chat Template for Continuation"):
        # Stage3: conversation 是一个列表，包含:
        # [
        #   {"role": "user", "content": "问题"},
        #   {"role": "assistant", "content": "未完成的回答"}
        # ]
        # 直接使用完整的对话历史，add_generation_prompt=True 会添加继续生成的提示
        text = tokenizer.apply_chat_template(
            conversation,  # 直接传入对话列表
            tokenize=False,
            add_generation_prompt=True  # 关键：让模型知道要继续生成
        )
        formatted_prompts.append(text)

    with open("outputs/formatted_prompts.json", "w") as f:
        json.dump(formatted_prompts, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Formatted {len(formatted_prompts)} continuation prompts")

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
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id] if tokenizer.eos_token_id else None
    )

    # 4. 执行生成
    print(f"Generating {n_samples} responses for {len(formatted_prompts)} prompts...")
    # vllm 会自动处理 batching，不需要手动分 batch
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 5. 解析结果
    all_responses = []
    
    # 保持与原输出格式一致：每行是一个 prompt，对应一个 response list
    # outputs 的顺序与 formatted_prompts 输入顺序一致
    for output in outputs:
        # output.outputs 是一个 list，长度为 n_samples
        batch_responses = [o.text for o in output.outputs]
        all_responses.append(batch_responses)

    # 6. 保存结果
    df['responses'] = all_responses
    
    os.makedirs(os.path.dirname(output_parquet) or '.', exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Results saved to {output_parquet}")
    print(f"Generated {len(df)} prompts × {n_samples} responses = {len(df) * n_samples} total responses")

def generate_responses(
    model_path: str,
    input_parquet: str,
    output_parquet: str,
    n_samples: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    gpu_memory_utilization: float = 0.9, # 新增: vllm显存占用比例
    tensor_parallel_size: int = 1,       # 新增: 多卡并行数量
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
    
    print(f"Loading data from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    
    if 'question' not in df.columns:
        raise ValueError("Input data must have a 'question' column")

    # 1. 预处理 Prompts：使用 Tokenizer 应用 Chat Template
    # 虽然 vllm 也能处理 chat template，但显式处理可以确保完全控制格式
    print(f"Processing prompts using tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    formatted_prompts = []
    for question in tqdm(df['question'], desc="Applying Chat Template"):
        # 纯字符串格式
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(text)

    with open("outputs/formatted_prompts.json", "w") as f:
        json.dump(formatted_prompts, f, ensure_ascii=False, indent=4)

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
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id] if tokenizer.eos_token_id else None
    )

    # 4. 执行生成
    print(f"Generating {n_samples} responses for {len(formatted_prompts)} prompts...")
    # vllm 会自动处理 batching，不需要手动分 batch
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 5. 解析结果
    all_responses = []
    
    # 保持与原输出格式一致：每行是一个 prompt，对应一个 response list
    # outputs 的顺序与 formatted_prompts 输入顺序一致
    for output in outputs:
        # output.outputs 是一个 list，长度为 n_samples
        batch_responses = [o.text for o in output.outputs]
        all_responses.append(batch_responses)

    # 6. 保存结果
    df['responses'] = all_responses
    
    os.makedirs(os.path.dirname(output_parquet) or '.', exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Results saved to {output_parquet}")
    print(f"Generated {len(df)} prompts × {n_samples} responses = {len(df) * n_samples} total responses")




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-performance generation script using vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="GPU memory utilization (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    generate_responses(
        model_path=args.model_path,
        input_parquet=args.input,
        output_parquet=args.output,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util
    )