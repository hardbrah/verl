#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化的 vLLM Rollout 脚本

使用 vLLM 引擎进行高效的批量推理，适用于大规模采样场景。
相比于 transformers，vLLM 提供了更高的吞吐量和更好的 GPU 利用率。

功能：
1. 使用 vLLM 引擎进行批量推理
2. 支持多样本采样（每个 prompt 生成多个 response）
3. 支持自定义采样参数（temperature, top_p, top_k 等）
4. 自动处理 chat template
5. 保存为 parquet 格式

输入格式：
- parquet 文件，包含 'prompt' 列（字符串或 chat 格式的列表）

输出格式：
- parquet 文件，包含原始列 + 'responses' 列（包含 n_samples 个响应的列表）
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from tqdm import tqdm
import argparse

# 添加 verl 到 Python 路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))


def setup_vllm_engine(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = True,
    max_num_seqs: int = 256,
    seed: int = 42,
):
    """
    初始化 vLLM 引擎
    
    Args:
        model_path: 模型路径
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU 显存利用率（0-1）
        max_model_len: 模型最大长度（None 表示使用模型配置）
        dtype: 数据类型（auto, bfloat16, float16）
        trust_remote_code: 是否信任远程代码
        enforce_eager: 是否强制使用 eager 模式（禁用 CUDA graphs）
        enable_prefix_caching: 是否启用前缀缓存
        max_num_seqs: 最大并发序列数
        seed: 随机种子
    
    Returns:
        LLM 引擎实例
    """
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vLLM is not installed. Please install it with:\n"
            "pip install vllm"
        )
    
    print(f"正在初始化 vLLM 引擎...")
    print(f"  模型路径: {model_path}")
    print(f"  张量并行: {tensor_parallel_size}")
    print(f"  显存利用率: {gpu_memory_utilization}")
    print(f"  数据类型: {dtype}")
    print(f"  最大序列数: {max_num_seqs}")
    print(f"  前缀缓存: {enable_prefix_caching}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=max_num_seqs,
        seed=seed,
    )
    
    print("✓ vLLM 引擎初始化完成\n")
    return llm


def prepare_prompts(
    df: pd.DataFrame,
    tokenizer,
    prompt_key: str = "prompt",
) -> List[str]:
    """
    准备 prompts，应用 chat template
    
    Args:
        df: 输入 DataFrame
        tokenizer: tokenizer 实例
        prompt_key: prompt 列名
    
    Returns:
        处理后的 prompt 列表
    """
    prompts = []
    
    for idx, row in df.iterrows():
        prompt = row[prompt_key]
        
        # 应用 chat template
        if isinstance(prompt, list):
            # 如果 prompt 已经是 chat 格式的 list
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        elif isinstance(prompt, str):
            # 如果是字符串，转换为 chat 格式
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        prompts.append(text)
    
    return prompts


def vllm_rollout(
    model_path: str,
    input_parquet: str,
    output_parquet: str,
    n_samples: int = 8,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    prompt_key: str = "prompt",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: str = "bfloat16",
    seed: int = 42,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = True,
    max_num_seqs: int = 256,
    trust_remote_code: bool = True,
):
    """
    使用 vLLM 引擎进行 rollout
    
    Args:
        model_path: 模型路径
        input_parquet: 输入 parquet 文件路径
        output_parquet: 输出 parquet 文件路径
        n_samples: 每个 prompt 生成多少个响应
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: nucleus sampling 参数
        top_k: top-k sampling 参数（-1 表示不使用）
        repetition_penalty: 重复惩罚
        prompt_key: prompt 列名
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU 显存利用率
        max_model_len: 模型最大长度
        dtype: 数据类型
        seed: 随机种子
        enforce_eager: 是否强制 eager 模式
        enable_prefix_caching: 是否启用前缀缓存
        max_num_seqs: 最大并发序列数
        trust_remote_code: 是否信任远程代码
    """
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    
    print("=" * 80)
    print("vLLM Rollout 脚本")
    print("=" * 80)
    
    # 1. 加载 tokenizer
    print(f"\n正在加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    print("✓ Tokenizer 加载完成")
    
    # 2. 初始化 vLLM 引擎
    llm = setup_vllm_engine(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=max_num_seqs,
        seed=seed,
    )
    
    # 3. 加载输入数据
    print(f"正在加载数据: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    
    if prompt_key not in df.columns:
        raise ValueError(f"Input data must have a '{prompt_key}' column")
    
    print(f"✓ 数据加载完成，共 {len(df)} 条")
    
    # 4. 准备 prompts
    print(f"\n正在准备 prompts...")
    prompts = prepare_prompts(df, tokenizer, prompt_key=prompt_key)
    print(f"✓ Prompts 准备完成，共 {len(prompts)} 条")
    
    # 5. 设置采样参数
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    
    print(f"\n采样参数：")
    print(f"  n_samples: {n_samples}")
    print(f"  max_tokens: {max_tokens}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  repetition_penalty: {repetition_penalty}")
    print(f"  seed: {seed}")
    
    # 6. 执行批量生成
    print(f"\n开始生成（这可能需要一段时间）...")
    print(f"预计生成: {len(prompts)} prompts × {n_samples} samples = {len(prompts) * n_samples} 总响应数")
    
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"✓ 生成完成")
    
    # 7. 处理输出
    print(f"\n正在处理输出...")
    all_responses = []
    
    for output in outputs:
        # 每个 output 包含 n_samples 个响应
        responses = [out.text for out in output.outputs]
        all_responses.append(responses)
    
    # 8. 将响应添加到 DataFrame
    df['responses'] = all_responses
    
    # 9. 保存结果
    os.makedirs(os.path.dirname(output_parquet) or '.', exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    
    print(f"✓ 结果已保存到: {output_parquet}")
    
    # 10. 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"输入 prompts: {len(df)}")
    print(f"每个 prompt 的采样数: {n_samples}")
    print(f"总响应数: {len(df) * n_samples}")
    
    # 计算响应长度统计
    all_lengths = []
    for responses in all_responses:
        for response in responses:
            all_lengths.append(len(response))
    
    if all_lengths:
        print(f"\n响应长度统计（字符数）：")
        print(f"  平均: {sum(all_lengths) / len(all_lengths):.0f}")
        print(f"  最小: {min(all_lengths)}")
        print(f"  最大: {max(all_lengths)}")
    
    # 显示示例
    print(f"\n示例输出（第 1 条）：")
    if len(df) > 0:
        print(f"Prompt: {str(df.iloc[0][prompt_key])[:200]}...")
        print(f"响应数: {len(all_responses[0])}")
        print(f"第 1 个响应: {all_responses[0][0][:200]}...")
    
    print("\n" + "=" * 80)
    print("Rollout 完成！")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="轻量化的 vLLM Rollout 脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

1. 基本用法：
   python vllm_rollout.py \\
       --model_path /path/to/model \\
       --input data.parquet \\
       --output output.parquet

2. 自定义采样参数：
   python vllm_rollout.py \\
       --model_path /path/to/model \\
       --input data.parquet \\
       --output output.parquet \\
       --n_samples 16 \\
       --max_tokens 4096 \\
       --temperature 0.8 \\
       --top_p 0.95

3. 多 GPU 并行：
   python vllm_rollout.py \\
       --model_path /path/to/model \\
       --input data.parquet \\
       --output output.parquet \\
       --tensor_parallel_size 2 \\
       --gpu_memory_utilization 0.95
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 parquet 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 parquet 文件路径"
    )
    
    # 采样参数
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="每个 prompt 生成多少个响应（默认：8）"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="最大生成 token 数（默认：2048）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度（默认：1.0）"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling 参数（默认：0.95）"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling 参数，-1 表示不使用（默认：-1）"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="重复惩罚（默认：1.0，无惩罚）"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="输入数据中的 prompt 列名（默认：prompt）"
    )
    
    # vLLM 引擎参数
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行大小（默认：1）"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU 显存利用率，0-1 之间（默认：0.9）"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="模型最大长度，None 表示使用模型配置（默认：None）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="数据类型（默认：bfloat16）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）"
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="强制使用 eager 模式，禁用 CUDA graphs（调试时有用）"
    )
    parser.add_argument(
        "--disable_prefix_caching",
        action="store_true",
        help="禁用前缀缓存"
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
        help="最大并发序列数（默认：256）"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="是否信任远程代码（默认：True）"
    )
    
    args = parser.parse_args()
    
    # 执行 rollout
    vllm_rollout(
        model_path=args.model_path,
        input_parquet=args.input,
        output_parquet=args.output,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        prompt_key=args.prompt_key,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.disable_prefix_caching,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()


