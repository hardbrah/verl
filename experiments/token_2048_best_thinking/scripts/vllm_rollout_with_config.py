#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用配置文件的 vLLM Rollout 脚本

这个脚本从 YAML 配置文件读取参数，使用起来更方便。
"""

import os
import sys
from pathlib import Path
import argparse
import yaml

# 添加 verl 到 Python 路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

from vllm_rollout import vllm_rollout


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="使用配置文件的 vLLM Rollout 脚本"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径（YAML 格式）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入文件路径（覆盖配置文件中的值）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（覆盖配置文件中的值）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"正在加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 提取参数
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    sampling_config = config.get('sampling', {})
    engine_config = config.get('engine', {})
    
    # 命令行参数覆盖配置文件
    input_path = args.input or data_config.get('input')
    output_path = args.output or data_config.get('output')
    
    if not input_path:
        raise ValueError("必须指定输入文件路径（通过 --input 或配置文件）")
    if not output_path:
        raise ValueError("必须指定输出文件路径（通过 --output 或配置文件）")
    
    # 执行 rollout
    vllm_rollout(
        model_path=model_config.get('path'),
        input_parquet=input_path,
        output_parquet=output_path,
        n_samples=sampling_config.get('n_samples', 8),
        max_tokens=sampling_config.get('max_tokens', 2048),
        temperature=sampling_config.get('temperature', 1.0),
        top_p=sampling_config.get('top_p', 0.95),
        top_k=sampling_config.get('top_k', -1),
        repetition_penalty=sampling_config.get('repetition_penalty', 1.0),
        prompt_key=data_config.get('prompt_key', 'prompt'),
        tensor_parallel_size=engine_config.get('tensor_parallel_size', 1),
        gpu_memory_utilization=engine_config.get('gpu_memory_utilization', 0.9),
        max_model_len=engine_config.get('max_model_len'),
        dtype=engine_config.get('dtype', 'bfloat16'),
        seed=sampling_config.get('seed', 42),
        enforce_eager=engine_config.get('enforce_eager', False),
        enable_prefix_caching=engine_config.get('enable_prefix_caching', True),
        max_num_seqs=engine_config.get('max_num_seqs', 256),
        trust_remote_code=model_config.get('trust_remote_code', True),
    )


if __name__ == "__main__":
    main()










