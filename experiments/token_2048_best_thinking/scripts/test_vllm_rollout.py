#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 vLLM Rollout 脚本

用少量数据快速验证脚本是否正常工作。
"""

import os
import sys
from pathlib import Path
import pandas as pd

# 添加 verl 到 Python 路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))


def create_test_data(output_path: str = "outputs/test_input.parquet"):
    """创建测试数据"""
    print("正在创建测试数据...")
    
    test_data = pd.DataFrame({
        'q_id': [0, 1, 2],
        'question': [
            "What is 2 + 2?",
            "Explain machine learning in simple terms.",
            "Write a haiku about technology."
        ],
        'prompt': [
            [{"role": "user", "content": "What is 2 + 2?"}],
            [{"role": "user", "content": "Explain machine learning in simple terms."}],
            [{"role": "user", "content": "Write a haiku about technology."}]
        ]
    })
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    test_data.to_parquet(output_path, index=False)
    
    print(f"✓ 测试数据已创建: {output_path}")
    print(f"  共 {len(test_data)} 条数据")
    return output_path


def run_test(model_path: str, fast_mode: bool = True):
    """运行测试"""
    from vllm_rollout import vllm_rollout
    
    print("\n" + "=" * 80)
    print("vLLM Rollout 快速测试")
    print("=" * 80)
    
    # 创建测试数据
    input_path = create_test_data("outputs/test_input.parquet")
    output_path = "outputs/test_output.parquet"
    
    # 测试参数（快速模式使用较小的值）
    if fast_mode:
        n_samples = 2
        max_tokens = 128
        max_num_seqs = 16
        print("\n使用快速模式（少量采样，快速验证）")
    else:
        n_samples = 8
        max_tokens = 2048
        max_num_seqs = 256
        print("\n使用完整模式（完整参数）")
    
    print(f"\n测试配置:")
    print(f"  模型: {model_path}")
    print(f"  采样数: {n_samples}")
    print(f"  最大 tokens: {max_tokens}")
    
    # 执行 rollout
    print("\n开始测试...")
    try:
        vllm_rollout(
            model_path=model_path,
            input_parquet=input_path,
            output_parquet=output_path,
            n_samples=n_samples,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=-1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            seed=42,
            enforce_eager=True,  # 快速测试使用 eager 模式
            enable_prefix_caching=True,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
        )
        
        # 验证输出
        print("\n验证输出...")
        df = pd.read_parquet(output_path)
        
        print(f"✓ 输出文件包含 {len(df)} 行")
        print(f"✓ 列: {list(df.columns)}")
        
        # 检查 responses 列
        if 'responses' in df.columns:
            first_responses = df.iloc[0]['responses']
            print(f"✓ 第一行包含 {len(first_responses)} 个响应")
            print(f"\n第一个响应示例:")
            print(f"  问题: {df.iloc[0]['question']}")
            print(f"  响应: {first_responses[0][:200]}...")
            
            # 统计
            total_responses = sum(len(row['responses']) for _, row in df.iterrows())
            print(f"\n✓ 总共生成 {total_responses} 个响应")
            print(f"✓ 预期: {len(df)} × {n_samples} = {len(df) * n_samples}")
            
            if total_responses == len(df) * n_samples:
                print("\n" + "=" * 80)
                print("✅ 测试通过！vLLM Rollout 工作正常")
                print("=" * 80)
                return True
            else:
                print("\n" + "=" * 80)
                print("❌ 测试失败：响应数量不匹配")
                print("=" * 80)
                return False
        else:
            print("\n" + "=" * 80)
            print("❌ 测试失败：输出中没有 'responses' 列")
            print("=" * 80)
            return False
            
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败：{e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="快速测试 vLLM Rollout 脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/datacenter/models/Qwen/Qwen3-4B-Instruct-2507",
        help="模型路径（默认：Qwen3-4B）"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="使用完整模式而不是快速模式"
    )
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        print("请使用 --model_path 指定正确的模型路径")
        sys.exit(1)
    
    # 运行测试
    success = run_test(args.model_path, fast_mode=not args.full)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()



