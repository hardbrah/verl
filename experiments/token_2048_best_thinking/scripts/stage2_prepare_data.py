#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段2：准备继续生成的数据

功能：
将阶段1的输出（原始问题 + token_2048）拼接成新的prompt，
使得模型认为它已经生成了token_2048，需要继续生成。

保持token_id和q_id不变，用于后续跟踪。

输入：stage1_output.{parquet,jsonl}
输出：
- stage2_input.{parquet,jsonl} - 用于verl生成
- stage2_continuation_prompts.{parquet,jsonl} - 8k个continuation_prompt记录

输出格式（stage2_continuation_prompts）：
- 8k个continuation_prompt：{token_id, q_id, question, gt_answer, token_2048, continuation_prompt}
  - continuation_prompt: 用于继续生成的完整prompt字符串
"""

import os
import sys
import json
from pathlib import Path

# 添加verl到Python路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

import pandas as pd
from tqdm import tqdm


def build_continuation_prompt(question: str, token_2048: str) -> str:
    """
    构建继续生成的prompt（原始字符串格式，不是chat格式）
    
    关键技术点：
    1. 使用Qwen的ChatML格式
    2. assistant部分不加 <|im_end|>，让模型认为还在生成中
    
    Args:
        question: 原始问题（带指令）
        token_2048: 已生成的2048个token
    
    Returns:
        拼接后的prompt字符串
    """
    # Qwen的ChatML格式
    prompt = (
        f"<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{token_2048}"
    )
    # 注意：这里故意不加 <|im_end|>
    
    return prompt


def save_dual_format(data, base_path):
    """
    同时保存jsonl和parquet两种格式
    
    Args:
        data: DataFrame或字典列表
        base_path: 基础路径（不含扩展名）
    """
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # 保存parquet格式
    parquet_path = f"{base_path}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Parquet格式已保存: {parquet_path}")
    
    # 保存jsonl格式
    jsonl_path = f"{base_path}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 将每行转为字典并保存为一行JSON
            json.dump(row.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    print(f"✓ JSONL格式已保存: {jsonl_path}")
    
    return parquet_path, jsonl_path


def prepare_continuation_data(
    stage1_output_path: str = "outputs/stage1_output.parquet",
    output_base: str = "outputs/stage2_input"
):
    """
    准备阶段2的输入数据
    
    保持token_id和q_id不变
    
    Args:
        stage1_output_path: 阶段1的输出路径
        output_base: 阶段2输入数据的保存路径（不含扩展名）
    """
    print(f"[阶段2] 正在准备继续生成的数据...")
    
    # 读取阶段1的输出
    print(f"读取阶段1输出: {stage1_output_path}")
    df = pd.read_parquet(stage1_output_path)
    print(f"读取完成，共 {len(df)} 条数据")
    
    # 验证必需字段
    required_fields = ['token_id', 'q_id', 'question', 'gt_answer', 'token_2048']
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        raise ValueError(f"缺少必需字段: {missing_fields}")
    
    # 统计信息
    n_questions = df['q_id'].nunique()
    n_tokens = len(df)
    print(f"- {n_questions} 个问题")
    print(f"- {n_tokens} 个token_2048")
    print(f"- token_id范围: {df['token_id'].min()} - {df['token_id'].max()}")
    print(f"- q_id范围: {df['q_id'].min()} - {df['q_id'].max()}")
    
    # 构建continuation prompt
    print("\n正在构建continuation prompts...")
    continuation_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
        token_id = row['token_id']
        q_id = row['q_id']
        question = row['question']
        gt_answer = row['gt_answer']
        token_2048 = row['token_2048']
        
        # 添加指令（如果还没有）
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        if instruction not in question:
            full_question = question.strip() + "\n\n" + instruction
        else:
            full_question = question
        
        # 构建continuation prompt (原始字符串格式)
        continuation_prompt_str = build_continuation_prompt(full_question, token_2048)
        
        # 为了与verl的main_generation兼容，我们需要把它包装成chat格式
        continuation_data.append({
            'token_id': token_id,  # 保持不变
            'q_id': q_id,          # 保持不变
            'question': question,
            'gt_answer': gt_answer,
            'token_2048': token_2048,
            'continuation_prompt_raw': continuation_prompt_str,  # 原始字符串
            # 同时也保存chat格式，供verl使用
            'continuation_prompt': [
                {"role": "user", "content": full_question},
                {"role": "assistant", "content": token_2048}
            ]
        })
    
    # 同时保存jsonl和parquet格式
    print(f"\n保存数据（jsonl和parquet格式）...")
    parquet_path, jsonl_path = save_dual_format(continuation_data, output_base)
    
    result_df = pd.DataFrame(continuation_data)
    
    # 额外保存一份只包含必需字段的continuation_prompts记录
    print(f"\n保存8k个continuation_prompt记录...")
    continuation_prompts_data = []
    for _, row in result_df.iterrows():
        continuation_prompts_data.append({
            'token_id': row['token_id'],
            'q_id': row['q_id'],
            'question': row['question'],
            'gt_answer': row['gt_answer'],
            'token_2048': row['token_2048'],
            'continuation_prompt': row['continuation_prompt_raw']  # 使用原始字符串格式
        })
    
    save_dual_format(continuation_prompts_data, "outputs/stage2_continuation_prompts")
    
    print(f"\n处理完成！")
    print(f"- 生成 {len(result_df)} 条continuation prompts")
    
    # 显示示例
    print("\n" + "=" * 80)
    print("示例数据预览（第1条）：")
    print("=" * 80)
    example = result_df.iloc[0]
    print(f"token_id: {example['token_id']}")
    print(f"q_id: {example['q_id']}")
    print(f"\n原始问题（前200字符）:")
    print(example['question'][:200])
    print(f"\ngt_answer: {example['gt_answer']}")
    print(f"\ntoken_2048（前200字符）:")
    print(example['token_2048'][:200])
    print(f"\ncontinuation_prompt_raw（前500字符）:")
    print(example['continuation_prompt_raw'][:500])
    print("..." if len(example['continuation_prompt_raw']) > 500 else "")
    print("=" * 80)
    
    # 统计信息
    print("\n统计信息：")
    avg_prompt_len = result_df['continuation_prompt_raw'].str.len().mean()
    max_prompt_len = result_df['continuation_prompt_raw'].str.len().max()
    min_prompt_len = result_df['continuation_prompt_raw'].str.len().min()
    
    print(f"- Continuation prompt 平均长度: {avg_prompt_len:.0f} 字符")
    print(f"- 最长: {max_prompt_len} 字符")
    print(f"- 最短: {min_prompt_len} 字符")
    
    # 估算token数（粗略估计：中英文混合约1字符=0.5 token）
    avg_tokens = avg_prompt_len * 0.5
    print(f"- 估计平均token数: ~{avg_tokens:.0f} tokens")
    
    if avg_tokens > 2048:
        print("\n⚠️  警告：平均prompt长度超过2048 tokens")
        print("   请确保stage3_config.yaml中的prompt_length足够大")
    
    return parquet_path


def main():
    """
    主函数
    """
    print("=" * 80)
    print("阶段2：准备继续生成的数据")
    print("=" * 80)
    
    # 获取实验根目录
    exp_root = Path(__file__).resolve().parent.parent
    os.chdir(exp_root)
    print(f"工作目录: {exp_root}\n")
    
    # 检查阶段1输出是否存在
    stage1_output = "outputs/stage1_output.parquet"
    if not os.path.exists(stage1_output):
        print(f"❌ 错误：找不到阶段1的输出文件: {stage1_output}")
        print("请先运行: python scripts/stage1_sample.py")
        sys.exit(1)
    
    # 准备数据
    prepare_continuation_data(
        stage1_output_path=stage1_output,
        output_base="outputs/stage2_input"
    )
    
    print("\n" + "=" * 80)
    print("阶段2完成！")
    print("=" * 80)
    print(f"✓ 输出文件:")
    print(f"  - outputs/stage2_input.parquet (8000条，用于verl生成)")
    print(f"  - outputs/stage2_input.jsonl (8000条)")
    print(f"  - outputs/stage2_continuation_prompts.parquet (8000条，continuation_prompt记录)")
    print(f"  - outputs/stage2_continuation_prompts.jsonl (8000条)")
    print(f"\n✓ 数据格式: 保持token_id和q_id不变，添加continuation_prompt")
    print(f"✓ 8k个continuation_prompt已保存（包含token_id、q_id、question、gt_answer、token_2048、continuation_prompt）")
    print(f"✓ 下一步: 运行 'python scripts/stage3_continuation.py'")
    print()
    
    # 重要提醒
    print("⚠️  重要提醒：")
    print("在运行阶段3之前，请检查 configs/stage3_config.yaml 中的配置：")
    print("- model.path: 确保模型路径正确")
    print("- rollout.prompt_length: 应 >= 平均prompt长度")
    print("- rollout.response_length: 建议设为4096或更大，确保生成完整")
    print("- data.batch_size: 如果显存不足，可以调小")
    print()


if __name__ == "__main__":
    main()
