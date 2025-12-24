#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3：对每个token_2048继续生成100次

功能：
1. 读取阶段2准备的continuation prompts
2. 使用verl的main_generation继续生成100次
3. 保存完整的rollouts并计算正确性
4. 计算每个token_2048的准确率（acc = 正确次数/100）

输入：stage2_input.parquet
输出：
- stage3_output.parquet - 临时中间结果
- stage3_rollouts.{parquet,jsonl} - 800k个完整rollout

输出格式（stage3_rollouts）：
- 800k个完整rollout：{token_id, q_id, question, gt_answer, token_2048, acc}
  - token_id: 该rollout对应的token_2048的全局ID
  - q_id: 问题ID
  - acc: 该token_2048继续生成100次的准确率（正确次数/100）
  注：每个token_2048有100个rollout，它们的acc值相同
"""

import os
import sys
from pathlib import Path

# 添加verl到Python路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

import pandas as pd
import numpy as np
import json
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

# 导入verl模块
from verl.trainer.main_generation import run_generation
from verl.utils.reward_score.math_dapo import compute_score


def verify_continuation_prompts(input_path: str):
    """
    验证continuation prompts是否格式正确
    
    Args:
        input_path: 阶段2输出的路径
    """
    print(f"[验证] 检查continuation prompts格式...")
    
    df = pd.read_parquet(input_path)
    
    # 检查必需的列
    required_columns = ['question_id', 'sample_id', 'token_2048', 
                       'ground_truth', 'continuation_prompt_raw']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")
    
    print(f"✓ 数据格式正确")
    print(f"✓ 共 {len(df)} 条数据")
    
    # 检查prompt长度
    avg_len = df['continuation_prompt_raw'].str.len().mean()
    max_len = df['continuation_prompt_raw'].str.len().max()
    
    print(f"✓ Prompt平均长度: {avg_len:.0f} 字符 (~{avg_len*0.5:.0f} tokens)")
    print(f"✓ Prompt最大长度: {max_len} 字符 (~{max_len*0.5:.0f} tokens)")
    
    return True


def prepare_generation_input(
    stage2_input_path: str,
    temp_input_path: str = "outputs/stage3_temp_input.parquet"
):
    """
    将阶段2的输出转换为verl generation可以直接使用的格式
    
    由于verl的tokenizer.apply_chat_template不支持raw string，
    我们需要一个特殊处理：将continuation_prompt包装成chat格式
    
    Args:
        stage2_input_path: 阶段2的输出
        temp_input_path: 临时输入文件
    """
    print(f"[准备] 转换数据格式用于生成...")
    
    df = pd.read_parquet(stage2_input_path)
    
    # 为了让verl能够使用，我们将continuation_prompt包装成chat格式
    # 使用已有的'continuation_prompt'列（包含user和assistant消息）
    
    df_for_gen = df[['continuation_prompt']].copy()
    df_for_gen.columns = ['prompt']  # 重命名为'prompt'以匹配config中的prompt_key
    
    # 保存
    os.makedirs(os.path.dirname(temp_input_path) or '.', exist_ok=True)
    df_for_gen.to_parquet(temp_input_path, index=False)
    
    print(f"✓ 准备完成，保存到: {temp_input_path}")
    
    return temp_input_path


def run_continuation_generation(config_path: str = "configs/stage3_config.yaml"):
    """
    运行继续生成
    
    Args:
        config_path: 配置文件路径
    """
    print(f"[生成] 开始继续生成...")
    
    # 使用hydra加载配置
    with hydra.initialize(config_path="../configs", version_base=None):
        config = hydra.compose(config_name="stage3_config")
        
        # 确保路径正确
        config.data.path = "outputs/stage3_temp_input.parquet"
        config.data.output_path = "outputs/stage3_raw_output.parquet"
        
        print("\n配置信息：")
        print(f"- 模型: {config.model.path}")
        print(f"- 输入: {config.data.path}")
        print(f"- 输出: {config.data.output_path}")
        print(f"- 采样次数: {config.data.n_samples}")
        print(f"- 生成长度: {config.rollout.response_length}")
        print(f"- GPU数量: {config.trainer.n_gpus_per_node}")
        print(f"- Batch size: {config.data.batch_size}")
        
        print("\n⏳ 开始生成（这将花费较长时间，约10小时）...")
        print("提示：")
        print("- 可以使用 Ctrl+C 中断")
        print("- 生成过程中会定期输出进度")
        print("- 如果遇到OOM错误，请减小batch_size")
        print()
        
        run_generation(config)
    
    print("\n✓ 生成完成！")


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


def process_continuation_output(
    stage2_input_path: str,
    raw_output_path: str,
    final_output_path: str,
    n_samples: int = 100
):
    """
    处理继续生成的结果，并保存800k个完整rollout
    
    Args:
        stage2_input_path: 阶段2的输入
        raw_output_path: verl生成的原始输出
        final_output_path: 最终输出路径（临时文件）
        n_samples: 每个token_2048继续生成的次数
    """
    print(f"[处理] 正在处理生成结果...")
    
    # 读取数据
    stage2_df = pd.read_parquet(stage2_input_path)
    raw_output_df = pd.read_parquet(raw_output_path)
    
    print(f"✓ 读取阶段2输入: {len(stage2_df)} 条")
    print(f"✓ 读取生成输出: {len(raw_output_df)} 条")
    
    # 处理每一条数据
    result_data = []
    
    print("\n正在计算正确性...")
    for idx in tqdm(range(len(raw_output_df)), desc="处理中"):
        stage2_row = stage2_df.iloc[idx]
        output_row = raw_output_df.iloc[idx]
        
        token_id = stage2_row['token_id']  # 使用token_id而不是question_id和sample_id
        q_id = stage2_row['q_id']
        question = stage2_row['question']
        token_2048 = stage2_row['token_2048']
        ground_truth = stage2_row['gt_answer']
        
        # verl的输出：responses是一个列表，包含n_samples个生成结果
        continuations = output_row['responses']
        
        for continuation_id, continuation in enumerate(continuations):
            # 完整的rollout = token_2048 + continuation
            full_rollout = token_2048 + continuation
            
            # 使用verl的数学答案验证函数
            try:
                score_result = compute_score(
                    solution_str=full_rollout,
                    ground_truth=ground_truth,
                    strict_box_verify=False
                )
                is_correct = score_result['acc']
                pred_answer = score_result.get('pred', '[INVALID]')
            except Exception as e:
                # 如果验证失败，标记为错误
                is_correct = False
                pred_answer = f"[ERROR: {str(e)}]"
            
            result_data.append({
                'token_id': token_id,
                'q_id': q_id,
                'question': question,
                'token_2048': token_2048,
                'continuation': continuation,
                'full_rollout': full_rollout,
                'gt_answer': ground_truth,
                'pred_answer': pred_answer,
                'is_correct': is_correct,
                'continuation_id': continuation_id
            })
    
    # 保存中间结果
    result_df = pd.DataFrame(result_data)
    os.makedirs(os.path.dirname(final_output_path) or '.', exist_ok=True)
    result_df.to_parquet(final_output_path, index=False)
    
    print(f"\n✓ 处理完成！")
    print(f"✓ 生成 {len(result_df)} 条完整rollouts")
    print(f"✓ 保存中间结果到: {final_output_path}")
    
    # 计算每个token_2048的准确率（acc = 正确次数/100）
    print("\n正在计算每个token_2048的准确率...")
    token_acc_dict = {}
    grouped = result_df.groupby('token_id')
    for token_id, group in tqdm(grouped, desc="计算acc"):
        correct_count = group['is_correct'].sum()
        total_count = len(group)
        acc = correct_count / total_count
        token_acc_dict[token_id] = acc
    
    # 为每个rollout添加acc字段
    result_df['acc'] = result_df['token_id'].map(token_acc_dict)
    
    # 保存800k个完整rollout（只包含必需字段）
    print("\n保存800k个完整rollout（jsonl和parquet格式）...")
    rollouts_data = []
    for _, row in result_df.iterrows():
        rollouts_data.append({
            'token_id': row['token_id'],
            'q_id': row['q_id'],
            'question': row['question'],
            'gt_answer': row['gt_answer'],
            'token_2048': row['full_rollout'],  # 完整rollout（token_2048 + continuation）
            'acc': row['acc']
        })
    
    save_dual_format(rollouts_data, "outputs/stage3_rollouts")
    
    # 统计信息
    print("\n统计信息：")
    total_correct = result_df['is_correct'].sum()
    total_count = len(result_df)
    overall_acc = total_correct / total_count
    
    print(f"- 总体准确率: {overall_acc:.2%} ({total_correct}/{total_count})")
    print(f"- 总rollout数: {len(result_df)} (预期: 8000 × 100 = 800,000)")
    
    # 按token_id统计
    print(f"\n- token_2048平均准确率: {result_df.groupby('token_id')['is_correct'].mean().mean():.2%}")
    print(f"- 最好的token_2048准确率: {result_df.groupby('token_id')['is_correct'].mean().max():.2%}")
    print(f"- 最差的token_2048准确率: {result_df.groupby('token_id')['is_correct'].mean().min():.2%}")
    
    # 按q_id统计
    print(f"\n- 问题平均准确率: {result_df.groupby('q_id')['is_correct'].mean().mean():.2%}")
    print(f"- 准确率最高的问题: {result_df.groupby('q_id')['is_correct'].mean().max():.2%}")
    print(f"- 准确率最低的问题: {result_df.groupby('q_id')['is_correct'].mean().min():.2%}")
    
    return final_output_path


def main():
    """
    主函数：完整的阶段3流程
    """
    print("=" * 80)
    print("阶段3：对每个token_2048继续生成100次")
    print("=" * 80)
    
    # 获取实验根目录
    exp_root = Path(__file__).resolve().parent.parent
    os.chdir(exp_root)
    print(f"工作目录: {exp_root}\n")
    
    # 检查阶段2输出是否存在
    stage2_input = "outputs/stage2_input.parquet"
    if not os.path.exists(stage2_input):
        print(f"❌ 错误：找不到阶段2的输出文件: {stage2_input}")
        print("请先运行: python scripts/stage2_prepare_data.py")
        sys.exit(1)
    
    # 步骤1：验证数据
    print("步骤1/4：验证continuation prompts")
    print("-" * 80)
    verify_continuation_prompts(stage2_input)
    
    # 步骤2：准备生成输入
    print("\n步骤2/4：准备生成输入")
    print("-" * 80)
    temp_input = prepare_generation_input(
        stage2_input_path=stage2_input,
        temp_input_path="outputs/stage3_temp_input.parquet"
    )
    
    # 步骤3：运行生成
    print("\n步骤3/4：运行继续生成")
    print("-" * 80)
    run_continuation_generation()
    
    # 步骤4：处理输出
    print("\n步骤4/4：处理生成结果并计算正确性")
    print("-" * 80)
    final_output = process_continuation_output(
        stage2_input_path=stage2_input,
        raw_output_path="outputs/stage3_raw_output.parquet",
        final_output_path="outputs/stage3_output.parquet",
        n_samples=100
    )
    
    print("\n" + "=" * 80)
    print("阶段3完成！")
    print("=" * 80)
    print(f"✓ 输出文件:")
    print(f"  - {final_output} (中间结果)")
    print(f"  - outputs/stage3_rollouts.parquet (800k个完整rollout)")
    print(f"  - outputs/stage3_rollouts.jsonl (800k个完整rollout)")
    print(f"\n✓ 数据格式: {{token_id, q_id, question, gt_answer, token_2048, acc}}")
    print(f"✓ acc = 该token_2048继续生成100次的准确率（正确次数/100）")
    print(f"✓ 下一步: 运行 'python scripts/stage4_select_best.py'")
    print()


if __name__ == "__main__":
    main()

