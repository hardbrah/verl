#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1：从DAPO-MATH-17k中随机采样1k个问题，并生成8个token_2048

功能：
1. 加载DAPO-MATH-17k数据集
2. 使用固定种子随机采样1000个问题（确保在不同平台和服务器上可复现）
3. 调用verl的main_generation生成responses
4. 保存结果（同时保存jsonl和parquet格式）

输出格式（jsonl和parquet）：
- 8k个token_2048：{token_id, q_id, question, gt_answer, token_2048}
  - token_id: 全局唯一ID (0-7999)
  - q_id: 问题ID (0-999)
  - question: 原始问题
  - gt_answer: ground truth答案
  - token_2048: 生成的2048个token
"""

import os
import sys
import json
from pathlib import Path

# 添加verl到Python路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

import pandas as pd
import numpy as np

# 固定随机种子，确保采样可复现
RANDOM_SEED = 42


def prepare_dapo_math_sample(
    dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k",
    sample_size: int = 1000,
    output_path: str = "outputs/stage1_sampled_questions.parquet",
    seed: int = RANDOM_SEED
):
    """
    从DAPO-MATH数据集中采样问题并准备为verl格式
    
    使用固定种子确保在不同平台和服务器上采样结果一致
    
    Args:
        dataset_name: HuggingFace数据集名称
        sample_size: 采样数量
        output_path: 输出路径
        seed: 随机种子（默认42，确保在不同平台和服务器上可复现）
    """
    print(f"[阶段1] 正在加载数据集: {dataset_name}")
    print(f"随机种子: {seed} (确保在不同平台和服务器上可复现)")
    
    # 加载数据集
    # try:
    #     dataset = load_dataset(dataset_name, "default", split="train")
    #     print(f"数据集加载成功，总共 {len(dataset)} 个问题")
    # except Exception as e:
    #     print(f"从HuggingFace加载失败: {e}")
    #     print("尝试从本地加载...")
    # 如果HuggingFace加载失败，尝试从本地文件加载
    local_path = os.path.expanduser("/datacenter/datasets/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet")
    if os.path.exists(local_path):
        dataset = pd.read_parquet(local_path)
        dataset = dataset.to_dict('records')
        print(f"从本地加载成功，总共 {len(dataset)} 个问题")
    else:
        raise FileNotFoundError(
            f"无法加载数据集。请确保：\n"
            f"1. 网络可以访问HuggingFace，或\n"
            f"2. 本地存在文件: {local_path}\n"
            f"运行 'bash recipe/dapo/prepare_dapo_data.sh' 下载数据"
        )
    
    # 使用固定种子随机采样
    np.random.seed(seed)
    total_size = len(dataset)
    
    if sample_size > total_size:
        print(f"警告：请求的采样数量 ({sample_size}) 大于数据集大小 ({total_size})")
        print(f"将使用全部 {total_size} 个问题")
        sample_size = total_size
        indices = list(range(total_size))
    else:
        # 使用固定种子采样，确保可复现
        indices = np.random.choice(total_size, size=sample_size, replace=False).tolist()
        # 保存采样索引，便于复现
        indices_save_path = output_path.replace('.parquet', '_sampling_indices.json')
        with open(indices_save_path, 'w') as f:
            json.dump({
                'seed': seed,
                'sample_size': sample_size,
                'total_size': total_size,
                'indices': indices
            }, f, indent=2)
        print(f"采样索引已保存到: {indices_save_path}")
    
    print(f"随机采样 {sample_size} 个问题")
    
    # 准备数据
    sampled_data = []
    for q_id, idx in enumerate(indices):
        if isinstance(dataset, list):
            item = dataset[idx]
        else:
            item = dataset[int(idx)]
        
        # BytedTsinghua-SIA/DAPO-Math-17k数据集格式：
        # - prompt: list (已经是chat格式，如 [{"role": "user", "content": "..."}])
        # - reward_model: dict (包含 ground_truth)
        # - data_source: str
        
        # 获取prompt（已经是chat格式的list）
        prompt = item.get('prompt', [])
        
        # 从prompt中提取问题文本（用于记录）
        question_text = ""
        if isinstance(prompt, list) and len(prompt) > 0:
            # 找到role为user的消息
            for msg in prompt:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    question_text = msg.get('content', '')
                    break
        
        # 获取ground_truth（在reward_model字典中）
        reward_model = item.get('reward_model', {})
        gt_answer = reward_model.get('ground_truth', '')
        
        # 确保gt_answer是字符串
        if isinstance(gt_answer, np.ndarray):
            gt_answer = gt_answer.item() if gt_answer.size == 1 else str(gt_answer)
        elif not isinstance(gt_answer, str):
            gt_answer = str(gt_answer)
        
        sampled_data.append({
            'q_id': q_id,
            'question': question_text,
            'gt_answer': gt_answer,
            'prompt': prompt  # 仅用于verl生成
        })
    
    # 保存为parquet文件
    df = pd.DataFrame(sampled_data)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"采样完成，已保存到: {output_path}")
    print(f"数据格式预览：")
    print(df[['q_id', 'question', 'gt_answer']].head(2))
    
    return output_path


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


def process_generation_output(
    generation_output_path: str,
    sampled_questions_path: str,
    final_output_base: str,
    n_samples: int = 8
):
    """
    处理生成结果，将其组织为实验需要的格式
    
    为每个token_2048分配全局唯一的token_id (0-7999)
    
    Args:
        generation_output_path: verl生成的原始输出路径
        sampled_questions_path: 采样的问题路径
        final_output_base: 最终输出路径（不含扩展名）
        n_samples: 每个问题的采样数
    """
    print(f"[阶段1] 正在处理生成结果...")
    
    # 读取生成结果
    gen_df = pd.read_parquet(generation_output_path)
    questions_df = pd.read_parquet(sampled_questions_path)
    
    # verl的输出格式：每一行包含一个问题的所有n_samples个responses
    # gen_df['responses'] 是一个列表，包含n个response
    
    result_data = []
    token_id = 0  # 全局token_id计数器
    
    for idx, row in gen_df.iterrows():
        q_id = idx
        question_data = questions_df.iloc[q_id]
        
        responses = row['responses']  # 这是一个包含n个response的列表
        
        for sample_idx, response in enumerate(responses):
            result_data.append({
                'token_id': token_id,  # 全局唯一ID (0-7999)
                'q_id': q_id,          # 问题ID (0-999)
                'question': question_data['question'],
                'gt_answer': question_data['gt_answer'],
                'token_2048': response
            })
            token_id += 1
    
    print(f"处理完成，生成 {len(result_data)} 条数据")
    print(f"预期：{len(questions_df)} 个问题 × {n_samples} 个采样 = {len(questions_df) * n_samples} 条")
    print(f"token_id 范围: 0 - {token_id - 1}")
    
    # 同时保存jsonl和parquet格式
    print("\n保存数据（jsonl和parquet格式）...")
    parquet_path, jsonl_path = save_dual_format(result_data, final_output_base)
    
    # 统计信息
    result_df = pd.DataFrame(result_data)
    print("\n统计信息：")
    print(f"- token_id范围: {result_df['token_id'].min()} - {result_df['token_id'].max()}")
    print(f"- q_id范围: {result_df['q_id'].min()} - {result_df['q_id'].max()}")
    print(f"- 平均token_2048长度: {result_df['token_2048'].str.len().mean():.0f} 字符")
    print(f"- 最短token_2048: {result_df['token_2048'].str.len().min()} 字符")
    print(f"- 最长token_2048: {result_df['token_2048'].str.len().max()} 字符")
    
    # 显示示例数据
    print("\n数据示例（前2条）：")
    for i in range(min(2, len(result_data))):
        item = result_data[i]
        print(f"\n[{i+1}] token_id={item['token_id']}, q_id={item['q_id']}")
        print(f"    question: {item['question'][:100]}...")
        print(f"    gt_answer: {item['gt_answer']}")
        print(f"    token_2048: {item['token_2048'][:100]}...")
    
    return parquet_path


def main():
    """
    主函数：完整的阶段1流程
    """
    print("=" * 80)
    print("阶段1：采样1k问题并生成8个token_2048")
    print("=" * 80)
    print(f"随机种子: {RANDOM_SEED} (确保在不同平台可复现)")
    
    # 获取实验根目录
    exp_root = Path(__file__).resolve().parent.parent
    os.chdir(exp_root)
    print(f"工作目录: {exp_root}\n")
    
    # 步骤1：采样问题
    print("步骤1/3：从DAPO-MATH-17k采样1000个问题")
    print("-" * 80)
    sampled_questions_path = prepare_dapo_math_sample(
        dataset_name="BytedTsinghua-SIA/DAPO-Math-17k",
        sample_size=20,
        output_path="outputs/stage1_sampled_questions.parquet",
        seed=RANDOM_SEED
    )
    
    # 步骤2：调用简单生成脚本
    print("\n步骤2/3：调用简单生成脚本生成8个token_2048")
    print("-" * 80)
    print("正在使用 transformers 直接生成...")
    
    # 导入简单生成函数
    from simple_generate import generate_responses
    
    model_path = "/datacenter/models/Qwen/Qwen3-4B-Instruct-2507"
    input_path = sampled_questions_path
    output_path = "outputs/stage1_raw_output.parquet"
    
    print("\n生成配置：")
    print(f"- 模型: {model_path}")
    print(f"- 输入: {input_path}")
    print(f"- 输出: {output_path}")
    print(f"- 采样次数: 8")
    print(f"- 生成长度: 2048 tokens")
    print(f"- 温度: 1.0")
    print(f"- Top-p: 0.95")
    print(f"- 随机种子: {RANDOM_SEED}")
    
    print("\n开始生成（这可能需要一段时间）...")
    
    # 设置随机种子
    import torch
    import numpy as np
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    generate_responses(
        model_path=model_path,
        input_parquet=input_path,
        output_parquet=output_path,
        n_samples=8,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        batch_size=1  # 使用小的batch size以避免OOM
    )
    
    print("\n生成完成！")
    
    # 步骤3：处理输出
    print("\n步骤3/3：处理生成结果并保存为jsonl+parquet格式")
    print("-" * 80)
    final_output_path = process_generation_output(
        generation_output_path="outputs/stage1_raw_output.parquet",
        sampled_questions_path=sampled_questions_path,
        final_output_base="outputs/stage1_output",  # 不含扩展名
        n_samples=8
    )
    
    print("\n" + "=" * 80)
    print("阶段1完成！")
    print("=" * 80)
    print(f"✓ 输出文件:")
    print(f"  - outputs/stage1_output.parquet (8000条)")
    print(f"  - outputs/stage1_output.jsonl (8000条)")
    print(f"  - outputs/stage1_sampled_questions_sampling_indices.json (采样索引，可复现)")
    print(f"\n✓ 数据格式: {{token_id, q_id, question, gt_answer, token_2048}}")
    print(f"✓ token_id范围: 0-7999 (全局唯一)")
    print(f"✓ q_id范围: 0-999 (问题ID)")
    print(f"\n✓ 下一步: 运行 'python scripts/stage2_prepare_data.py'")
    print()


if __name__ == "__main__":
    main()
