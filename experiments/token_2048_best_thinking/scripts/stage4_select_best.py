#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段4：计算token_2048_acc并选择最佳token_2048

功能：
1. 读取阶段3的输出（包含所有完整rollouts及其正确性）
2. 对每个token_2048，计算其token_2048_acc（100次继续生成的准确率）
3. 对每个问题，选择token_2048_acc最高的token_2048作为最佳思考路径
4. 保存结果并生成分析报告

输入：stage3_output.parquet
输出：
- final_best.{parquet,jsonl}: 1k个最佳token_2048
- token_2048_accuracies.parquet: 8k个token_2048的准确率（中间结果）
- analysis_report.txt: 详细分析报告
- accuracy_distribution.png: 准确率分布图（可选）

输出格式（final_best）：
- 1k个最佳token_2048：{token_id, q_id, question, gt_answer, token_2048, acc}
  - token_id: 该最佳token_2048的全局ID
  - q_id: 问题ID
  - acc: 该token_2048继续生成100次的准确率（正确次数/100）
"""

import os
import sys
from pathlib import Path

# 添加verl到Python路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os


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


def compute_token_2048_acc(stage3_output_path: str):
    """
    计算每个token_2048的准确率
    
    Args:
        stage3_output_path: 阶段3的输出路径
        
    Returns:
        DataFrame with columns: token_id, q_id, token_2048_acc, token_2048, question, gt_answer
    """
    print(f"[计算] 正在计算token_2048_acc...")
    
    # 读取数据
    df = pd.read_parquet(stage3_output_path)
    print(f"✓ 读取 {len(df)} 条rollout数据")
    
    # 验证数据完整性
    n_tokens = df['token_id'].nunique()
    n_questions = df['q_id'].nunique()
    n_continuations = df.groupby('token_id').size().iloc[0] if len(df) > 0 else 0
    
    print(f"✓ 数据结构: {n_tokens} 个token_2048 × {n_continuations} 次继续生成")
    print(f"✓ 预期总数: {n_tokens * n_continuations}")
    print(f"✓ 问题数: {n_questions}")
    
    # 按token_id分组，计算每个token_2048的准确率
    print("\n正在计算每个token_2048的准确率...")
    
    acc_data = []
    grouped = df.groupby('token_id')
    
    for token_id, group in tqdm(grouped, desc="计算中"):
        # 计算这个token_2048的准确率
        correct_count = group['is_correct'].sum()
        total_count = len(group)
        token_2048_acc = correct_count / total_count
        
        # 获取相关信息（所有行相同，取第一行）
        q_id = group.iloc[0]['q_id']
        token_2048 = group.iloc[0]['token_2048']
        question = group.iloc[0]['question']
        gt_answer = group.iloc[0]['gt_answer']
        
        acc_data.append({
            'token_id': token_id,
            'q_id': q_id,
            'token_2048': token_2048,
            'question': question,
            'gt_answer': gt_answer,
            'token_2048_acc': token_2048_acc,
            'correct_count': correct_count,
            'total_count': total_count
        })
    
    acc_df = pd.DataFrame(acc_data)
    
    print(f"\n✓ 计算完成，共 {len(acc_df)} 个token_2048")
    print(f"✓ 平均准确率: {acc_df['token_2048_acc'].mean():.2%}")
    print(f"✓ 最高准确率: {acc_df['token_2048_acc'].max():.2%}")
    print(f"✓ 最低准确率: {acc_df['token_2048_acc'].min():.2%}")
    
    return acc_df


def select_best_token_2048(acc_df: pd.DataFrame):
    """
    为每个问题选择最佳的token_2048
    
    Args:
        acc_df: token_2048准确率数据（包含token_id, q_id, question, gt_answer等）
        
    Returns:
        DataFrame with best token_2048 for each question
    """
    print(f"\n[选择] 正在为每个问题选择最佳token_2048...")
    
    best_data = []
    analysis_data = []  # 用于生成分析报告的详细数据
    
    for q_id in tqdm(acc_df['q_id'].unique(), desc="选择中"):
        # 获取这个问题的所有token_2048
        question_accs = acc_df[acc_df['q_id'] == q_id]
        
        # 找到最佳的token_2048（准确率最高）
        best_idx = question_accs['token_2048_acc'].idxmax()
        best_row = question_accs.loc[best_idx]
        
        # 收集所有准确率
        all_accs = question_accs['token_2048_acc'].tolist()
        all_token_ids = question_accs['token_id'].tolist()
        
        # 计算统计信息
        acc_std = np.std(all_accs)
        acc_range = max(all_accs) - min(all_accs)
        acc_mean = np.mean(all_accs)
        
        # 保存最佳token_2048（按要求的格式）
        best_data.append({
            'token_id': best_row['token_id'],
            'q_id': q_id,
            'question': best_row['question'],
            'gt_answer': best_row['gt_answer'],
            'token_2048': best_row['token_2048'],
            'acc': best_row['token_2048_acc']
        })
        
        # 保存详细分析数据
        analysis_data.append({
            'q_id': q_id,
            'question': best_row['question'],
            'gt_answer': best_row['gt_answer'],
            'best_token_id': best_row['token_id'],
            'best_token_2048': best_row['token_2048'],
            'best_acc': best_row['token_2048_acc'],
            'all_token_ids': all_token_ids,
            'all_accs': all_accs,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'acc_range': acc_range,
            'improvement_over_mean': best_row['token_2048_acc'] - acc_mean
        })
    
    best_df = pd.DataFrame(best_data)
    analysis_df = pd.DataFrame(analysis_data)
    
    print(f"\n✓ 选择完成，共 {len(best_df)} 个问题")
    print(f"✓ 最佳token_2048平均准确率: {best_df['acc'].mean():.2%}")
    print(f"✓ 相比平均提升: {analysis_df['improvement_over_mean'].mean():.2%}")
    
    return best_df, analysis_df


def generate_analysis_report(analysis_df: pd.DataFrame, acc_df: pd.DataFrame, output_path: str):
    """
    生成详细的分析报告
    
    Args:
        analysis_df: 分析数据（包含详细的统计信息）
        acc_df: 所有token_2048准确率数据
        output_path: 报告输出路径
    """
    print(f"\n[分析] 正在生成分析报告...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Token 2048 最佳思考实验 - 分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. 基本统计
    report_lines.append("## 1. 基本统计")
    report_lines.append("-" * 80)
    report_lines.append(f"总问题数: {len(analysis_df)}")
    report_lines.append(f"每个问题的token_2048数量: {len(acc_df) // len(analysis_df)}")
    report_lines.append(f"每个token_2048的继续生成次数: 100")
    report_lines.append(f"总rollout数量: {len(acc_df) * 100}")
    report_lines.append("")
    
    # 2. 最佳token_2048准确率分析
    report_lines.append("## 2. 最佳token_2048准确率分析")
    report_lines.append("-" * 80)
    report_lines.append(f"平均准确率: {analysis_df['best_acc'].mean():.4f} ({analysis_df['best_acc'].mean()*100:.2f}%)")
    report_lines.append(f"中位数准确率: {analysis_df['best_acc'].median():.4f} ({analysis_df['best_acc'].median()*100:.2f}%)")
    report_lines.append(f"最高准确率: {analysis_df['best_acc'].max():.4f} ({analysis_df['best_acc'].max()*100:.2f}%)")
    report_lines.append(f"最低准确率: {analysis_df['best_acc'].min():.4f} ({analysis_df['best_acc'].min()*100:.2f}%)")
    report_lines.append(f"标准差: {analysis_df['best_acc'].std():.4f}")
    report_lines.append("")
    
    # 准确率分布
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(analysis_df['best_acc'], bins=bins)
    report_lines.append("准确率分布:")
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = hist[i]
        pct = count / len(analysis_df) * 100
        report_lines.append(f"  [{low:.1f}-{high:.1f}): {count:4d} ({pct:5.1f}%)")
    report_lines.append("")
    
    # 3. 所有token_2048的平均表现
    report_lines.append("## 3. 所有token_2048的平均表现")
    report_lines.append("-" * 80)
    report_lines.append(f"所有token_2048平均准确率: {acc_df['token_2048_acc'].mean():.4f} ({acc_df['token_2048_acc'].mean()*100:.2f}%)")
    report_lines.append(f"最佳选择 vs 平均: +{analysis_df['best_acc'].mean() - acc_df['token_2048_acc'].mean():.4f} "
                       f"(+{(analysis_df['best_acc'].mean() - acc_df['token_2048_acc'].mean())*100:.2f}%)")
    report_lines.append("")
    
    # 4. 选择的价值分析
    report_lines.append("## 4. 选择的价值分析")
    report_lines.append("-" * 80)
    report_lines.append(f"平均每个问题的准确率提升: {analysis_df['improvement_over_mean'].mean():.4f} "
                       f"({analysis_df['improvement_over_mean'].mean()*100:.2f}%)")
    report_lines.append(f"准确率提升的标准差: {analysis_df['improvement_over_mean'].std():.4f}")
    report_lines.append(f"最大提升: {analysis_df['improvement_over_mean'].max():.4f} ({analysis_df['improvement_over_mean'].max()*100:.2f}%)")
    report_lines.append(f"最小提升: {analysis_df['improvement_over_mean'].min():.4f} ({analysis_df['improvement_over_mean'].min()*100:.2f}%)")
    report_lines.append("")
    
    # 5. 多样性分析
    report_lines.append("## 5. 多样性分析（不同token_2048的差异）")
    report_lines.append("-" * 80)
    report_lines.append(f"平均准确率标准差: {analysis_df['acc_std'].mean():.4f}")
    report_lines.append(f"平均准确率范围: {analysis_df['acc_range'].mean():.4f}")
    report_lines.append("")
    report_lines.append("说明：")
    report_lines.append("- 标准差越大，说明不同思考路径的效果差异越大")
    report_lines.append("- 范围越大，说明选择好的起始路径更重要")
    report_lines.append("")
    
    # 6. Top 10 最佳问题
    report_lines.append("## 6. Top 10 最佳问题（准确率最高）")
    report_lines.append("-" * 80)
    top10 = analysis_df.nlargest(10, 'best_acc')
    for idx, row in top10.iterrows():
        report_lines.append(f"\n问题 {row['q_id']}:")
        report_lines.append(f"  准确率: {row['best_acc']:.2%}")
        report_lines.append(f"  问题: {row['question'][:100]}...")
        report_lines.append(f"  答案: {row['gt_answer']}")
    report_lines.append("")
    
    # 7. Bottom 10 最差问题
    report_lines.append("## 7. Bottom 10 最差问题（准确率最低）")
    report_lines.append("-" * 80)
    bottom10 = analysis_df.nsmallest(10, 'best_acc')
    for idx, row in bottom10.iterrows():
        report_lines.append(f"\n问题 {row['q_id']}:")
        report_lines.append(f"  准确率: {row['best_acc']:.2%}")
        report_lines.append(f"  问题: {row['question'][:100]}...")
        report_lines.append(f"  答案: {row['gt_answer']}")
    report_lines.append("")
    
    # 8. 按token_id分析（哪些token_2048被选为最佳）
    report_lines.append("## 8. 按token_id分析")
    report_lines.append("-" * 80)
    report_lines.append(f"被选为最佳的token_2048数量: {analysis_df['best_token_id'].nunique()}")
    report_lines.append(f"总共的token_2048数量: {acc_df['token_id'].nunique()}")
    report_lines.append("")
    
    # 9. 实验结论
    report_lines.append("## 9. 实验结论")
    report_lines.append("-" * 80)
    avg_improvement = analysis_df['improvement_over_mean'].mean() * 100
    report_lines.append(f"1. 通过选择最佳的token_2048，平均可以提升 {avg_improvement:.2f}% 的准确率")
    report_lines.append(f"2. 最佳token_2048的平均准确率为 {analysis_df['best_acc'].mean()*100:.2f}%")
    report_lines.append(f"3. 不同思考路径的准确率标准差为 {analysis_df['acc_std'].mean():.4f}，说明起始路径的选择很重要")
    
    if analysis_df['acc_range'].mean() > 0.3:
        report_lines.append(f"4. 准确率范围达到 {analysis_df['acc_range'].mean():.2f}，说明有些问题对起始路径非常敏感")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 分析报告已保存到: {output_path}")
    
    # 也打印到控制台
    print("\n" + report_text)
    
    return report_text


def plot_accuracy_distribution(analysis_df: pd.DataFrame, output_path: str):
    """
    绘制准确率分布图（可选，需要matplotlib）
    
    Args:
        analysis_df: 分析数据
        output_path: 图片输出路径
    """
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n[可视化] 正在生成准确率分布图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 最佳准确率分布
        axes[0, 0].hist(analysis_df['best_acc'], bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Best Accuracy')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Best Token_2048 Accuracy')
        axes[0, 0].axvline(analysis_df['best_acc'].mean(), color='red', 
                          linestyle='--', label=f"Mean: {analysis_df['best_acc'].mean():.2%}")
        axes[0, 0].legend()
        
        # 2. 准确率提升分布
        axes[0, 1].hist(analysis_df['improvement_over_mean'], bins=20, edgecolor='black')
        axes[0, 1].set_xlabel('Improvement over Mean')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Improvement of Best vs Mean')
        axes[0, 1].axvline(0, color='gray', linestyle='-', alpha=0.5)
        
        # 3. 准确率范围分布
        axes[1, 0].hist(analysis_df['acc_range'], bins=20, edgecolor='black')
        axes[1, 0].set_xlabel('Accuracy Range')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Range of Accuracies across Token_2048s')
        
        # 4. 散点图：平均准确率 vs 最佳准确率
        axes[1, 1].scatter(analysis_df['acc_mean'], analysis_df['best_acc'], alpha=0.5)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='y=x')
        axes[1, 1].set_xlabel('Mean Accuracy')
        axes[1, 1].set_ylabel('Best Accuracy')
        axes[1, 1].set_title('Mean vs Best Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化图表已保存到: {output_path}")
        
    except ImportError:
        print("⚠️  matplotlib未安装，跳过可视化")
        print("   如需可视化，请运行: pip install matplotlib")


def main():
    """
    主函数：完整的阶段4流程
    """
    print("=" * 80)
    print("阶段4：计算准确率并选择最佳token_2048")
    print("=" * 80)
    
    # 获取实验根目录
    exp_root = Path(__file__).resolve().parent.parent
    os.chdir(exp_root)
    print(f"工作目录: {exp_root}\n")
    
    # 检查输入文件
    stage3_output = "outputs/stage3_output.parquet"
    
    if not os.path.exists(stage3_output):
        print(f"❌ 错误：找不到阶段3的输出文件: {stage3_output}")
        print("请先运行: python scripts/stage3_continuation.py")
        sys.exit(1)
    
    # 步骤1：计算token_2048_acc
    print("步骤1/4：计算每个token_2048的准确率")
    print("-" * 80)
    acc_df = compute_token_2048_acc(stage3_output)
    
    # 保存中间结果
    acc_df.to_parquet("outputs/token_2048_accuracies.parquet", index=False)
    print(f"✓ token_2048准确率已保存到: outputs/token_2048_accuracies.parquet")
    
    # 步骤2：选择最佳token_2048
    print("\n步骤2/4：为每个问题选择最佳token_2048")
    print("-" * 80)
    best_df, analysis_df = select_best_token_2048(acc_df)
    
    # 保存最终结果（jsonl和parquet格式）
    print("\n保存1k个最佳token_2048（jsonl和parquet格式）...")
    save_dual_format(best_df, "outputs/final_best")
    
    # 同时保存分析数据用于生成报告
    analysis_df.to_parquet("outputs/final_best_analysis.parquet", index=False)
    print(f"✓ 分析数据已保存到: outputs/final_best_analysis.parquet")
    
    # 步骤3：生成分析报告
    print("\n步骤3/4：生成详细分析报告")
    print("-" * 80)
    report_path = "outputs/analysis_report.txt"
    generate_analysis_report(analysis_df, acc_df, report_path)
    
    # 步骤4：可视化（可选）
    print("\n步骤4/4：生成可视化图表")
    print("-" * 80)
    plot_path = "outputs/accuracy_distribution.png"
    plot_accuracy_distribution(analysis_df, plot_path)
    
    print("\n" + "=" * 80)
    print("阶段4完成！实验全部完成！")
    print("=" * 80)
    print("\n输出文件：")
    print(f"  ✓ outputs/final_best.parquet - 1k个最佳token_2048数据")
    print(f"  ✓ outputs/final_best.jsonl - 1k个最佳token_2048数据")
    print(f"  ✓ {report_path} - 详细分析报告")
    print(f"  ✓ outputs/token_2048_accuracies.parquet - 8k个token_2048的准确率")
    print(f"  ✓ outputs/final_best_analysis.parquet - 分析数据")
    if os.path.exists(plot_path):
        print(f"  ✓ {plot_path} - 准确率分布图")
    
    print(f"\n✓ 数据格式（final_best）: {{token_id, q_id, question, gt_answer, token_2048, acc}}")
    print(f"✓ acc = 该token_2048继续生成100次的准确率（正确次数/100）")
    
    print("\n快速查看结果：")
    print(f"  python -c \"import pandas as pd; df = pd.read_parquet('outputs/final_best.parquet'); print(df.head())\"")
    print()


if __name__ == "__main__":
    main()

