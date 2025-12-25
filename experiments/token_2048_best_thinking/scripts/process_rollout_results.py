#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM Rollout 结果处理工具

提供常用的结果处理和分析功能：
1. 展平响应（将多个响应展开为多行）
2. 统计分析（长度分布、生成质量等）
3. 导出为 JSONL 格式
4. 过滤和筛选
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import argparse
from typing import Optional

# 添加 verl 到 Python 路径
verl_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(verl_root))


def flatten_responses(input_path: str, output_path: str, add_sample_id: bool = True):
    """
    展平响应：将每个 prompt 的多个响应展开为多行
    
    输入格式:
      q_id | question | responses (list)
    
    输出格式:
      q_id | question | response | sample_id
    
    Args:
        input_path: 输入 parquet 文件
        output_path: 输出 parquet 文件
        add_sample_id: 是否添加 sample_id 列
    """
    print(f"正在展平响应: {input_path}")
    df = pd.read_parquet(input_path)
    
    if 'responses' not in df.columns:
        raise ValueError("输入文件必须包含 'responses' 列")
    
    flattened_data = []
    
    for idx, row in df.iterrows():
        responses = row['responses']
        base_data = {k: v for k, v in row.items() if k != 'responses'}
        
        for sample_idx, response in enumerate(responses):
            record = base_data.copy()
            record['response'] = response
            if add_sample_id:
                record['sample_id'] = sample_idx
            flattened_data.append(record)
    
    result_df = pd.DataFrame(flattened_data)
    result_df.to_parquet(output_path, index=False)
    
    print(f"✓ 展平完成: {len(df)} 行 → {len(result_df)} 行")
    print(f"✓ 输出文件: {output_path}")
    return result_df


def analyze_responses(input_path: str, response_col: str = 'responses'):
    """
    分析响应的统计信息
    
    Args:
        input_path: 输入 parquet 文件
        response_col: 响应列名（'responses' 或 'response'）
    """
    print(f"正在分析响应: {input_path}")
    df = pd.read_parquet(input_path)
    
    if response_col not in df.columns:
        raise ValueError(f"输入文件必须包含 '{response_col}' 列")
    
    # 收集所有响应
    all_responses = []
    if response_col == 'responses':
        # 列表格式
        for _, row in df.iterrows():
            all_responses.extend(row[response_col])
    else:
        # 单个响应格式
        all_responses = df[response_col].tolist()
    
    # 计算统计信息
    lengths = [len(r) for r in all_responses]
    word_counts = [len(r.split()) for r in all_responses]
    
    print("\n" + "=" * 80)
    print("响应统计分析")
    print("=" * 80)
    print(f"总响应数: {len(all_responses)}")
    
    print(f"\n字符数统计:")
    print(f"  平均: {sum(lengths) / len(lengths):.0f}")
    print(f"  最小: {min(lengths)}")
    print(f"  最大: {max(lengths)}")
    print(f"  中位数: {sorted(lengths)[len(lengths)//2]}")
    
    print(f"\n词数统计:")
    print(f"  平均: {sum(word_counts) / len(word_counts):.1f}")
    print(f"  最小: {min(word_counts)}")
    print(f"  最大: {max(word_counts)}")
    print(f"  中位数: {sorted(word_counts)[len(word_counts)//2]}")
    
    # 长度分布
    print(f"\n长度分布（字符数）:")
    bins = [0, 100, 500, 1000, 2000, 5000, float('inf')]
    bin_labels = ['0-100', '100-500', '500-1K', '1K-2K', '2K-5K', '5K+']
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = sum(1 for l in lengths if low <= l < high)
        pct = count / len(lengths) * 100
        print(f"  {bin_labels[i]:>10}: {count:>5} ({pct:>5.1f}%)")
    
    # 检查空响应
    empty_count = sum(1 for r in all_responses if not r.strip())
    if empty_count > 0:
        print(f"\n⚠️  发现 {empty_count} 个空响应 ({empty_count/len(all_responses)*100:.1f}%)")
    
    print("=" * 80)
    
    return {
        'total_responses': len(all_responses),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_words': sum(word_counts) / len(word_counts),
        'empty_count': empty_count,
    }


def export_to_jsonl(input_path: str, output_path: str, flatten: bool = True):
    """
    导出为 JSONL 格式
    
    Args:
        input_path: 输入 parquet 文件
        output_path: 输出 jsonl 文件
        flatten: 是否展平响应
    """
    print(f"正在导出为 JSONL: {input_path}")
    df = pd.read_parquet(input_path)
    
    if flatten and 'responses' in df.columns:
        # 展平
        records = []
        for idx, row in df.iterrows():
            responses = row['responses']
            base_data = {k: v for k, v in row.items() if k != 'responses'}
            
            for sample_idx, response in enumerate(responses):
                record = base_data.copy()
                record['response'] = response
                record['sample_id'] = sample_idx
                records.append(record)
    else:
        # 不展平
        records = df.to_dict('records')
    
    # 写入 JSONL
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✓ 导出完成: {len(records)} 条记录")
    print(f"✓ 输出文件: {output_path}")


def filter_by_length(
    input_path: str,
    output_path: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    response_col: str = 'response'
):
    """
    按长度过滤响应
    
    Args:
        input_path: 输入 parquet 文件
        output_path: 输出 parquet 文件
        min_length: 最小长度（字符数）
        max_length: 最大长度（字符数）
        response_col: 响应列名
    """
    print(f"正在按长度过滤: {input_path}")
    df = pd.read_parquet(input_path)
    
    if response_col not in df.columns:
        raise ValueError(f"输入文件必须包含 '{response_col}' 列")
    
    original_count = len(df)
    
    # 应用过滤
    if min_length is not None:
        df = df[df[response_col].str.len() >= min_length]
        print(f"  过滤最小长度 {min_length}: {original_count} → {len(df)}")
    
    if max_length is not None:
        df = df[df[response_col].str.len() <= max_length]
        print(f"  过滤最大长度 {max_length}: {original_count} → {len(df)}")
    
    df.to_parquet(output_path, index=False)
    
    print(f"✓ 过滤完成: 保留 {len(df)}/{original_count} 条记录 ({len(df)/original_count*100:.1f}%)")
    print(f"✓ 输出文件: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Rollout 结果处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 展平响应（将多个响应展开为多行）:
   python process_rollout_results.py flatten \\
       --input outputs/rollout_output.parquet \\
       --output outputs/flattened_output.parquet

2. 统计分析:
   python process_rollout_results.py analyze \\
       --input outputs/rollout_output.parquet

3. 导出为 JSONL:
   python process_rollout_results.py export \\
       --input outputs/rollout_output.parquet \\
       --output outputs/output.jsonl

4. 按长度过滤:
   python process_rollout_results.py filter \\
       --input outputs/flattened_output.parquet \\
       --output outputs/filtered_output.parquet \\
       --min_length 100 \\
       --max_length 5000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # flatten 命令
    flatten_parser = subparsers.add_parser('flatten', help='展平响应')
    flatten_parser.add_argument('--input', type=str, required=True, help='输入 parquet 文件')
    flatten_parser.add_argument('--output', type=str, required=True, help='输出 parquet 文件')
    flatten_parser.add_argument('--no_sample_id', action='store_true', help='不添加 sample_id 列')
    
    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='分析响应')
    analyze_parser.add_argument('--input', type=str, required=True, help='输入 parquet 文件')
    analyze_parser.add_argument('--response_col', type=str, default='responses', help='响应列名')
    
    # export 命令
    export_parser = subparsers.add_parser('export', help='导出为 JSONL')
    export_parser.add_argument('--input', type=str, required=True, help='输入 parquet 文件')
    export_parser.add_argument('--output', type=str, required=True, help='输出 jsonl 文件')
    export_parser.add_argument('--no_flatten', action='store_true', help='不展平响应')
    
    # filter 命令
    filter_parser = subparsers.add_parser('filter', help='按长度过滤')
    filter_parser.add_argument('--input', type=str, required=True, help='输入 parquet 文件')
    filter_parser.add_argument('--output', type=str, required=True, help='输出 parquet 文件')
    filter_parser.add_argument('--min_length', type=int, help='最小长度（字符数）')
    filter_parser.add_argument('--max_length', type=int, help='最大长度（字符数）')
    filter_parser.add_argument('--response_col', type=str, default='response', help='响应列名')
    
    args = parser.parse_args()
    
    if args.command == 'flatten':
        flatten_responses(args.input, args.output, add_sample_id=not args.no_sample_id)
    elif args.command == 'analyze':
        analyze_responses(args.input, response_col=args.response_col)
    elif args.command == 'export':
        export_to_jsonl(args.input, args.output, flatten=not args.no_flatten)
    elif args.command == 'filter':
        filter_by_length(
            args.input,
            args.output,
            min_length=args.min_length,
            max_length=args.max_length,
            response_col=args.response_col
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()



