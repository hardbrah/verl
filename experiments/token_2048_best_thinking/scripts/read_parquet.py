#!/usr/bin/env python3
"""
读取parquet文件并打印第一条数据
"""
import pandas as pd
import json

# 读取parquet文件
parquet_path = "/datacenter/datasets/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet"
print(f"正在读取文件: {parquet_path}")

# 读取数据
df = pd.read_parquet(parquet_path)

print(f"\n数据集总行数: {len(df)}")
print(f"数据集列名: {df.columns.tolist()}")

# 获取第一条数据
first_row = df.iloc[0]

print("\n" + "="*80)
print("第一条数据:")
print("="*80)

# 以更美观的格式打印第一条数据
for column in df.columns:
    value = first_row[column]
    print(f"\n【{column}】:")
    if isinstance(value, str) and len(value) > 200:
        print(value[:200] + "...")
    else:
        print(value)

print("\n" + "="*80)

# 也可以以字典格式打印（处理非JSON可序列化类型）
print("\n以字典格式查看第一条数据:")
first_row_dict = {}
for key, value in first_row.to_dict().items():
    # 处理可能的numpy类型
    if hasattr(value, 'tolist'):
        first_row_dict[key] = value.tolist()
    else:
        first_row_dict[key] = value

try:
    print(json.dumps(first_row_dict, ensure_ascii=False, indent=2))
except (TypeError, ValueError) as e:
    print(f"无法以JSON格式打印: {e}")
    print(first_row_dict)

