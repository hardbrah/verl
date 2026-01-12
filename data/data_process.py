import pandas as pd

def add_data_source(file_path, source_name="gsm8k", output_path=None):
    df = pd.read_parquet(file_path)
    # 增加 data_source 列
    df['data_source'] = source_name
    # 覆盖保存
    df.to_parquet(output_path)
    print(f"已成功为 {output_path} 添加 data_source: {source_name}")

# 处理你的训练集和测试集
add_data_source("/data/datasets/openai/gsm8k/main/train-00000-of-00001.parquet", output_path="/data/chenhaotian/verl/data/train.parquet")
add_data_source("/data/datasets/openai/gsm8k/main/test-00000-of-00001.parquet", output_path="/data/chenhaotian/verl/data/test.parquet")