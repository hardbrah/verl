# vLLM Rollout 快速开始

> 5 分钟快速上手 vLLM Rollout 脚本

## 前置条件

```bash
# 1. 确保已安装 vLLM
pip install vllm

# 2. 验证安装
python -c "import vllm; print('vLLM version:', vllm.__version__)"

# 3. 检查 GPU
nvidia-smi
```

## 方式一：一键测试（推荐新手）

```bash
cd /mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking

# 快速测试（2-3 分钟）
bash scripts/vllm_shortcuts.sh test

# 如果测试通过，可以运行演示
bash scripts/vllm_shortcuts.sh rollout-demo
```

## 方式二：使用命令行参数

```bash
# 基本用法
python scripts/vllm_rollout.py \
    --model_path /datacenter/models/Qwen/Qwen3-4B-Instruct-2507 \
    --input outputs/stage1_sampled_questions.parquet \
    --output outputs/vllm_rollout_output.parquet

# 自定义参数
python scripts/vllm_rollout.py \
    --model_path /datacenter/models/Qwen/Qwen3-4B-Instruct-2507 \
    --input your_input.parquet \
    --output your_output.parquet \
    --n_samples 16 \
    --max_tokens 2048 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9
```

## 方式三：使用配置文件

```bash
# 1. 编辑配置文件（可选）
vim configs/vllm_rollout_config.yaml

# 2. 运行
python scripts/vllm_rollout_with_config.py \
    --config configs/vllm_rollout_config.yaml \
    --input your_input.parquet \
    --output your_output.parquet
```

## 方式四：使用 Bash 脚本

```bash
# 使用默认参数
bash scripts/run_vllm_rollout.sh

# 使用自定义参数（通过环境变量）
MODEL_PATH=/path/to/model \
INPUT_PATH=input.parquet \
OUTPUT_PATH=output.parquet \
N_SAMPLES=16 \
bash scripts/run_vllm_rollout.sh
```

## 准备输入数据

输入数据必须是 parquet 格式，包含 `prompt` 列：

### 示例 1：从 CSV 转换

```python
import pandas as pd

# 读取 CSV
df = pd.read_csv('questions.csv')

# 转换为 chat 格式
df['prompt'] = df['question'].apply(
    lambda q: [{"role": "user", "content": q}]
)

# 保存为 parquet
df.to_parquet('input.parquet', index=False)
```

### 示例 2：手动创建

```python
import pandas as pd

data = pd.DataFrame({
    'q_id': [0, 1, 2],
    'question': [
        "What is 2+2?",
        "Explain AI.",
        "Write a haiku."
    ],
    'prompt': [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Explain AI."}],
        [{"role": "user", "content": "Write a haiku."}]
    ]
})

data.to_parquet('input.parquet', index=False)
```

## 查看结果

```python
import pandas as pd

# 读取结果
df = pd.read_parquet('output.parquet')

print(f"共 {len(df)} 个 prompts")
print(f"每个 prompt 有 {len(df.iloc[0]['responses'])} 个响应")

# 查看第一个 prompt 的所有响应
for i, response in enumerate(df.iloc[0]['responses']):
    print(f"\n响应 {i+1}:")
    print(response[:200] + "...")
```

## 后处理结果

### 1. 统计分析

```bash
python scripts/process_rollout_results.py analyze \
    --input output.parquet
```

### 2. 展平响应（每个响应一行）

```bash
python scripts/process_rollout_results.py flatten \
    --input output.parquet \
    --output flattened_output.parquet
```

### 3. 导出为 JSONL

```bash
python scripts/process_rollout_results.py export \
    --input output.parquet \
    --output output.jsonl
```

### 4. 过滤长度

```bash
python scripts/process_rollout_results.py filter \
    --input flattened_output.parquet \
    --output filtered_output.parquet \
    --min_length 100 \
    --max_length 5000
```

## 常见问题

### Q1: 显存不足怎么办？

```bash
# 降低显存使用
python scripts/vllm_rollout.py \
    --gpu_memory_utilization 0.7 \
    --max_num_seqs 64 \
    ...
```

### Q2: 如何使用多个 GPU？

```bash
# 设置张量并行
python scripts/vllm_rollout.py \
    --tensor_parallel_size 2 \  # 使用 2 个 GPU
    ...
```

### Q3: 如何加快生成速度？

```bash
# 增加并发和缓存
python scripts/vllm_rollout.py \
    --max_num_seqs 512 \
    --enable_prefix_caching \
    --gpu_memory_utilization 0.95 \
    ...
```

### Q4: 结果不可复现？

```bash
# 设置固定种子
python scripts/vllm_rollout.py \
    --seed 42 \
    ...
```

## 完整工作流示例

```bash
#!/bin/bash
# 完整的数据处理流程

cd /mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking

# 1. 准备数据
python -c "
import pandas as pd
df = pd.DataFrame({
    'q_id': list(range(10)),
    'question': [f'Question {i}' for i in range(10)],
    'prompt': [[{'role': 'user', 'content': f'Question {i}'}] for i in range(10)]
})
df.to_parquet('outputs/my_questions.parquet', index=False)
print('✓ 数据准备完成')
"

# 2. 执行 rollout
python scripts/vllm_rollout.py \
    --model_path /datacenter/models/Qwen/Qwen3-4B-Instruct-2507 \
    --input outputs/my_questions.parquet \
    --output outputs/my_responses.parquet \
    --n_samples 8 \
    --max_tokens 2048

# 3. 分析结果
python scripts/process_rollout_results.py analyze \
    --input outputs/my_responses.parquet

# 4. 展平结果
python scripts/process_rollout_results.py flatten \
    --input outputs/my_responses.parquet \
    --output outputs/my_responses_flat.parquet

# 5. 导出 JSONL
python scripts/process_rollout_results.py export \
    --input outputs/my_responses.parquet \
    --output outputs/my_responses.jsonl

echo "✓ 完成！"
```

## 性能参考

### 单 GPU (A100 80GB)
- **配置**: `tensor_parallel_size=1, gpu_memory_utilization=0.95`
- **性能**: ~2000-3000 tokens/s
- **适用**: 中小规模任务（< 10K prompts）

### 双 GPU (2x A100)
- **配置**: `tensor_parallel_size=2, gpu_memory_utilization=0.95`
- **性能**: ~4000-6000 tokens/s
- **适用**: 大规模任务（> 10K prompts）

### 预估时间

假设平均每个响应 1000 tokens：

| Prompts | Samples | 总 Tokens | 单 GPU 时间 | 双 GPU 时间 |
|---------|---------|-----------|-------------|-------------|
| 100     | 8       | 800K      | ~5 分钟     | ~2 分钟     |
| 1,000   | 8       | 8M        | ~45 分钟    | ~20 分钟    |
| 10,000  | 8       | 80M       | ~7 小时     | ~3.5 小时   |

## 下一步

- 📚 阅读完整文档: [docs/VLLM_ROLLOUT_GUIDE.md](VLLM_ROLLOUT_GUIDE.md)
- 🔧 查看高级配置: [configs/vllm_rollout_config.yaml](../configs/vllm_rollout_config.yaml)
- 💡 查看更多示例: [scripts/](../scripts/)

## 获取帮助

```bash
# 查看命令行帮助
python scripts/vllm_rollout.py --help

# 查看快捷命令帮助
bash scripts/vllm_shortcuts.sh help

# 查看处理工具帮助
python scripts/process_rollout_results.py --help
```

祝使用愉快！ 🚀



