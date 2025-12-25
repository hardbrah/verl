# vLLM Rollout 脚本使用指南

轻量化的 vLLM Rollout 脚本，用于高效的批量推理。

## 特点

- ✨ **高性能**: 使用 vLLM 引擎，相比 transformers 提供更高的吞吐量
- 🚀 **易用**: 支持命令行参数和配置文件两种方式
- 🎯 **灵活**: 支持自定义采样参数、多样本生成
- 💾 **简洁**: 输入输出都是 parquet 格式，方便数据处理
- 🔧 **可配置**: 支持张量并行、显存优化等高级特性

## 安装依赖

```bash
# 安装 vLLM（如果尚未安装）
pip install vllm

# 或者从源码安装最新版本
pip install git+https://github.com/vllm-project/vllm.git
```

## 快速开始

### 方法 1: 使用命令行参数

```bash
python scripts/vllm_rollout.py \
    --model_path /datacenter/models/Qwen/Qwen3-4B-Instruct-2507 \
    --input outputs/stage1_sampled_questions.parquet \
    --output outputs/vllm_rollout_output.parquet \
    --n_samples 8 \
    --max_tokens 2048
```

### 方法 2: 使用配置文件

```bash
# 1. 编辑配置文件
vim configs/vllm_rollout_config.yaml

# 2. 运行脚本
python scripts/vllm_rollout_with_config.py \
    --config configs/vllm_rollout_config.yaml
```

### 方法 3: 使用 Bash 脚本

```bash
# 使用默认参数
bash scripts/run_vllm_rollout.sh

# 或者通过环境变量自定义参数
MODEL_PATH=/path/to/model \
N_SAMPLES=16 \
MAX_TOKENS=4096 \
bash scripts/run_vllm_rollout.sh
```

## 输入输出格式

### 输入格式

输入文件必须是 parquet 格式，包含 `prompt` 列：

```python
import pandas as pd

# 示例 1: 字符串格式的 prompt
df = pd.DataFrame({
    'q_id': [0, 1, 2],
    'prompt': [
        "What is 2+2?",
        "Explain quantum computing.",
        "Write a poem about AI."
    ]
})

# 示例 2: Chat 格式的 prompt
df = pd.DataFrame({
    'q_id': [0, 1],
    'prompt': [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Explain quantum computing."}]
    ]
})

df.to_parquet('input.parquet', index=False)
```

### 输出格式

输出文件包含原始列 + `responses` 列：

```python
import pandas as pd

df = pd.read_parquet('output.parquet')

# df 包含:
# - 所有原始列（如 q_id, prompt 等）
# - responses: 列表，包含 n_samples 个生成的响应

# 示例：访问第一个问题的所有响应
responses_for_q0 = df.iloc[0]['responses']  # 长度为 n_samples 的列表
print(f"第 1 个响应: {responses_for_q0[0]}")
print(f"第 2 个响应: {responses_for_q0[1]}")
```

## 配置参数详解

### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | 必需 | 模型路径 |
| `trust_remote_code` | bool | True | 是否信任远程代码（Qwen 等模型需要） |

### 采样配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_samples` | int | 8 | 每个 prompt 生成多少个响应 |
| `max_tokens` | int | 2048 | 最大生成 token 数 |
| `temperature` | float | 1.0 | 采样温度（越低越确定性） |
| `top_p` | float | 0.95 | Nucleus sampling 参数 |
| `top_k` | int | -1 | Top-k sampling（-1 表示不使用） |
| `repetition_penalty` | float | 1.0 | 重复惩罚（1.0 表示无惩罚） |
| `seed` | int | 42 | 随机种子（确保可复现） |

### vLLM 引擎配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tensor_parallel_size` | int | 1 | 张量并行大小（GPU 数量） |
| `gpu_memory_utilization` | float | 0.9 | GPU 显存利用率（0-1） |
| `max_model_len` | int | None | 模型最大长度 |
| `dtype` | str | bfloat16 | 数据类型 |
| `enforce_eager` | bool | False | 强制 eager 模式 |
| `enable_prefix_caching` | bool | True | 启用前缀缓存 |
| `max_num_seqs` | int | 256 | 最大并发序列数 |

## 性能调优建议

### 单 GPU (如 A100 80GB)

```yaml
engine:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.95
  max_num_seqs: 256
  enable_prefix_caching: true
  dtype: bfloat16
```

**预期性能**: ~2000-3000 tokens/s

### 多 GPU (如 2x A100)

```yaml
engine:
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.95
  max_num_seqs: 512
  enable_prefix_caching: true
  dtype: bfloat16
```

**预期性能**: ~4000-6000 tokens/s

### 显存不足时

```yaml
engine:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8  # 降低显存使用
  max_num_seqs: 128  # 减少并发数
  dtype: bfloat16
```

### 提高吞吐量

```yaml
engine:
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.95
  max_num_seqs: 512  # 增加并发数
  enable_prefix_caching: true  # 启用缓存
  dtype: bfloat16  # 使用 bfloat16
```

## 常见问题

### 1. OOM (显存不足)

**问题**: 运行时报 CUDA out of memory 错误

**解决方案**:
```bash
# 降低显存利用率
python scripts/vllm_rollout.py \
    --gpu_memory_utilization 0.7 \
    --max_num_seqs 64 \
    ...
```

### 2. 速度太慢

**问题**: 生成速度不如预期

**解决方案**:
```bash
# 增加并发数和显存利用率
python scripts/vllm_rollout.py \
    --gpu_memory_utilization 0.95 \
    --max_num_seqs 512 \
    --enable_prefix_caching \
    ...
```

### 3. 多 GPU 不工作

**问题**: 使用多个 GPU 但速度没有提升

**解决方案**:
```bash
# 确保设置了正确的张量并行大小
python scripts/vllm_rollout.py \
    --tensor_parallel_size 2 \  # 应等于 GPU 数量
    ...
```

### 4. 结果不可复现

**问题**: 多次运行结果不一致

**解决方案**:
```bash
# 设置固定的随机种子
python scripts/vllm_rollout.py \
    --seed 42 \
    ...
```

## 与其他方案的对比

### vs. transformers (simple_generate.py)

| 特性 | vLLM Rollout | Transformers |
|------|--------------|--------------|
| 吞吐量 | 🚀 高（2-5x） | 一般 |
| 显存效率 | 💪 优秀 | 一般 |
| 易用性 | ✅ 简单 | ✅ 简单 |
| 批量推理 | ✅ 优化 | ❌ 未优化 |
| 启动时间 | ⏱️ 较慢 | ⏱️ 较快 |

**推荐使用场景**:
- **vLLM**: 大规模批量推理（>100 prompts）
- **transformers**: 小规模测试、调试

### vs. verl 完整框架

| 特性 | vLLM Rollout | verl 框架 |
|------|--------------|-----------|
| 功能 | 仅推理 | 完整 RLHF |
| 复杂度 | 🟢 低 | 🔴 高 |
| 依赖 | 少 | 多 |
| 灵活性 | 高 | 中 |
| 性能 | 优秀 | 优秀 |

**推荐使用场景**:
- **vLLM Rollout**: 只需要生成响应，不需要训练
- **verl 框架**: 需要完整的 RLHF 训练流程

## 高级用法

### 处理大规模数据

对于非常大的数据集，可以分批处理：

```python
import pandas as pd

# 读取大文件
df = pd.read_parquet('large_input.parquet')

# 分批处理
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size]
    batch_df.to_parquet(f'batch_{i}.parquet', index=False)
    
    # 处理每个批次
    os.system(f'python scripts/vllm_rollout.py \
        --input batch_{i}.parquet \
        --output batch_{i}_output.parquet \
        ...')

# 合并结果
results = []
for i in range(0, len(df), batch_size):
    results.append(pd.read_parquet(f'batch_{i}_output.parquet'))
final_df = pd.concat(results, ignore_index=True)
final_df.to_parquet('final_output.parquet', index=False)
```

### 自定义 prompt 格式

如果你的 prompt 列名不是 `prompt`：

```bash
python scripts/vllm_rollout.py \
    --prompt_key "my_custom_prompt_column" \
    ...
```

### 与 Ray 集成

对于超大规模分布式推理，可以与 Ray 集成：

```python
import ray
from vllm_rollout import vllm_rollout

ray.init(address='auto')

@ray.remote
def process_batch(input_path, output_path):
    vllm_rollout(
        model_path=...,
        input_parquet=input_path,
        output_parquet=output_path,
        ...
    )

# 并行处理多个批次
futures = [
    process_batch.remote(f'batch_{i}.parquet', f'output_{i}.parquet')
    for i in range(num_batches)
]
ray.get(futures)
```

## 示例：完整的工作流

```bash
#!/bin/bash
# 完整的 rollout 工作流示例

# 1. 准备数据
python scripts/prepare_data.py \
    --output outputs/questions.parquet

# 2. 执行 rollout
python scripts/vllm_rollout.py \
    --model_path /datacenter/models/Qwen/Qwen3-4B-Instruct-2507 \
    --input outputs/questions.parquet \
    --output outputs/responses.parquet \
    --n_samples 16 \
    --max_tokens 2048 \
    --temperature 1.0 \
    --top_p 0.95 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.95

# 3. 后处理
python scripts/post_process.py \
    --input outputs/responses.parquet \
    --output outputs/final_results.parquet

echo "完成！"
```

## 监控和调试

### 查看 vLLM 日志

vLLM 会输出详细的统计信息，包括：
- 吞吐量 (tokens/s)
- GPU 利用率
- 缓存命中率

### 性能分析

```bash
# 使用 nvidia-smi 监控 GPU
watch -n 1 nvidia-smi

# 使用 Python profiler
python -m cProfile -o profile.stats scripts/vllm_rollout.py ...
python -m pstats profile.stats
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

Apache License 2.0



