# vLLM Rollout 脚本集合

轻量化的 vLLM Rollout 脚本，用于高效的批量推理。

## 📦 包含的文件

### 核心脚本
- **`vllm_rollout.py`**: 主要的 vLLM rollout 脚本（命令行版本）
- **`vllm_rollout_with_config.py`**: 使用配置文件的版本
- **`process_rollout_results.py`**: 结果处理工具（展平、分析、导出等）
- **`test_vllm_rollout.py`**: 快速测试脚本

### 辅助脚本
- **`run_vllm_rollout.sh`**: Bash 启动脚本
- **`vllm_shortcuts.sh`**: 快捷命令脚本

### 配置文件
- **`../configs/vllm_rollout_config.yaml`**: YAML 配置文件模板

### 文档
- **`../docs/VLLM_QUICKSTART.md`**: 快速开始指南（5 分钟上手）
- **`../docs/VLLM_ROLLOUT_GUIDE.md`**: 完整使用指南
- **`README_VLLM_ROLLOUT.md`**: 本文件

## 🚀 快速开始

### 1️⃣ 最简单的方式（快捷命令）

```bash
# 快速测试
bash scripts/vllm_shortcuts.sh test

# 演示运行
bash scripts/vllm_shortcuts.sh rollout-demo
```

### 2️⃣ 命令行方式

```bash
python scripts/vllm_rollout.py \
    --model_path /datacenter/models/Qwen/Qwen3-4B-Instruct-2507 \
    --input your_input.parquet \
    --output your_output.parquet \
    --n_samples 8 \
    --max_tokens 2048
```

### 3️⃣ 配置文件方式

```bash
# 编辑配置
vim configs/vllm_rollout_config.yaml

# 运行
python scripts/vllm_rollout_with_config.py \
    --config configs/vllm_rollout_config.yaml \
    --input your_input.parquet \
    --output your_output.parquet
```

详细说明请查看 [快速开始指南](../docs/VLLM_QUICKSTART.md)

## ✨ 主要特性

- ✅ **高性能**: 使用 vLLM 引擎，吞吐量提升 2-5x
- ✅ **易用**: 支持命令行、配置文件、Bash 脚本多种方式
- ✅ **灵活**: 丰富的采样参数配置
- ✅ **多 GPU**: 支持张量并行
- ✅ **完整工具链**: 包含数据处理、分析、导出等工具
- ✅ **详细文档**: 快速开始 + 完整指南

## 📊 性能对比

| 方案 | 吞吐量 | 显存效率 | 适用场景 |
|------|--------|----------|----------|
| vLLM Rollout | 🚀🚀🚀 高 | 💪 优秀 | 大规模批量推理 |
| Transformers | ⚡ 一般 | 👍 一般 | 小规模测试 |
| verl 完整框架 | 🚀🚀🚀 高 | 💪 优秀 | 完整 RLHF 训练 |

**推荐场景**: 只需要生成响应，不需要训练的场景

## 📖 文档

- 🚀 [快速开始指南](../docs/VLLM_QUICKSTART.md) - 5 分钟快速上手
- 📚 [完整使用指南](../docs/VLLM_ROLLOUT_GUIDE.md) - 详细配置和高级用法
- ⚙️ [配置文件示例](../configs/vllm_rollout_config.yaml)

## 🛠️ 工具使用

### 主脚本: vllm_rollout.py

```bash
# 查看帮助
python scripts/vllm_rollout.py --help

# 基本用法
python scripts/vllm_rollout.py \
    --model_path MODEL_PATH \
    --input INPUT.parquet \
    --output OUTPUT.parquet \
    [可选参数...]

# 重要参数:
#   --n_samples N          每个 prompt 生成多少个响应
#   --max_tokens N         最大生成 token 数
#   --temperature FLOAT    采样温度
#   --top_p FLOAT          Top-p sampling
#   --tensor_parallel_size N  张量并行大小（GPU 数量）
#   --gpu_memory_utilization FLOAT  显存利用率（0-1）
```

### 结果处理: process_rollout_results.py

```bash
# 查看帮助
python scripts/process_rollout_results.py --help

# 展平响应（将多个响应展开为多行）
python scripts/process_rollout_results.py flatten \
    --input output.parquet \
    --output flattened.parquet

# 统计分析
python scripts/process_rollout_results.py analyze \
    --input output.parquet

# 导出 JSONL
python scripts/process_rollout_results.py export \
    --input output.parquet \
    --output output.jsonl

# 按长度过滤
python scripts/process_rollout_results.py filter \
    --input flattened.parquet \
    --output filtered.parquet \
    --min_length 100 \
    --max_length 5000
```

### 快捷命令: vllm_shortcuts.sh

```bash
# 查看帮助
bash scripts/vllm_shortcuts.sh help

# 快速测试
bash scripts/vllm_shortcuts.sh test

# 演示运行
bash scripts/vllm_shortcuts.sh rollout-demo

# 分析结果
bash scripts/vllm_shortcuts.sh analyze output.parquet

# 展平结果
bash scripts/vllm_shortcuts.sh flatten output.parquet flat.parquet

# 导出 JSONL
bash scripts/vllm_shortcuts.sh export output.parquet output.jsonl
```

## 📝 输入输出格式

### 输入格式

Parquet 文件，必须包含 `prompt` 列：

```python
import pandas as pd

df = pd.DataFrame({
    'q_id': [0, 1],
    'prompt': [
        [{"role": "user", "content": "Question 1"}],
        [{"role": "user", "content": "Question 2"}]
    ]
})
```

### 输出格式

Parquet 文件，包含原始列 + `responses` 列：

```python
# 每行的 responses 是一个列表，包含 n_samples 个响应
df = pd.read_parquet('output.parquet')
responses = df.iloc[0]['responses']  # 列表，长度为 n_samples
```

## 🔧 常见配置

### 单 GPU (A100 80GB)
```bash
python scripts/vllm_rollout.py \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --max_num_seqs 256 \
    ...
```

### 多 GPU (2x A100)
```bash
python scripts/vllm_rollout.py \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.95 \
    --max_num_seqs 512 \
    ...
```

### 显存不足
```bash
python scripts/vllm_rollout.py \
    --gpu_memory_utilization 0.7 \
    --max_num_seqs 64 \
    ...
```

## 🐛 常见问题

### Q: OOM (显存不足)
```bash
# 降低显存使用
--gpu_memory_utilization 0.7 --max_num_seqs 64
```

### Q: 速度太慢
```bash
# 增加并发和缓存
--max_num_seqs 512 --enable_prefix_caching
```

### Q: 多 GPU 不工作
```bash
# 确保设置正确的张量并行
--tensor_parallel_size 2  # 应等于 GPU 数量
```

### Q: 结果不可复现
```bash
# 设置固定种子
--seed 42
```

## 📦 依赖安装

```bash
# 安装 vLLM
pip install vllm

# 或从源码安装
pip install git+https://github.com/vllm-project/vllm.git

# 验证安装
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

## 📈 性能参考

### 吞吐量（tokens/s）
- 单 GPU (A100 80GB): ~2000-3000 tokens/s
- 双 GPU (2x A100): ~4000-6000 tokens/s

### 预估时间（平均每响应 1000 tokens）

| Prompts | Samples | 单 GPU | 双 GPU |
|---------|---------|--------|--------|
| 100     | 8       | ~5 min | ~2 min |
| 1,000   | 8       | ~45 min| ~20 min|
| 10,000  | 8       | ~7 hr  | ~3.5 hr|

## 🎯 完整工作流示例

```bash
#!/bin/bash
# 1. 准备输入数据
python prepare_data.py --output inputs.parquet

# 2. 执行 rollout
python scripts/vllm_rollout.py \
    --model_path /path/to/model \
    --input inputs.parquet \
    --output outputs.parquet \
    --n_samples 16 \
    --max_tokens 2048

# 3. 分析结果
python scripts/process_rollout_results.py analyze \
    --input outputs.parquet

# 4. 展平并过滤
python scripts/process_rollout_results.py flatten \
    --input outputs.parquet \
    --output flat.parquet

python scripts/process_rollout_results.py filter \
    --input flat.parquet \
    --output filtered.parquet \
    --min_length 100

# 5. 导出
python scripts/process_rollout_results.py export \
    --input filtered.parquet \
    --output final.jsonl

echo "✓ 完成！"
```

## 📄 许可证

Apache License 2.0

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**快速链接**:
- [快速开始](../docs/VLLM_QUICKSTART.md)
- [完整指南](../docs/VLLM_ROLLOUT_GUIDE.md)
- [配置文件](../configs/vllm_rollout_config.yaml)



