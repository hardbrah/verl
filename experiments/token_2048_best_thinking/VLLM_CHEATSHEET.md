# vLLM Rollout 快速参考卡片

> 一页纸速查表 - 打印或保存到桌面

## 🚀 一分钟开始

```bash
# 1. 快速测试（推荐首次使用）
bash scripts/vllm_shortcuts.sh test

# 2. 基本使用
python scripts/vllm_rollout.py \
    --model_path /path/to/model \
    --input input.parquet \
    --output output.parquet

# 3. 查看帮助
python scripts/vllm_rollout.py --help
```

## 📋 常用命令

### 快捷命令（最简单）

```bash
bash scripts/vllm_shortcuts.sh test          # 快速测试
bash scripts/vllm_shortcuts.sh rollout-demo  # 演示运行
bash scripts/vllm_shortcuts.sh analyze FILE  # 分析结果
bash scripts/vllm_shortcuts.sh flatten IN OUT # 展平结果
```

### 主脚本

```bash
# 基本
python scripts/vllm_rollout.py \
    --model_path MODEL --input IN --output OUT

# 自定义采样
python scripts/vllm_rollout.py \
    --model_path MODEL --input IN --output OUT \
    --n_samples 16 --max_tokens 4096 --temperature 0.8

# 多 GPU
python scripts/vllm_rollout.py \
    --model_path MODEL --input IN --output OUT \
    --tensor_parallel_size 2 --gpu_memory_utilization 0.95
```

### 结果处理

```bash
# 展平（每个响应一行）
python scripts/process_rollout_results.py flatten \
    --input output.parquet --output flat.parquet

# 统计分析
python scripts/process_rollout_results.py analyze \
    --input output.parquet

# 导出 JSONL
python scripts/process_rollout_results.py export \
    --input output.parquet --output output.jsonl

# 按长度过滤
python scripts/process_rollout_results.py filter \
    --input flat.parquet --output filtered.parquet \
    --min_length 100 --max_length 5000
```

## ⚙️ 重要参数

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_samples` | 8 | 每个 prompt 生成几个响应 |
| `--max_tokens` | 2048 | 最大生成 token 数 |
| `--temperature` | 1.0 | 采样温度（越低越确定） |
| `--top_p` | 0.95 | Top-p sampling |
| `--seed` | 42 | 随机种子（可复现） |

### 性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tensor_parallel_size` | 1 | GPU 数量 |
| `--gpu_memory_utilization` | 0.9 | 显存利用率 (0-1) |
| `--max_num_seqs` | 256 | 最大并发序列数 |
| `--enable_prefix_caching` | True | 启用前缀缓存 |

## 🎛️ 常用配置

### 单 GPU 高性能
```bash
--tensor_parallel_size 1 \
--gpu_memory_utilization 0.95 \
--max_num_seqs 256
```

### 双 GPU 高性能
```bash
--tensor_parallel_size 2 \
--gpu_memory_utilization 0.95 \
--max_num_seqs 512
```

### 显存不足
```bash
--gpu_memory_utilization 0.7 \
--max_num_seqs 64
```

### 调试模式
```bash
--enforce_eager \
--gpu_memory_utilization 0.5
```

## 📁 文件格式

### 输入 (Parquet)

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': [
        [{"role": "user", "content": "问题1"}],
        [{"role": "user", "content": "问题2"}]
    ]
})
df.to_parquet('input.parquet')
```

### 输出 (Parquet)

```python
df = pd.read_parquet('output.parquet')
# df['responses'] 是列表，包含 n_samples 个响应
print(df.iloc[0]['responses'])  # 第一个 prompt 的所有响应
```

## 🐛 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| **OOM 显存不足** | `--gpu_memory_utilization 0.7 --max_num_seqs 64` |
| **速度太慢** | `--max_num_seqs 512 --enable_prefix_caching` |
| **多 GPU 不工作** | 检查 `--tensor_parallel_size` 是否等于 GPU 数 |
| **结果不可复现** | 设置 `--seed 42` |
| **CUDA error** | 降低 `--max_num_seqs` 或 `--gpu_memory_utilization` |

## 📊 性能参考

### 吞吐量 (tokens/s)

- **单 A100**: ~2000-3000
- **双 A100**: ~4000-6000

### 时间预估（1000 prompts × 8 samples × 1000 tokens）

- **单 A100**: ~45 分钟
- **双 A100**: ~20 分钟

## 🔗 快速链接

| 资源 | 位置 |
|------|------|
| 快速开始 | `docs/VLLM_QUICKSTART.md` |
| 完整指南 | `docs/VLLM_ROLLOUT_GUIDE.md` |
| 配置模板 | `configs/vllm_rollout_config.yaml` |
| 脚本说明 | `scripts/README_VLLM_ROLLOUT.md` |

## 🎯 完整工作流

```bash
# 1. 准备数据
python prepare_data.py --output input.parquet

# 2. 执行 rollout
python scripts/vllm_rollout.py \
    --model_path /path/to/model \
    --input input.parquet \
    --output output.parquet \
    --n_samples 8 --max_tokens 2048

# 3. 分析结果
python scripts/process_rollout_results.py analyze \
    --input output.parquet

# 4. 展平并过滤
python scripts/process_rollout_results.py flatten \
    --input output.parquet --output flat.parquet

python scripts/process_rollout_results.py filter \
    --input flat.parquet --output filtered.parquet \
    --min_length 100

# 5. 导出
python scripts/process_rollout_results.py export \
    --input filtered.parquet --output final.jsonl
```

## 💡 最佳实践

1. **首次使用**: 先运行 `bash scripts/vllm_shortcuts.sh test`
2. **调优性能**: 从保守配置开始，逐步增加并发数
3. **显存管理**: 监控 `nvidia-smi`，避免 OOM
4. **可复现性**: 始终设置固定的 `--seed`
5. **批处理**: 对大数据集分批处理，避免一次性加载

## 📞 获取帮助

```bash
# 命令行帮助
python scripts/vllm_rollout.py --help
python scripts/process_rollout_results.py --help
bash scripts/vllm_shortcuts.sh help

# 查看示例
cat docs/VLLM_QUICKSTART.md
cat docs/VLLM_ROLLOUT_GUIDE.md
```

---

**版本**: v1.0.0 | **更新**: 2025-12-25









