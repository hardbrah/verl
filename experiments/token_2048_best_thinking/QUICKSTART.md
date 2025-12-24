# 快速入门指南

本文档提供最简化的实验运行步骤。完整文档请参考 [README.md](README.md)。

## 5分钟快速开始

### 步骤1：配置模型路径

编辑配置文件，设置您的Qwen3-4B模型路径：

```bash
# 编辑阶段1配置
vim configs/stage1_config.yaml
# 修改第20行：path: ~/models/Qwen3-4B-Instruct-2507
# 改为您的实际模型路径

# 编辑阶段3配置（确保与阶段1一致）
vim configs/stage3_config.yaml
# 同样修改model.path
```

### 步骤2：检查GPU配置

确保配置文件中的GPU数量与您的实际环境匹配：

```bash
# configs/stage1_config.yaml 和 stage3_config.yaml
# 第4行：n_gpus_per_node: 8  # 改为您的GPU数量
```

### 步骤3：运行实验

```bash
# 一键运行完整实验
bash run_experiment.sh
```

就这么简单！脚本会自动完成所有4个阶段。

## 如果想分阶段运行

有时您可能想分阶段运行以便调试或中断：

```bash
# 阶段1：采样并生成token_2048（约1小时）
python scripts/stage1_sample.py

# 阶段2：准备数据（几秒钟）
python scripts/stage2_prepare_data.py

# 阶段3：继续生成（约10小时）
python scripts/stage3_continuation.py

# 阶段4：分析结果（几分钟）
python scripts/stage4_select_best.py
```

## 常见问题速查

### Q: 模型路径不存在？
**A:** 确保您已下载Qwen3-4B-Instruct-2507模型，并在配置文件中设置正确路径。

### Q: 显存不足（OOM）？
**A:** 减小batch_size：
- `stage1_config.yaml`: 将 `data.batch_size` 从 16 改为 8 或 4
- `stage3_config.yaml`: 将 `data.batch_size` 从 8 改为 4 或 2

### Q: 没有DAPO-MATH数据集？
**A:** 脚本会自动从HuggingFace下载。如果网络不通，运行：
```bash
cd /mnt/nas/chenhaotian/verl
bash recipe/dapo/prepare_dapo_data.sh
```

### Q: 时间太长，能减少数据量吗？
**A:** 可以！编辑 `scripts/stage1_sample.py`：
```python
# 第124行左右，将sample_size改小
sample_size=100,  # 原来是1000，改为100测试
```
或者减少采样次数：
- `stage1_config.yaml`: `n_samples: 2` (原来是8)
- `stage3_config.yaml`: `n_samples: 10` (原来是100)

### Q: 如何查看中间结果？
**A:** 使用pandas：
```python
import pandas as pd

# 查看阶段1输出
df1 = pd.read_parquet('outputs/stage1_output.parquet')
print(df1.head())

# 查看阶段3输出
df3 = pd.read_parquet('outputs/stage3_output.parquet')
print(f"总体准确率: {df3['is_correct'].mean():.2%}")

# 查看最终结果
df_final = pd.read_parquet('outputs/final_best.parquet')
print(df_final[['question_id', 'best_acc']].head(10))
```

## 预期输出

实验成功后，您应该看到：

```
outputs/
├── stage1_output.parquet          # 8000条 (1000个问题 × 8个采样)
├── stage2_input.parquet            # 8000条 (continuation prompts)
├── stage3_output.parquet           # 800000条 (8000 × 100)
├── final_best.parquet              # 1000条 (每个问题的最佳token_2048)
├── analysis_report.txt             # 详细分析报告
└── accuracy_distribution.png       # 可视化图表
```

## 查看结果

```bash
# 查看分析报告
cat outputs/analysis_report.txt

# 快速统计
python -c "
import pandas as pd
df = pd.read_parquet('outputs/final_best.parquet')
print(f'平均最佳准确率: {df[\"best_acc\"].mean():.2%}')
print(f'最高准确率: {df[\"best_acc\"].max():.2%}')
print(f'最低准确率: {df[\"best_acc\"].min():.2%}')
"
```

## 故障排查

### Ray初始化失败
```bash
# 清理Ray缓存
ray stop
rm -rf /tmp/ray
```

### vLLM错误
确保安装了正确版本的vLLM：
```bash
pip install vllm>=0.8.5
```

### 其他问题
查看日志文件（如果有），或者查看完整的[README.md](README.md)。

## 实验时间估算

| 阶段 | 数据量 | 预计时间 (8×A100) |
|------|--------|-------------------|
| 阶段1 | 1000 → 8000 | ~1小时 |
| 阶段2 | 数据处理 | ~10秒 |
| 阶段3 | 8000 → 800000 | ~10小时 |
| 阶段4 | 分析 | ~5分钟 |
| **总计** | | **~11小时** |

小规模测试（100个问题）约需1-2小时。

## 联系与支持

如有问题，请查看：
1. [README.md](README.md) - 完整文档
2. [verl官方文档](https://github.com/volcengine/verl)

祝实验顺利！🎉

