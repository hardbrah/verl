# Token 2048 最佳思考实验

## 实验目标

本实验旨在通过多次采样和评估，找到每个数学问题的"最佳思考路径"。

## 实验流程

### 整体设计

```
DAPO-MATH-17k (17k个问题)
    ↓ 随机采样1k
1000个问题
    ↓ 阶段1: 每个问题生成8次，max_new_tokens=2048
8000个 token_2048
    ↓ 阶段2: 为每个token_2048构建继续生成的prompt
8000个 continuation_prompt (原问题 + token_2048)
    ↓ 阶段3: 每个continuation_prompt生成100次，max_new_tokens=足够大
800,000个 完整rollout
    ↓ 阶段4: 计算每个token_2048的准确率并选择最佳
1000个 token_2048_best (每个问题一个)
```

## 核心原理解析

### 1. 什么是 token_2048？

`token_2048` 是模型在回答一个数学问题时，**前2048个token的思考过程**。

**例子：**
- 问题：计算 √(25) + 3²
- token_2048：模型生成的前2048个token，可能包含："首先，我需要计算两个部分。√(25) = 5，因为5×5=25。然后3² = 9..."

### 2. 为什么要"继续生成"？

在第一阶段，我们只让模型生成2048个token，可能还**没有得到最终答案**。我们需要让模型"继续说完"，得到完整的解答过程。

### 3. 如何实现"继续生成"？

**核心思想**：将已生成的2048个token拼接到原始prompt后面，让模型以为这是它"已经说过的话"，然后继续生成。

**技术细节**：

#### 第一阶段输入输出：
```
输入 (Prompt):
<|im_start|>user
计算 √(25) + 3²<|im_end|>
<|im_start|>assistant

输出 (token_2048):
首先，我需要计算两个部分...(2048个token)
```

#### 第二阶段构造继续生成的Prompt：
```
新输入 (Continuation Prompt):
<|im_start|>user
计算 √(25) + 3²<|im_end|>
<|im_start|>assistant
首先，我需要计算两个部分...(2048个token)

注意：这里没有 <|im_end|>，让模型认为assistant还在说话中
```

这样模型就会"接着往下说"，完成剩余的推理过程。

### 4. 为什么要采样8次和100次？

- **8次采样（阶段1）**：探索不同的思考路径。同一个问题可能有多种解题思路。
- **100次采样（阶段3）**：评估每条思考路径的"质量"。如果一条路径继续下去，100次中有80次能得到正确答案，说明这条路径质量高。

### 5. 如何评估 token_2048 的质量？

**token_2048_acc（准确率）**：

```
token_2048_acc = (继续生成100次中正确的次数) / 100
```

对于每个问题的8个token_2048，我们选择token_2048_acc最高的那个，作为这个问题的"最佳思考路径"。

## 与verl框架的关系

### 框架代码复用

1. **main_generation.py**：用于生成responses，无需修改
2. **math_dapo.py**：用于验证数学答案正确性，无需修改
3. **配置系统**：通过yaml配置控制生成参数，无需修改

### 我们的代码职责

我们的实验代码负责：
- 数据采样和准备
- 在两个阶段之间进行数据转换（prompt重构）
- 调用verl的生成模块
- 计算准确率和选择最佳路径

**设计原则**：外部编排，内部复用，保持verl框架的独立性。

## 目录结构

```
token_2048_best_thinking/
├── README.md                    # 本文档
├── configs/
│   ├── stage1_config.yaml      # 阶段1配置：采样8个token_2048
│   └── stage3_config.yaml      # 阶段3配置：继续生成100次
├── scripts/
│   ├── stage1_sample.py        # 阶段1：从DAPO-MATH采样并生成
│   ├── stage2_prepare_data.py  # 阶段2：构建continuation prompts
│   ├── stage3_continuation.py  # 阶段3：继续生成
│   └── stage4_select_best.py   # 阶段4：计算准确率并选择最佳
├── run_experiment.sh           # 完整实验运行脚本
└── outputs/                    # 输出目录（自动创建）
    ├── stage1_output.parquet   # 8k个token_2048
    ├── stage2_input.parquet    # 8k个continuation_prompt
    ├── stage3_output.parquet   # 800k个完整rollout
    └── final_best.parquet      # 1k个最佳token_2048
```

## 使用方法

### 前置条件

1. 确保已安装verl及其依赖：
```bash
cd /mnt/nas/chenhaotian/verl
pip install -e .
```

2. 准备DAPO-MATH数据集（如果还没有）：
```bash
bash recipe/dapo/prepare_dapo_data.sh
```

3. 准备Qwen3-4B-Instruct-2507模型（根据您的实际路径修改配置文件）

### 运行实验

#### 方式1：一键运行完整实验
```bash
cd /mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking
bash run_experiment.sh
```

#### 方式2：分阶段运行（便于调试）

**阶段1：采样并生成token_2048**
```bash
python scripts/stage1_sample.py
```
输出：`outputs/stage1_output.parquet` (8000条数据，每个问题8个token_2048)

**阶段2：准备continuation prompts**
```bash
python scripts/stage2_prepare_data.py
```
输出：`outputs/stage2_input.parquet` (8000条数据，每条包含拼接后的prompt)

**阶段3：继续生成**
```bash
python scripts/stage3_continuation.py
```
输出：`outputs/stage3_output.parquet` (800,000条数据，每个token_2048有100个完整rollout)

**阶段4：计算准确率并选择最佳**
```bash
python scripts/stage4_select_best.py
```
输出：`outputs/final_best.parquet` (1000条数据，每个问题一个最佳token_2048)

## 输出数据格式

### stage1_output.parquet
```python
{
    'question_id': 0,              # 问题ID (0-999)
    'question': "...",             # 原始问题
    'ground_truth': "...",         # 正确答案
    'sample_id': 0,                # 采样ID (0-7)
    'token_2048': "..."            # 生成的2048 tokens
}
```

### stage3_output.parquet
```python
{
    'question_id': 0,              # 问题ID
    'sample_id': 0,                # token_2048的采样ID
    'continuation_id': 0,          # 继续生成的采样ID (0-99)
    'token_2048': "...",           # 前2048个token
    'full_rollout': "...",         # 完整的rollout (token_2048 + 继续生成的部分)
    'ground_truth': "...",         # 正确答案
    'is_correct': True/False       # 是否正确
}
```

### final_best.parquet
```python
{
    'question_id': 0,              # 问题ID
    'question': "...",             # 原始问题
    'ground_truth': "...",         # 正确答案
    'best_sample_id': 3,           # 最佳token_2048的采样ID
    'best_token_2048': "...",      # 最佳token_2048
    'best_acc': 0.87,              # 最佳准确率 (87%)
    'all_accs': [0.45, 0.67, ...]  # 所有8个token_2048的准确率
}
```

## 配置说明

### 重要参数

#### stage1_config.yaml
- `data.path`: DAPO-MATH数据集路径
- `data.n_samples`: 8 (每个问题采样8次)
- `model.path`: Qwen3-4B-Instruct-2507模型路径
- `rollout.response_length`: 2048 (生成2048个token)
- `rollout.temperature`: 1.0 (采样温度)

#### stage3_config.yaml
- `data.n_samples`: 100 (每个token_2048继续生成100次)
- `rollout.response_length`: 4096 (足够大，确保生成完整)
- 其他参数与stage1保持一致

### 计算资源需求

**预估**：
- 阶段1：8 GPUs × ~1小时
- 阶段3：8 GPUs × ~10小时
- 总显存：建议每GPU至少40GB (A100/H100)

**优化建议**：
- 如果显存不足，可以调小batch_size
- 如果想加速实验，可以先用100个问题测试（修改stage1_sample.py中的sample_size）

## 实验结果分析

运行完成后，您可以使用以下脚本分析结果：

```python
import pandas as pd

# 加载最终结果
df = pd.read_parquet('outputs/final_best.parquet')

# 统计
print(f"平均最佳准确率: {df['best_acc'].mean():.2%}")
print(f"最高准确率: {df['best_acc'].max():.2%}")
print(f"最低准确率: {df['best_acc'].min():.2%}")

# 查看分布
import matplotlib.pyplot as plt
df['best_acc'].hist(bins=20)
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Distribution of Best Token_2048 Accuracy')
plt.savefig('accuracy_distribution.png')
```

## 常见问题

### Q1: 为什么不直接让模型生成完整答案？
A: 我们想研究"思考过程的前半部分"对"最终结果"的影响。不同的起始思考路径，会导致不同的成功率。

### Q2: 如果第一阶段就生成了完整答案怎么办？
A: 这是正常的。第二阶段继续生成时，模型可能会输出<|im_end|>或者重复内容。我们的评估脚本会正确处理这种情况。

### Q3: 为什么要采样100次这么多？
A: 统计学上，样本越多，准确率估计越准确。如果资源有限，可以改成50次或30次，但会降低估计的可靠性。

### Q4: 这个实验的意义是什么？
A: 这个实验可以帮助我们理解：
- 哪些早期思考路径更容易导向正确答案
- 训练数据中哪些"好的开头"值得学习
- 如何设计更好的推理策略

## 技术细节补充

### Chat Template格式

Qwen模型使用ChatML格式：
```
<|im_start|>user
{用户问题}<|im_end|>
<|im_start|>assistant
{助手回答}<|im_end|>
```

在继续生成时，我们故意不加最后的`<|im_end|>`，让模型认为回答还没结束。

### 并行化策略

- 阶段1和阶段3都使用vLLM进行高效并行生成
- Ray分布式框架自动处理多GPU负载均衡
- 无需手动管理进程和显存

### 错误处理

脚本包含完善的错误处理：
- 数据不完整：自动跳过
- 生成失败：记录日志并继续
- OOM错误：自动降低batch size（需手动调整配置）

## 参考文献

- DAPO: https://arxiv.org/abs/2410.07524
- verl框架: https://github.com/volcengine/verl
- Qwen模型: https://github.com/QwenLM/Qwen

## 作者与维护

如有问题，请提Issue或联系项目维护者。

