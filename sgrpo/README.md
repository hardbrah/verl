# S-GRPO: Sampling-based GRPO with Decaying Reward

S-GRPO是一种基于GRPO的改进算法，通过截断思维链采样和衰减奖励机制来鼓励模型产生更简洁有效的推理过程。

run_sgrpo_example.sh
    └── sgrpo/main_sgrpo.py
            ├── sgrpo/ray_trainer.py (RaySGRPOTrainer)
            │       ├── verl/trainer/ppo/ray_trainer.py (RayPPOTrainer)
            │       │       ├── verl/protocol.py (DataProto)
            │       │       ├── verl/single_controller/ray/ (RayWorkerGroup)
            │       │       ├── verl/trainer/ppo/core_algos.py
            │       │       ├── verl/trainer/ppo/metric_utils.py
            │       │       └── verl/utils/* (各种工具)
            │       │
            │       ├── sgrpo/core_algos.py (compute_sgrpo_advantage)
            │       └── sgrpo/decaying_reward.py (SGRPORewardManager)
            │               └── verl/workers/reward_manager/abstract.py
            │
            ├── verl/trainer/ppo/reward.py (load_reward_manager)
            ├── verl/utils/dataset/rl_dataset.py (collate_fn, RLHFDataset)
            ├── verl/workers/fsdp_workers.py (ActorRolloutRefWorker)
            └── sgrpo/config/sgrpo_trainer.yaml

## 与GRPO的核心区别

| 特性 | GRPO | S-GRPO |
|------|------|--------|
| **Group构成** | 对一个query采样N个完整响应 | 对一个query采样1个完整响应，然后截断m次 |
| **每个query的样本数** | N | m+1（m个截断 + 1个完整）|
| **奖励函数** | 标准奖励 | 衰减奖励（鼓励更短的正确答案） |
| **优势估计** | (r - mean) / std | r - mean（不除以std） |

## 算法概述

### 核心思想

对于每个query，S-GRPO执行以下步骤：

1. **生成完整响应（CoT0）**：对query进行一次rollout，生成一个完整的chain-of-thought响应，设其token数为n。

2. **均匀采样截断点**：根据超参数m，从[1, n]中均匀采样m个位置（从小到大排列）。

3. **创建截断思维链**：使用采样的m个位置截断响应，得到m个长度递增的截断思维链：CoT1, CoT2, ..., CoTm。

4. **强制输出答案**：将每个截断思维链拼接提示词 "Time is limited, stop thinking and start answering.\n</think>\n\n"，让模型从截断点继续生成答案。

5. **计算衰减奖励**：
   - 将所有响应按**CoT1, CoT2, ..., CoTm, CoT0**的顺序排列（按长度从短到长）
   - 对于每个answer_i，如果正确：`reward_i = 1/(2^N_right)`，其中N_right是到当前位置为止的正确答案累计数量
   - 如果答案错误：`reward_i = 0`

6. **GRPO优势估计**：将这m+1个响应作为一个group，使用GRPO的优势估计方法（但不除以标准差）计算advantage，然后更新模型。

### 衰减奖励的计算

衰减奖励的关键是：**N_right 表示在当前答案之前（不包括当前答案）的正确答案数量**

| 答案顺序 | 是否正确 | N_right (之前正确数) | 奖励 |
|---------|---------|---------------------|------|
| answer_1 | ✓ | 0 | 1/(2^0) = **1.0** |
| answer_2 | ✗ | - | 0 |
| answer_3 | ✓ | 1 | 1/(2^1) = **0.5** |
| answer_4 | ✓ | 2 | 1/(2^2) = **0.25** |
| answer_5 | ✗ | - | 0 |

### 衰减奖励的直觉

衰减奖励设计鼓励模型：
- **更早产生正确答案**：第一个正确答案获得最高奖励1分
- 较短的思维链如果能产生正确答案，将获得更高奖励
- 错误答案始终获得0分

## 文件结构

```
sgrpo/
├── __init__.py              # 模块入口
├── core_algos.py            # 核心算法（衰减奖励、S-GRPO优势估计）
├── data_processor.py        # 数据处理（截断采样、强制回答输入准备）
├── decaying_reward.py       # 衰减奖励管理器
├── ray_trainer.py           # S-GRPO分布式训练器
├── main_sgrpo.py           # 主入口
├── config/
│   ├── __init__.py
│   └── sgrpo_trainer.yaml  # 配置文件
├── examples/
│   ├── __init__.py
│   └── run_sgrpo_example.sh # 示例运行脚本
└── README.md               # 本文档
```

## 快速开始

### 安装

确保已安装verl框架：

```bash
cd /path/to/verl
pip install -e .
```

### 配置

主要的S-GRPO特定配置参数：

```yaml
sgrpo:
  # 截断点数量（m）
  num_truncations: 4
  
  # 强制回答提示词
  force_answer_prompt: "Time is limited, stop thinking and start answering.\n</think>\n\n"
  
  # 强制回答的最大token数
  answer_max_tokens: 256
  
  # 截断比例范围
  min_truncation_ratio: 0.1
  max_truncation_ratio: 0.9

algorithm:
  # 使用S-GRPO优势估计器
  adv_estimator: sgrpo
  
  # 重要：S-GRPO不使用标准差归一化
  norm_adv_by_std_in_grpo: False
```

### 运行训练

```bash
# 使用示例脚本
bash sgrpo/examples/run_sgrpo_example.sh

# 或者直接使用Python
python -m sgrpo.main_sgrpo \
    actor_rollout_ref.model.path="your_model_path" \
    data.train_files="your_train_data.parquet" \
    data.val_files="your_val_data.parquet" \
    sgrpo.num_truncations=4 \
    algorithm.adv_estimator="sgrpo"
```

### WandB日志和样本记录

S-GRPO支持详细的训练日志记录：

```bash
python -m sgrpo.main_sgrpo \
    # ... 其他配置 ...
    trainer.logger='["console","wandb"]' \
    trainer.project_name="sgrpo_project" \
    trainer.experiment_name="run_1" \
    trainer.rollout_data_dir="./outputs/sgrpo_samples" \
    trainer.log_val_generations=5
```

#### 配置说明

| 配置项 | 说明 | 示例值 |
|-------|------|--------|
| `trainer.logger` | 日志后端 | `'["console","wandb"]'` |
| `trainer.rollout_data_dir` | 样本保存基础目录（启用详细样本日志） | `"./outputs/sgrpo_samples"` |
| `trainer.rollout_data_dump_freq` | 样本保存频率（每N步保存一次，-1禁用） | `20` |
| `trainer.log_val_generations` | WandB中记录的验证样本数 | `5` |

> **注意**：样本会自动保存到 `{rollout_data_dir}/{experiment_name}/` 子目录中，不同实验的样本不会相互覆盖。例如，如果设置 `experiment_name='qwen3_8b_sgrpo'`，样本会保存到 `./outputs/sgrpo_samples/qwen3_8b_sgrpo/` 目录。

#### 保存的样本内容

当设置`rollout_data_dir`后，每个训练步骤会保存JSONL文件，包含：

```json
{
  "step": 100,
  "sample_idx": 0,
  "original_prompt": "问题: 计算 3 + 5 = ?",
  "complete_response_cot0": "<think>让我计算一下...</think>答案是8",
  "ground_truth": "8",
  "truncations": [
    {
      "truncation_idx": 1,
      "type": "cot_1",
      "truncated_input": "问题: 计算 3 + 5 = ?\n<think>让...\nTime is limited, stop thinking...",
      "response": "8",
      "extracted_answer": "8",
      "is_correct": true,
      "decaying_reward": 1.0
    },
    {
      "truncation_idx": 2,
      "type": "cot_2",
      "truncated_input": "...",
      "response": "...",
      "extracted_answer": "8",
      "is_correct": true,
      "decaying_reward": 0.5
    },
    // ... 更多截断 ...
    {
      "truncation_idx": 0,
      "type": "cot_0 (complete)",
      "response": "<think>让我计算一下...</think>答案是8",
      "extracted_answer": "8",
      "is_correct": true,
      "decaying_reward": 0.25
    }
  ]
}
```

#### WandB中记录的指标

| 指标 | 说明 |
|------|------|
| `sgrpo/accuracy` | 所有响应的整体准确率 |
| `sgrpo/accuracy_cot_1` | 最短截断(CoT1)的准确率 |
| `sgrpo/accuracy_cot_2` | 第二短截断(CoT2)的准确率 |
| ... | |
| `sgrpo/accuracy_cot_0_complete` | 完整响应(CoT0)的准确率 |
| `sgrpo/mean_decaying_reward` | 平均衰减奖励 |
| `sgrpo/mean_reward_cot_1` | CoT1的平均奖励 |
| `sgrpo/num_truncations` | 截断数量m |
| `sgrpo/original_batch_size` | 原始batch大小 |
| `sgrpo/expanded_batch_size` | 扩展后的batch大小 (batch_size × (m+1)) |

## API 参考

### 核心函数

#### `compute_decaying_reward`

```python
from sgrpo import compute_decaying_reward

rewards = compute_decaying_reward(
    answers=["ans1", "ans2", "ans3", "ans4", "ans0"],  # 按CoT长度排序
    verify_fn=your_verify_function,
    ground_truth="correct_answer"
)
```

#### `compute_sgrpo_advantage`

```python
from sgrpo import compute_sgrpo_advantage

advantages, returns = compute_sgrpo_advantage(
    token_level_rewards=reward_tensor,
    response_mask=mask_tensor,
    index=uid_array,
)
```

### 数据处理

#### `SGRPODataProcessor`

```python
from sgrpo import SGRPODataProcessor

processor = SGRPODataProcessor(
    tokenizer=tokenizer,
    num_truncations=4,
    force_answer_prompt="Time is limited...",
)

# 创建截断序列
truncated_seqs = processor.create_truncated_sequences(
    prompt_ids=prompt_tensor,
    response_ids=response_tensor,
    sample_index=0,
)

# 准备强制回答输入
force_answer_batch = processor.prepare_force_answer_inputs(truncated_seqs)
```

### 奖励管理器

#### `SGRPORewardManager`

```python
from sgrpo import create_sgrpo_reward_manager

reward_manager = create_sgrpo_reward_manager(
    config=config,
    tokenizer=tokenizer,
    num_truncations=4,
)

result = reward_manager(batch, return_dict=True)
reward_tensor = result["reward_tensor"]
```

## 算法流程图

```
Query
  │
  ▼
┌─────────────────────────────────────┐
│ Step 1: 生成完整响应 CoT0 (n tokens) │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Step 2: 均匀采样m个截断位置          │
│         positions = sample([1,n], m) │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: 创建m个截断输入                                  │
│         CoT_i = prompt + truncated_response[:pos_i]     │
│                + "Time is limited..."                   │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Step 4: 生成m个强制回答              │
│         answer_i = generate(CoT_i)   │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: 计算衰减奖励                                     │
│         对于 [answer1, ..., answerm, answer0]           │
│         if correct: reward = 1/(2^累计正确数)            │
│         if wrong:   reward = 0                          │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 6: S-GRPO优势估计 (不除以std)                       │
│         advantage_i = reward_i - mean(rewards in group) │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Step 7: 更新模型                     │
└─────────────────────────────────────┘
```

## 参考

- GRPO: Group Relative Policy Optimization
- Dr.GRPO: 无标准差归一化的GRPO变体

## License

Apache 2.0
