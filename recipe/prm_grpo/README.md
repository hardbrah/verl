# PRM-GRPO: GRPO with Process Reward Model (Latent States)

## 概述

PRM-GRPO 是一个基于标准 GRPO (Group Relative Policy Optimization) 的变体，使用 Process Reward Model (PRM) 通过 **latent states** 来评估截断的思维链质量。

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      PRM-GRPO Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Question + Truncated Thinking Chain                         │
│              │                                               │
│              ▼                                               │
│  ┌───────────────────────────────────┐                      │
│  │     Actor (Qwen3-4B-Instruct)     │                      │
│  │                                   │                      │
│  │   Layer 0  ─────────────────┐     │                      │
│  │   Layer 1                   │     │                      │
│  │   ...                       │     │                      │
│  │   Layer 15 ─────────────────┼──── │ ──► Latent States    │
│  │   ...                       │     │     [batch, seq, 2560]│
│  │   Layer N                   │     │                      │
│  └───────────────────────────────────┘                      │
│              │                                               │
│              ▼                                               │
│  ┌───────────────────────────────────┐                      │
│  │     PRM Backbone (Qwen3-4B)       │                      │
│  │                                   │                      │
│  │   inputs_embeds ◄── Latent States │                      │
│  │   Layer 0  ◄────────────────┘     │                      │
│  │   Layer 1                         │                      │
│  │   ...                             │                      │
│  │   Layer N                         │                      │
│  │         │                         │                      │
│  │         ▼                         │                      │
│  │   Last Token Hidden State         │                      │
│  │         │                         │                      │
│  │         ▼                         │                      │
│  │   Regression Head                 │                      │
│  │   (2560 → 2560 → 1)               │                      │
│  │         │                         │                      │
│  │         ▼                         │                      │
│  │   sigmoid(score) ∈ [0, 1]         │                      │
│  └───────────────────────────────────┘                      │
│              │                                               │
│              ▼                                               │
│        PRM Score (Process Reward)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 与标准 GRPO 的区别

| 特性 | 标准 GRPO | PRM-GRPO |
|------|-----------|----------|
| Reward 类型 | Outcome reward (最终答案正确性) | Process reward (思维链开头质量) |
| 验证方式 | 规则验证器 (math verifier) | PRM 打分模型 (latent states) |
| Response 长度 | 完整解答 (1024+ tokens) | 截断的思维链开头 (~64 tokens) |
| 评估维度 | 最终答案是否正确 | 思维过程开头的质量 |
| 输入形式 | Token IDs | Latent States (hidden states) |

## PRM 模型详解

### 模型结构

PRM 基于 `/data/models/Qwen/Qwen3-4B-Instruct-2507` 模型：

- **Latent Extractor**: 从 Actor 的第 15 层提取 hidden states
- **PRM Backbone**: Qwen3-4B-Instruct，接收 latent states 作为 inputs_embeds
- **Regression Head**:
  - dense: 2560 → 2560 (ReLU/Tanh)
  - out_proj: 2560 → 1
  - sigmoid activation → 输出 [0, 1] 范围的 score

### Checkpoint 格式

PRM checkpoint 位于:
```
/data/chenhaotian/verl/checkpoints/latentqa/math_shepherd_20260116/best_model.pt
```

包含:
- `model_state_dict`: 包含 `backbone.xxx` 和 `regression_head.xxx`
- `config`: 训练配置（extract_layer=15, hidden_size=2560 等）

### 参考代码

- Latent states 提取: `/data/chenhaotian/latentqa/math_sheperd/preprocess_latent_states.py`
- PRM 训练: `/data/chenhaotian/latentqa/math_sheperd/train_math_shepherd_step1_scorer.py`

## 文件结构

```
recipe/prm_grpo/
├── __init__.py                  # 模块初始化
├── prm_model.py                 # PRM 模型定义（LatentExtractor, PRMBackbone, PRMScorer）
├── prm_reward_manager.py        # PRM Reward Manager + Step Logger
├── run_prm_grpo_qwen3_4b.sh     # 训练脚本
└── README.md                    # 本文档
```

## 使用方法

### 1. 准备工作

确保以下路径正确：
- 策略模型: `/data/models/Qwen/Qwen3-4B-Instruct-2507`
- PRM checkpoint: `/data/chenhaotian/verl/checkpoints/latentqa/math_shepherd_20260116/best_model.pt`
- 训练数据: `/data/chenhaotian/verl/data/train.parquet`
- 验证数据: `/data/chenhaotian/verl/data/test.parquet`

### 2. 运行训练

```bash
cd /data/chenhaotian/verl
bash recipe/prm_grpo/run_prm_grpo_qwen3_4b.sh
```

### 3. 关键配置参数

```bash
# Response 长度配置
MAX_RESPONSE_LENGTH=64   # 截断的思维链长度 (~40 tokens + buffer)
MAX_PROMPT_LENGTH=192    # Question 最大长度

# PRM 配置
PRM_MAX_LENGTH=256       # PRM 输入最大长度 (prompt + response)
PRM_EXTRACT_LAYER=15     # 从 Actor 的第几层提取 latent states

# Rollout 配置
N_ROLLOUTS=5             # 每个 question 的 rollout 数量

# 学习率
LEARNING_RATE=1e-6

# KL 正则化
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
```

## 日志记录

### Wandb 日志

训练过程中会记录以下 metrics 到 wandb：

**训练指标**:
- `train/loss`: 训练损失
- `train/accuracy`: 训练准确率
- `train/learning_rate`: 学习率
- `train/grad_norm`: 梯度范数
- `train/kl`: KL 散度

**验证指标**:
- `val/loss`: 验证损失
- `val/accuracy`: 验证准确率
- `val/prm_score_mean`: PRM 平均分数

**Reward 指标**:
- `reward/prm_score_mean`: PRM 分数均值
- `reward/prm_score_std`: PRM 分数标准差
- `reward/advantage_mean`: Advantage 均值
- `reward/advantage_std`: Advantage 标准差

### JSON 日志

每个训练 step 会保存详细数据到 JSON 文件:

```
logs/{experiment_name}/
├── step_000001.json    # 第 1 步的详细数据
├── step_000002.json    # 第 2 步的详细数据
├── ...
└── validation_step_000005.json  # 验证数据
```

JSON 文件包含:
- `step`: 当前步数
- `timestamp`: 时间戳
- `samples`: 样本详情（prompt, response, prm_score, ground_truth）
- `statistics`: 统计信息（mean, std, min, max, median）
- `extra_info`: 其他配置信息

## 配置示例

### 完整配置

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=512 \
    data.max_prompt_length=192 \
    data.max_response_length=64 \
    actor_rollout_ref.model.path="/data/models/Qwen/Qwen3-4B-Instruct-2507" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=5 \
    trainer.val_before_train=True \
    reward_manager.source=importlib \
    reward_manager.name=PRMRewardManager \
    reward_manager.module.path=/data/chenhaotian/verl/recipe/prm_grpo/prm_reward_manager.py \
    +reward_model.reward_kwargs.prm_checkpoint_path="/path/to/prm.pt" \
    +reward_model.reward_kwargs.prm_backbone_path="/path/to/backbone" \
    +reward_model.reward_kwargs.prm_extract_layer=15 \
    +reward_model.reward_kwargs.prm_max_length=256 \
    +reward_model.reward_kwargs.log_dir="/path/to/logs"
```

## 注意事项

1. **GPU 内存**: PRM 会额外加载一个 Latent Extractor（与 Actor 相同的模型），需要确保有足够的显存
2. **Response 长度**: `max_response_length` 设置为较小值（64），因为 PRM 评估的是截断的思维链开头
3. **验证**: 训练前会先进行一轮验证 (`trainer.val_before_train=True`)
4. **日志**: 所有 metrics 同时记录到 wandb 和 JSON 文件

## 引用

如果您使用此代码，请引用相关论文：

- GRPO: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Process Reward Models: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
