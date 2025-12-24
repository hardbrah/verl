# 实验原理详解

本文档深入解释实验的核心原理，帮助您理解每个步骤背后的技术细节。

## 目录
1. [核心概念](#核心概念)
2. [技术原理](#技术原理)
3. [实现细节](#实现细节)
4. [常见疑问](#常见疑问)

---

## 核心概念

### 什么是"最佳思考路径"？

在解决数学问题时，模型可能会尝试多种不同的思考方式。例如，对于同一个问题：

**问题**：证明 √2 是无理数

**思考路径1**（反证法）：
```
假设√2是有理数，可以表示为p/q（最简分数）...
[继续推导]
→ 得出矛盾，所以√2是无理数 ✓
```

**思考路径2**（直接证明）：
```
尝试找到√2的精确分数表示...
[尝试各种分数]
→ 无法找到，但这不能证明... ✗
```

不同的**起始思考方式**会导致不同的成功率。我们的实验就是要找到哪种起始思考最有可能成功。

### 什么是 token_2048？

`token_2048` 是模型思考的**前2048个token**（约1500-2000个中文字，或更多英文单词）。

**为什么是2048？**
- 足够长：能表达一个完整的思考方向
- 足够短：还没有得出最终答案（留下评估空间）
- 技术合理：符合标准的序列长度设置

### 核心假设

我们的实验基于以下假设：

> **假设1**：不同的起始思考路径质量不同  
> **假设2**：好的起始路径更容易导向正确答案  
> **假设3**：通过多次采样可以评估路径质量  

---

## 技术原理

### 原理1：如何实现"继续生成"

#### 传统方法的问题

在标准的LLM生成中：
```
输入：问题
输出：完整答案
```

如果我们想让模型"继续说下去"，不能简单地：
```python
# ❌ 错误方法
response_part1 = model.generate(prompt, max_tokens=2048)
response_part2 = model.generate(response_part1, max_tokens=2048)  # 模型会把part1当作新问题！
```

#### 我们的解决方案：Prompt重构

**核心思想**：构造一个"模型已经说了一半"的场景

```python
# ✓ 正确方法
# 步骤1：生成前2048个token
original_prompt = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
response_part1 = model.generate(original_prompt, max_tokens=2048)

# 步骤2：构造continuation prompt
continuation_prompt = (
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    f"{response_part1}"  # 已生成的部分
    # 注意：这里故意不加 <|im_end|>
)

# 步骤3：让模型继续生成
response_part2 = model.generate(continuation_prompt, max_tokens=4096)

# 完整答案
full_answer = response_part1 + response_part2
```

#### 为什么这样有效？

模型的训练数据格式是：
```
<|im_start|>user
{用户问题}<|im_end|>
<|im_start|>assistant
{完整回答}<|im_end|>
```

当我们提供一个**没有结束标记**的assistant回答时，模型会自然地认为：
- "哦，我已经说了这些话"
- "现在我需要继续说完"

这就像在对话中被打断，然后继续说下去。

### 原理2：准确率作为质量评估指标

#### 为什么用准确率？

对于一个token_2048，我们想知道：**从这个起点继续，能成功的概率有多大？**

这个概率无法直接计算，但可以**统计估计**：

```
token_2048_acc = (继续生成100次中成功的次数) / 100
```

#### 为什么是100次？

统计学原理：样本越大，估计越准确。

- 10次：误差约 ±30%
- 50次：误差约 ±14%
- 100次：误差约 ±10%
- 1000次：误差约 ±3%

100次是**效率和准确性的折衷**。

#### 为什么不直接看第一次生成是否正确？

单次生成包含太多随机性。考虑两个token_2048：

**Token_2048_A**（好）：
- 100次生成中，80次正确 → acc = 0.80

**Token_2048_B**（差）：
- 100次生成中，20次正确 → acc = 0.20

但第一次生成时：
- A 可能不幸失败（20%概率）
- B 可能幸运成功（20%概率）

**多次采样能消除随机性，揭示真实质量。**

### 原理3：为什么需要8个初始采样？

对于每个问题，我们生成8个不同的token_2048，因为：

1. **探索多样性**：同一个问题可能有多种思考方式
2. **找到最优**：8个中总有一个更好
3. **统计稳定**：避免单一采样的偶然性

**类比**：就像多次投篮，总能找到最佳手感。

---

## 实现细节

### 细节1：Chat Template的处理

#### Qwen模型的ChatML格式

```
<|im_start|>user
{用户消息}<|im_end|>
<|im_start|>assistant
{助手消息}<|im_end|>
```

#### 关键点

1. **`<|im_start|>` 和 `<|im_end|>`** 是特殊token，不是普通文本
2. **换行符** 是格式的一部分，不能省略
3. **assistant未结束** = 没有 `<|im_end|>` → 模型会继续生成

#### 在verl中的实现

```python
# 方法1：使用chat格式（阶段1）
prompt = [
    {"role": "user", "content": question},
]
# tokenizer.apply_chat_template会自动添加<|im_start|>assistant\n

# 方法2：使用continuation格式（阶段3）
continuation_prompt = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": token_2048}  # 已生成的部分
]
# tokenizer会生成带有未结束assistant的格式
```

### 细节2：与verl框架的集成

#### verl的main_generation工作流程

```
1. 读取parquet数据
2. 使用tokenizer.apply_chat_template处理prompt
3. 调用vLLM生成responses
4. 保存结果到parquet
```

#### 我们的适配策略

```
阶段1（标准生成）：
  输入：标准chat格式
  verl处理：正常流程
  输出：8个token_2048

阶段2（数据转换）：
  独立脚本：构造continuation prompts
  不调用verl：纯数据处理

阶段3（继续生成）：
  输入：continuation chat格式
  verl处理：tokenizer自动处理为正确格式
  输出：100个continuation
```

#### 为什么这样设计？

- **最小侵入**：不修改verl源码
- **最大复用**：使用verl的所有功能（分布式、vLLM、Ray等）
- **清晰分离**：我们的逻辑在外部，verl保持独立

### 细节3：数学答案验证

#### 使用verl的math_dapo模块

```python
from verl.utils.reward_score.math_dapo import compute_score

result = compute_score(
    solution_str=full_rollout,
    ground_truth=correct_answer
)

is_correct = result['acc']  # True or False
```

#### 验证流程

1. **提取答案**：从模型输出中找到 `\boxed{...}` 或 `Answer: ...`
2. **标准化**：去除单位、空格、LaTeX格式等
3. **比较**：与ground truth比较

#### 支持的答案格式

- LaTeX：`\boxed{42}`
- 自然语言：`Answer: 42`
- 表达式：`The answer is 42.`

---

## 常见疑问

### Q1: 为什么不用beam search？

**A**: Beam search寻找的是**单一最优路径**，而我们要研究**不同路径的质量分布**。

### Q2: token_2048会不会已经包含完整答案？

**A**: 可能会。如果已经包含完整答案：
- 继续生成时，模型可能：
  - 输出`<|im_end|>`（正常结束）
  - 重复之前的答案
  - 补充解释
- 验证时，我们检查完整rollout，所以不影响正确性判断

### Q3: 为什么不直接训练一个判别器？

**A**: 我们的目标是**研究**，不是**优化**：
- 研究：理解哪些思考路径更好
- 优化：提高模型性能

判别器是优化方法，但不能帮助我们理解"为什么"。

### Q4: 8个采样够吗？

**A**: 取决于问题复杂度：
- 简单问题：可能只有1-2种主要思路，8个够
- 复杂问题：可能有很多思路，8个是采样
- 我们不追求"穷举所有思路"，而是"找到一个好的"

### Q5: 如果两个token_2048准确率相同？

**A**: 随机选择一个（pandas的`idxmax()`会选第一个）。实际上，100次采样下，准确率完全相同的概率很小。

### Q6: 能用其他模型吗？

**A**: 可以！只需要：
1. 修改`model.path`
2. 确认模型的chat template格式
3. 如果不是ChatML格式，需要修改`stage2_prepare_data.py`中的`build_continuation_prompt`函数

### Q7: 能用其他数据集吗？

**A**: 可以！只需要：
1. 数据集有`question`和`answer`字段
2. 答案可以被`math_dapo.compute_score`验证
3. 修改`stage1_sample.py`中的数据加载部分

---

## 实验的学术意义

### 研究问题

1. **不同起始路径的质量差异有多大？**
   - 通过`acc_std`和`acc_range`度量

2. **选择最佳起始路径能带来多大提升？**
   - 通过`improvement_over_mean`度量

3. **哪些问题对起始路径更敏感？**
   - 通过分析高`acc_range`的问题

### 潜在应用

1. **训练数据筛选**：选择高质量的思考路径作为训练数据
2. **推理策略**：在实际应用中，先生成多个起始路径，选择最有希望的继续
3. **模型分析**：理解模型在哪些类型的问题上更稳定

---

## 进一步阅读

- **verl框架**: https://github.com/volcengine/verl
- **DAPO论文**: https://arxiv.org/abs/2410.07524
- **vLLM**: https://github.com/vllm-project/vllm
- **Qwen模型**: https://github.com/QwenLM/Qwen

---

## 总结

本实验通过：
1. **生成多样化的起始思考路径**（阶段1）
2. **评估每条路径的质量**（阶段3）
3. **选择最优路径**（阶段4）

来研究数学推理中**起始思考路径的重要性**。

核心技术：
- Prompt重构实现继续生成
- 统计采样评估路径质量
- 最小侵入式集成verl框架

**这是一个研究工具，不是生产系统。它的价值在于帮助我们理解和改进推理模型。**

