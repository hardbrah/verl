# Stage3 续写 - 快速参考

## 你的问题
> 这些都是模型被中断的思考过程，我想要模型接着这个思维过程思考，该怎样构造提示词？直接tokenizer.apply_chat_template就行了吗？

## 快速回答

### ✅ 可以直接用，但要注意用法！

```python
# ✅ 正确
formatted_text = tokenizer.apply_chat_template(
    conversation,              # 直接传入对话列表
    tokenize=False,
    add_generation_prompt=True  # 必须为 True！
)
```

---

## 可视化对比

### 数据流程

```
┌─────────────────────────────────────────────────────────┐
│ stage3_temp_input.parquet                                │
├─────────────────────────────────────────────────────────┤
│ question 列 (已经是对话列表格式)                         │
│                                                          │
│ [                                                        │
│   {                                                      │
│     "role": "user",                                      │
│     "content": "Solve the following math problem..."   │
│   },                                                     │
│   {                                                      │
│     "role": "assistant",                                 │
│     "content": "We are given that... [未完成]"          │
│   }                                                      │
│ ]                                                        │
└─────────────────────────────────────────────────────────┘
                    ↓
                    ↓ 直接传入
                    ↓
┌─────────────────────────────────────────────────────────┐
│ tokenizer.apply_chat_template(                          │
│     conversation,              ← 直接使用                │
│     tokenize=False,                                     │
│     add_generation_prompt=True ← 必须！                 │
│ )                                                        │
└─────────────────────────────────────────────────────────┘
                    ↓
                    ↓
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 格式化后的文本                                           │
├─────────────────────────────────────────────────────────┤
│ <|im_start|>user                                        │
│ Solve the following math problem...                     │
│ <|im_end|>                                              │
│ <|im_start|>assistant                                   │
│ We are given that... [未完成的内容]                     │
│                      ↑                                   │
│                      └─ 模型从这里继续生成               │
└─────────────────────────────────────────────────────────┘
```

---

## 对比：正确 vs 错误

### ❌ 错误方式

```python
# 错误 1: 重复包装
tokenizer.apply_chat_template(
    [{"role": "user", "content": conversation}],  # ❌ conversation 已经是列表
    tokenize=False,
    add_generation_prompt=True
)

# 错误 2: 忘记 add_generation_prompt
tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=False  # ❌ 模型不会继续生成
)

# 错误 3: 只传入最后的回答
tokenizer.apply_chat_template(
    [{"role": "assistant", "content": conversation[-1]['content']}],  # ❌ 丢失上下文
    tokenize=False,
    add_generation_prompt=True
)
```

### ✅ 正确方式

```python
# 正确：简单直接
tokenizer.apply_chat_template(
    conversation,              # ✅ 直接传入
    tokenize=False,
    add_generation_prompt=True  # ✅ 必须为 True
)
```

---

## Stage1 vs Stage3 对比

### Stage1（从头开始）

```python
# Stage1: 只有用户问题
conversation = [
    {"role": "user", "content": "问题"}
]

# 生成
text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
# 结果: 模型从头开始回答
```

### Stage3（续写思考）

```python
# Stage3: 有用户问题 + 部分回答
conversation = [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "已思考的部分..."}
]

# 生成
text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
# 结果: 模型继续之前的思考
```

---

## 3 个关键点

### 1️⃣ 你的数据已经是对话格式
- ✅ 不需要再构造
- ✅ 不需要再包装
- ✅ 直接使用即可

### 2️⃣ `add_generation_prompt=True` 是必须的
- ✅ 添加继续生成的提示符
- ✅ 让模型知道要继续
- ✅ 不加这个参数模型会停止

### 3️⃣ 保留完整的对话历史
- ✅ 包括用户问题
- ✅ 包括部分回答
- ✅ 模型需要完整上下文

---

## 验证命令

```bash
# 检查数据格式
python scripts/check_continuation_format.py

# 应该看到:
# ✅ 数据是列表格式
# ✅ 包含 2 轮对话
# ✅ 最后一轮是 assistant
```

---

## 总结

| 要点 | 答案 |
|------|------|
| 可以用 apply_chat_template 吗？ | ✅ 可以 |
| 需要特殊处理吗？ | ❌ 不需要 |
| 需要 add_generation_prompt=True 吗？ | ✅ 必须 |
| 需要重新包装数据吗？ | ❌ 不需要 |

**一句话总结**：直接用 `tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)` 就行！
