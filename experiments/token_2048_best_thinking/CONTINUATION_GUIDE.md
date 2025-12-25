# 续写思维过程指南

## 问题理解

你有一个 parquet 文件，其中包含**被中断的思考过程**（对话历史），你想让模型继续这个思维过程。

## 关键点

### ❌ **错误做法**：直接使用 `tokenizer.apply_chat_template`

如果你的数据已经是完整的对话列表格式，**不应该**再次包装它：

```python
# ❌ 错误！会导致嵌套
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],  # 这里 question 已经是列表了
    tokenize=False,
    add_generation_prompt=True
)
```

### ✅ **正确做法**：直接使用对话列表

你的数据格式是：
```python
[
  {"role": "user", "content": "问题..."},
  {"role": "assistant", "content": "未完成的回答..."}
]
```

这**已经是对话格式**了！你应该：

```python
# ✅ 正确！直接使用完整的对话历史
text = tokenizer.apply_chat_template(
    conversation,  # 直接传入整个对话列表
    tokenize=False,
    add_generation_prompt=True  # 关键：添加生成提示符
)
```

## 完整示例代码

```python
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 1. 读取数据
df = pd.read_parquet('outputs/stage3_temp_input.parquet')

# 2. 加载 tokenizer
model_path = "your_model_path"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 3. 处理每个对话
formatted_prompts = []
for conversation in df['question']:  # 'question' 列包含对话列表
    # conversation 格式:
    # [
    #   {"role": "user", "content": "..."},
    #   {"role": "assistant", "content": "未完成的回答..."}
    # ]
    
    # 直接使用完整的对话历史
    text = tokenizer.apply_chat_template(
        conversation,  # 传入整个对话列表
        tokenize=False,
        add_generation_prompt=True  # 这会添加助手继续生成的提示
    )
    formatted_prompts.append(text)

# 4. 使用 vLLM 生成
llm = LLM(model=model_path, trust_remote_code=True)
sampling_params = SamplingParams(
    n=8,  # 生成 8 个样本
    temperature=1.0,
    top_p=0.95,
    max_tokens=2048
)

outputs = llm.generate(formatted_prompts, sampling_params)

# 5. 提取结果
for output in outputs:
    completions = [o.text for o in output.outputs]
    print(completions)
```

## `add_generation_prompt=True` 的作用

这个参数非常重要！它会：

1. **保留完整的对话历史**（用户消息 + 助手的未完成回答）
2. **添加模型继续生成的提示符**

例如，对于 Qwen 模型，会生成类似：
```
<|im_start|>user
问题内容<|im_end|>
<|im_start|>assistant
未完成的回答... [← 模型会从这里继续]
```

没有 `add_generation_prompt=True`，模型不知道应该继续生成。

## 与 Stage1/Stage2 的区别

### Stage1（从头开始回答）
```python
# 只有用户问题，模型从零开始回答
conversation = [
    {"role": "user", "content": "问题"}
]
```

### Stage3（继续思考）
```python
# 有用户问题 + 部分回答，模型继续思考
conversation = [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "已经思考的部分..."}
]
```

## 验证你的提示词

你可以先打印出格式化后的文本，确保格式正确：

```python
# 取第一个样本验证
sample_conversation = df['question'].iloc[0]
formatted_text = tokenizer.apply_chat_template(
    sample_conversation,
    tokenize=False,
    add_generation_prompt=True
)

print("=" * 80)
print("格式化后的提示词:")
print("=" * 80)
print(formatted_text)
print("=" * 80)
```

你应该看到：
- ✅ 完整的用户问题
- ✅ 模型已经生成的部分回答
- ✅ 结尾有继续生成的提示符（如 `<|im_start|>assistant` 但没有 `<|im_end|>`）
- ❌ 不应该有嵌套的对话格式

## 总结

**关键点**：
1. 你的数据**已经是对话格式**（列表形式）
2. **直接传入**对话列表给 `apply_chat_template`
3. **必须设置** `add_generation_prompt=True`
4. **不要**重新包装成 `[{"role": "user", "content": ...}]`

这样模型就会自然地继续之前的思考过程！

